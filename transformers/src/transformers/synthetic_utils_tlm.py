"""
Class and function definitions for word-based modifications
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import concatenate_datasets
from copy import deepcopy
import random
import pdb
from datasets import DatasetDict
import math

def create_position_ids_from_input_ids(input_ids, padding_idx):
    an_array = np.array(input_ids)
    mask = np.not_equal(an_array, padding_idx)
    incremental_indices = np.cumsum(mask, axis=1) * mask
    return (incremental_indices.astype(int) + padding_idx).tolist()

def split_inputs(examples, tlm_generation_rate):
    half_length = len(examples['input_ids'][0])//2
    num_sentences = len(examples['input_ids'])
    num_choices = math.ceil(num_sentences*tlm_generation_rate)
    select_indices = np.random.choice(num_sentences, num_choices, False)

    examples1 = {}
    examples2 = {}
    syn_examples = {}
    for key in examples.keys():
        if key != 'tlm_data':
            examples1[key] = [deepcopy(examples[key][j][:half_length]) for j in select_indices]
            examples2[key] = [deepcopy(examples[key][j][half_length:]) for j in select_indices]
        syn_examples[key] = deepcopy(examples[key])

    orig_examples1 = deepcopy(examples1)
    orig_examples2 = deepcopy(examples2)

    examples1['lang_type_ids'] = [[1 for _ in range(half_length)] for i in range(num_choices)]
    examples2['lang_type_ids'] = [[1 for _ in range(half_length)] for i in range(num_choices)]

    return syn_examples, orig_examples1, orig_examples2, examples1, examples2, half_length, num_choices

def merge_inputs(syn_examples, orig_examples1, orig_examples2, examples1, examples2, half_length, num_sentences, pad_token_id):

    num_rows = len(syn_examples['input_ids'])
    num_cols = len(syn_examples['input_ids'][0])

    syn_examples['position_ids'] = create_position_ids_from_input_ids(syn_examples['input_ids'], pad_token_id)
    syn_examples["tlm_data"] = [[0] for i in range(num_rows)]
    syn_examples['lang_type_ids'] = [[1 for _ in range(num_cols)] for i in range(num_rows)]

    if num_sentences > 0:
        orig_examples1['position_ids'] = create_position_ids_from_input_ids(orig_examples1['input_ids'], pad_token_id)
        orig_examples2['position_ids'] = create_position_ids_from_input_ids(orig_examples2['input_ids'], pad_token_id)
        examples1['position_ids'] = create_position_ids_from_input_ids(examples1['input_ids'], pad_token_id)
        examples2['position_ids'] = create_position_ids_from_input_ids(examples2['input_ids'], pad_token_id)

        for key in examples1.keys():
            for j in range(num_sentences):
                orig_examples1[key][j] += deepcopy(examples1[key][j])
                orig_examples2[key][j] += deepcopy(examples2[key][j])
    
        for key in orig_examples1.keys():
            orig_examples1[key] += deepcopy(orig_examples2[key])
        
        num_rows = len(orig_examples1['input_ids'])
        orig_examples1["tlm_data"] = [[1] for i in range(num_rows)]

        for key in syn_examples.keys():
            syn_examples[key] += deepcopy(orig_examples1[key])

    return syn_examples

def add_lang_and_pos(datasets, pad_token_id):
    if type(datasets) is dict or (type(datasets) is DatasetDict):
        for key in datasets.keys():
            ncols = len(datasets[key]['input_ids'][0])
            nrows = len(datasets[key]['input_ids'])
            datasets[key] = datasets[key].add_column(
                "lang_type_ids",
                [[0 for _ in range(ncols)] for i in range(nrows)]
            )
            datasets[key] = datasets[key].add_column(
                "position_ids",
                create_position_ids_from_input_ids(datasets[key]['input_ids'], pad_token_id)
            )
    else:
        ncols = len(datasets['input_ids'][0])
        nrows = len(datasets['input_ids'])
        datasets.add_column(
            "lang_type_ids",
            [[0 for _ in range(ncols)] for i in range(nrows)]
        )
        datasets.add_column(
            "position_ids",
            create_position_ids_from_input_ids(datasets['input_ids'], pad_token_id)
        )

def add_tlm_data_label(datasets):
    if type(datasets) is dict or (type(datasets) is DatasetDict):
        for key in datasets.keys():
            num_rows = len(datasets[key]['input_ids'])
            datasets[key] = datasets[key].add_column("tlm_data", [[0] for i in range(num_rows)])
    else:
        datasets = datasets.add_column("tlm_data", [[0] for i in range(num_rows)])

def create_modified_dataset(data_args, map_function, datasets):
    # # Create new dataset using map function
    # modified_dataset = datasets.map(
    #     map_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers
    # )
    
    if type(datasets) is dict or (type(datasets) is DatasetDict):
        modified_dataset = {}
        for key in datasets.keys():
            modified_dataset[key] = datasets[key].map(
                map_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers
            )
    else:
        modified_dataset = datasets.map(
            map_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers
        )
    
    # pdb.set_trace()
    # Step 3: Check if a modified dataset needs to be ADDED or if it should be REPLACED
    columns=['input_ids', 'attention_mask', 'special_tokens_mask', 'lang_type_ids', 'position_ids']

    # pdb.set_trace()
    if data_args.word_modification == 'add':
        # Check if there are multiple datasets or if it's a single dataset
        if 'keys' in dir(datasets):
            # Concatenate the two datasets
            combined_dataset = {}

            for key in datasets.keys():
                keys = list(datasets[key].features.keys())
                tensor_keys = [x for x in columns if x in keys]
                combined_dataset[key] = concatenate_datasets([datasets[key], modified_dataset[key]])
                # combined_dataset[key].set_format(type='torch', columns=tensor_keys)
            
            # pdb.set_trace()
            # for k in datasets[key].features.keys():
            # pdb.set_trace()
            return combined_dataset
        else:
            keys = list(datasets.features.keys())
            tensor_keys = [x for x in columns if x in keys]
            combined_dataset = concatenate_datasets([datasets, modified_dataset])
            # combined_dataset.set_format(type='torch', columns=tensor_keys)
            return combined_dataset

    elif data_args.word_modification == 'replace':
        # Check if there are multiple datasets or if it's a single dataset
        # pdb.set_trace()
        if 'keys' in dir(datasets):
            replaced_dataset = {}

            for key in modified_dataset.keys():
                keys = list(datasets[key].features.keys())
                tensor_keys = [x for x in columns if x in keys]
                replaced_dataset[key] = modified_dataset[key]
                # replaced_dataset[key].set_format(type='torch', columns=tensor_keys)

            return replaced_dataset
        else:
            keys = list(datasets.features.keys())
            tensor_keys = [x for x in columns if x in keys]
            replaced_dataset = modified_dataset
            # replaced_dataset.set_format(type='torch', columns=tensor_keys)
            return replaced_dataset


# Can ignore this one
def modify_inputs_permute(data_args, training_args, datasets, task_name):
    # Step 1: Load the vocab mapping
    # Function for modifying string json to integer json
    # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
    def jsonKV2int(x):
        if isinstance(x, dict):
                return {int(k):(int(v) if isinstance(v, str) else v) for k,v in x.items()}
        return x
    
    # Load the the vocabulary file
    if data_args.permute_vocabulary:
        with open(data_args.vocab_permutation_file, 'r') as fp:
            vocab_mapping = json.load(fp, object_hook=jsonKV2int)
    
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # Step 2: Create a modified dataset
    # Map function for datasets.map
    def map_function(examples):
        num_rows = len(examples['input_ids'])
        num_cols = len(examples['input_ids'][0])
        for j in range(num_rows):
            examples['input_ids'][j] = [vocab_mapping[examples['input_ids'][j][i]] for i in range(num_cols)]
        return examples

    # Step 3: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)
        
# Perturbation (not in the paper) - ignore for now
def modify_inputs_words(data_args, training_args, datasets, task_name, tokenizer=None):
    # Get the sampling range for modifying the words
    sampling_range = [int(i) for i in data_args.modify_words_range.strip().split('-')]

    # Make sure the upper bound is lesser than the tokenizer length
    sampling_range[1] = min(sampling_range[1], len(tokenizer))

    # SEP, CLS, or PAD tokens
    sep_cls_pad = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token), tokenizer.convert_tokens_to_ids(tokenizer.pad_token)]

    # Step 1: Create map function for modification
    def map_function(examples):
        num_rows = len(examples['input_ids'])
        num_cols = len(examples['input_ids'][0])
        for j in range(num_rows):
            # examples['input_ids'][j] = [examples['input_ids'][j][i] for i in range(len(examples['input_ids'][j])) if np.random.binomial(data_args.modify_words_probability) == 0 else np.random.randint(low=sampling_range[0], high=sampling_range[1])]
            examples['input_ids'][j] = [examples['input_ids'][j][i] if (np.random.binomial(1, data_args.modify_words_probability) == 0 or (not sampling_range[0] <= examples['input_ids'][j][i] <= sampling_range[1]) or examples['input_ids'][j][i] in sep_cls_pad) else np.random.randint(low=sampling_range[0], high=sampling_range[1]) for i in range(num_cols)]
        return examples

    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)

# Just for QA (inversion)
# TODO: Still need to implement this
def modify_inputs_invert_qa(data_args, training_args, datasets, task_name, tokenizer=None, negative_label=None):
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # If the task is QA, then call a different function

    # TODO: </s> index is hard coded here. Pull it from the tokenizer instead.
    # TODO: pad_index is hard coded here. Pull it from the tokenizer instead.
    
    def map_function(examples):
        def reverse_list(s):
            s.reverse()
            return s
        
        def reverse_substr(sent_indices):
            temp_sent_indices = deepcopy(sent_indices)
            start_idx = 0
            current_idx = 0

            # Create a list with numbers from 0 to len(sent_indices)
            number_indices = list(range(len(sent_indices)))
            temp_number_indices = list(range(len(sent_indices)))

            sentence_length = len(sent_indices)

            # Indices to consider for [SEP] and [CLS]
            if tokenizer:
                sep_cls = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
            else:
                # If tokenizer is not passed, then use the default RoBERTa tokenizer tokens
                sep_cls = [0, 2]

            for i in range(sentence_length):
                if (sent_indices[i] in sep_cls) or (i == (sentence_length - 1)):
                    # flip sentence on start_idx, i
                    if i > start_idx:
                        if (i == (sentence_length - 1)) and (not (sent_indices[i] in sep_cls)):
                            sent_indices[start_idx: i+1] = reverse_list(temp_sent_indices[start_idx: i+1])
                            number_indices[start_idx: i+1] = reverse_list(temp_number_indices[start_idx: i+1])
                        else:
                            sent_indices[start_idx: i] = reverse_list(temp_sent_indices[start_idx: i])
                            number_indices[start_idx: i] = reverse_list(temp_number_indices[start_idx: i])
                    start_idx = i+1
            return sent_indices, number_indices

        num_rows = len(examples['input_ids'])
        for j in range(num_rows):
            example_length = len(examples['input_ids'][j])
            modified_input_ids, number_indices = reverse_substr(examples['input_ids'][j])

            # Since it's a QA task, make modifications to other keys before `input_ids`
            # Train set
            if 'start_positions' in examples:
                temp = examples['start_positions'][j]
                examples['start_positions'][j] = number_indices[examples['end_positions'][j]]
                examples['end_positions'][j] = number_indices[temp]

            # Validation set
            if 'offset_mapping' in examples:
                examples['offset_mapping'][j] = [examples['offset_mapping'][j][idx] for idx in number_indices]

            # Modify the inputs
            examples['input_ids'][j] = [modified_input_ids[i] for i in range(example_length)]

        return examples

    # Step 2: Return modified dataset
    return create_modified_dataset(data_args, map_function, datasets)


# This is the inversion
def modify_inputs_invert(data_args, training_args, datasets, task_name, examples1, examples2, syn_examples, tokenizer=None, negative_label=None):
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # If the task is QA, then call a different function
    if task_name == 'qa':
        return modify_inputs_invert_qa(data_args, training_args, datasets, task_name, tokenizer, negative_label)

    # TODO: </s> index is hard coded here. Pull it from the tokenizer instead.
    # TODO: pad_index is hard coded here. Pull it from the tokenizer instead.
    
    def reverse_list(s):
        s.reverse()
        return s
    # Reverse function for NER/POS labels
    def reverse_substr_ner_pos(sent_indices):
        temp_sent_indices = deepcopy(sent_indices)

        sentence_length = len(sent_indices)

        sent_indices[1:-1] = reverse_list(temp_sent_indices[1:-1])

        """
        # Now, if the labels were ['O', 'B-PER', 'I-PER'], they are modified to ['I-PER', 'B-PER', 'O']
        # Change it to ['B-PER', 'I-PER', 'O']
        # negative_label is the label index corresponding to 'O'
        if negative_label and task_name == 'ner':
            negative_labels = [negative_label]
            temp_sent_indices = deepcopy(sent_indices)
            start_idx = -1

            for i in range(sentence_length):
                if sent_indices[i] not in negative_labels:
                    # Check if this is the first occurrence of an entity tag
                    if start_idx < 0:
                        start_idx = i
                        # If this is the last token in the sentence, then we don't have reverse it
                    elif  start_idx >= 0 and (i == (sentence_length - 1)):
                        # If it's the last token of the sentence and it is not 'O', then flip
                        sent_indices[start_idx: i+1] = reverse_list(temp_sent_indices[start_idx: i+1])
                else:
                    if start_idx > 0:
                        # Flip the labels
                        sent_indices[start_idx: i] = reverse_list(temp_sent_indices[start_idx: i])
                        start_idx = -1
        """
            
        return sent_indices
    
    def reverse_substr(sent_indices):
        temp_sent_indices = deepcopy(sent_indices)
        start_idx = 0
        current_idx = 0

        sentence_length = len(sent_indices)

        # Indices to consider for [SEP] and [CLS]
        if tokenizer:
            sep_cls = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
        else:
            # If tokenizer is not passed, then use the default RoBERTa tokenizer tokens
            sep_cls = [0, 2]

        for i in range(sentence_length):
            if (sent_indices[i] in sep_cls) or (i == (sentence_length - 1)):
                # flip sentence on start_idx, i
                if i > start_idx:
                    if (i == (sentence_length - 1)) and (not (sent_indices[i] in sep_cls)):
                        sent_indices[start_idx: i+1] = reverse_list(temp_sent_indices[start_idx: i+1])
                    else:
                        sent_indices[start_idx: i] = reverse_list(temp_sent_indices[start_idx: i])
                start_idx = i+1
        return sent_indices

    def perform_inversion(data):
        num_rows = len(data['input_ids'])
        for j in range(num_rows):
            l = len(data['input_ids'][j])
            modified_examples = reverse_substr(data['input_ids'][j])
            data['input_ids'][j] = [modified_examples[i] for i in range(l)]
            # If it's a token classification task, flip the labels too
            if task_name in ['ner', 'pos']:
                modified_labels = reverse_substr_ner_pos(data['labels'][j])
                data['labels'][j] = [modified_labels[i] for i in range(l)]

    perform_inversion(syn_examples)
    perform_inversion(examples1)
    perform_inversion(examples2)
    return    

# Transliteration
def modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, examples1, examples2, syn_examples, tokenizer):
    # Should we modify special tokens? That is contained in boolean data_args.shift_special
    if data_args.shift_special:
        special_tokens = [tokenizer.pad_token_id]
    else:
        special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

    # Vocabulary size
    vocab_size = tokenizer.vocab_size
    # If we are replacing only a fraction of the words, create a list
    # We use the same variable data_args.modify_words_probability here

    if data_args.one_to_one_file is not None:
        dont_modify = np.load(open(data_args.one_to_one_file, 'rb'))

        # Step 1: Create map function for modification
        def one_to_one_mapping(data):
            num_rows = len(data['input_ids'])
            num_cols = len(data['input_ids'][0])
            for j in range(num_rows):
                data['input_ids'][j] = [data['input_ids'][j][i] if (data['input_ids'][j][i] in special_tokens or data['input_ids'][j][i] in dont_modify) else (data['input_ids'][j][i] + vocab_size)  for i in range(num_cols)]

        one_to_one_mapping(syn_examples)
        one_to_one_mapping(examples1)
        one_to_one_mapping(examples2)
        return

    else:
        # Step 1: Create map function for modification
        def one_to_one_mapping(data):
            num_rows = len(data['input_ids'])
            num_cols = len(data['input_ids'][0])
            for j in range(num_rows):
                data['input_ids'][j] = [data['input_ids'][j][i] if (data['input_ids'][j][i] in special_tokens) else (data['input_ids'][j][i] + vocab_size)  for i in range(num_cols)]

        # pdb.set_trace()
        one_to_one_mapping(syn_examples)
        one_to_one_mapping(examples1)
        one_to_one_mapping(examples2)
        return

# Permutation
def modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, examples1, examples2, syn_examples, tokenizer=None, negative_label=None):
    # Check the arguments
    assert data_args.word_modification == 'add' or data_args.word_modification == 'replace', "Illegal option for argument word_modification"

    # This modification doesn't work for qa
    assert task_name != 'qa', "Permutation doesn't work for QA."
    
    # def map_function(examples):
    def permute_list(sent, labels=None):
        if labels is not None:
            sent, labels = zip(*random.sample(list(zip(sent, labels)), len(sent)))
            return sent, labels
        else:
            sent = random.sample(sent, len(sent))
            return sent
    
    def permute_substr(sent_indices, sent_labels=None):
        temp_sent_indices = deepcopy(sent_indices)
        temp_sent_labels = deepcopy(sent_labels) if sent_labels is not None else None
        start_idx = 0
        current_idx = 0

        sentence_length = len(sent_indices)

        # Indices to consider for [SEP] and [CLS]
        if tokenizer:
            sep_cls = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
        else:
            # If tokenizer is not passed, then use the default RoBERTa tokenizer tokens
            sep_cls = [0, 2]

        for i in range(sentence_length):
            if (sent_indices[i] in sep_cls) or (i == (sentence_length - 1)):
                # flip sentence on start_idx, i
                if i > start_idx:
                    if (i == (sentence_length - 1)) and (not (sent_indices[i] in sep_cls)):
                        if sent_labels is not None:
                            sent_indices[start_idx: i+1], sent_labels[start_idx: i+1] = permute_list(temp_sent_indices[start_idx: i+1], labels=temp_sent_labels[start_idx: i+1])
                        else:
                            sent_indices[start_idx: i+1] = permute_list(temp_sent_indices[start_idx: i+1])
                    else:
                        if sent_labels is not None:
                            sent_indices[start_idx: i], sent_labels[start_idx: i] = permute_list(temp_sent_indices[start_idx: i], labels=temp_sent_labels[start_idx: i])
                        else:
                            sent_indices[start_idx: i] = permute_list(temp_sent_indices[start_idx: i])
                start_idx = i+1
        if sent_labels is not None:
            return sent_indices, sent_labels
        else:
            return sent_indices
    
    def perform_permutation(data):
        num_rows = len(data['input_ids'])
        for j in range(num_rows):
            l = len(data['input_ids'][j])
            if task_name in ['ner', 'pos']:
                modified_examples, modified_labels = permute_substr(data['input_ids'][j], sent_labels=data['labels'][j])
                data['input_ids'][j] = [modified_examples[i] for i in range(l)]
                data['labels'][j] = [modified_labels[i] for i in range(l)]
            else:
                modified_examples = permute_substr(data['input_ids'][j])
                data['input_ids'][j] = [modified_examples[i] for i in range(l)]

    perform_permutation(examples1)
    perform_permutation(examples2)
    perform_permutation(syn_examples)
    return

def apply_modifications(data_args, training_args, datasets, task_name, tokenizer=None, negative_label=None):
    # pdb.set_trace()
    def map_function(examples):
        nonlocal data_args, training_args, datasets, task_name, tokenizer, negative_label

        # pdb.set_trace()
        syn_examples, orig_examples1, orig_examples2, examples1, examples2, half_length, num_sentences = split_inputs(examples, data_args.tlm_generation_rate)

        has_mod = False
        if data_args.permute_vocabulary:
            # datasets = modify_inputs_permute(data_args, training_args, datasets, task_name)
            raise NotImplementedError("We don't use permute_vocabulary transformation")
            has_mod = True
        if data_args.modify_words:
            # datasets = modify_inputs_words(data_args, training_args, datasets, task_name, tokenizer)
            raise NotImplementedError("We don't use modify_words transformation")
            has_mod = True
        if data_args.invert_word_order:
            datasets = modify_inputs_invert(data_args, training_args, datasets, task_name, examples1, examples2, syn_examples, tokenizer)
            has_mod = True
        if data_args.one_to_one_mapping:
            datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, examples1, examples2, syn_examples, tokenizer)
            has_mod = True
        if data_args.permute_words:
            datasets = modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, examples1, examples2, syn_examples, tokenizer)
            has_mod = True

        if not has_mod:
            raise NotImplementedError("Must have a transformation for TLM pretraining!")

        return merge_inputs(syn_examples, orig_examples1, orig_examples2, examples1, examples2, half_length, num_sentences, tokenizer.pad_token_id)

    return create_modified_dataset(data_args, map_function, datasets)


def modify_inputs_synthetic(data_args, training_args, datasets, task_name=None, task_type='tlm', tokenizer=None):
    add_lang_and_pos(datasets, tokenizer.pad_token_id)
    add_tlm_data_label(datasets)

    if task_type in ['glue', 'xnli', 'ner', 'pos', 'qa', 'tatoeba']:
        data_args.preprocessing_num_workers = None
    
    # # If multiple word modifications are being performed, then handle them separately
    # if data_args.one_to_one_mapping and data_args.invert_word_order:
    #     original_datasets = deepcopy(datasets)
    #     original_word_modification = data_args.word_modification
    #     data_args.word_modification = 'replace'
    #     datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer)
    #     datasets = modify_inputs_invert(data_args, training_args, datasets, task_name, tokenizer)

    #     if original_word_modification == 'replace':
    #         return datasets
    #     elif original_word_modification == 'add':
    #         if 'keys' in dir(datasets):
    #             # Concatenate the two datasets
    #             combined_dataset = {}

    #             for key in datasets.keys():
    #                 combined_dataset[key] = concatenate_datasets([original_datasets[key], datasets[key]])
                
    #             return combined_dataset

    # # If multiple word modifications are being performed, then handle them separately
    # if data_args.one_to_one_mapping and data_args.permute_words:
    #     original_datasets = deepcopy(datasets)
    #     original_word_modification = data_args.word_modification
    #     data_args.word_modification = 'replace'
    #     datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer)
    #     datasets = modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, tokenizer)

    #     if original_word_modification == 'replace':
    #         return datasets
    #     elif original_word_modification == 'add':
    #         if 'keys' in dir(datasets):
    #             # Concatenate the two datasets
    #             combined_dataset = {}

    #             for key in datasets.keys():
    #                 combined_dataset[key] = concatenate_datasets([original_datasets[key], datasets[key]])
                
    #             return combined_dataset

    # If we need to sample only a part of the dataset, handle it separately
    if 'target_dataset_ratio' in dir(data_args) and data_args.target_dataset_ratio is not None:
        original_datasets = deepcopy(datasets)
        original_word_modification = data_args.word_modification
        data_args.word_modification = 'replace'

    # have a separate function called modify_inputs
    # splits the inputs, then calls each of the modification functions in turn
    # finally, merges them together and calls create_modified_dataset
    # if data_args.permute_vocabulary:
    #     datasets = modify_inputs_permute(data_args, training_args, datasets, task_name)
    # if data_args.modify_words:
    #     datasets = modify_inputs_words(data_args, training_args, datasets, task_name, tokenizer)
    # if data_args.invert_word_order:
    #     datasets = modify_inputs_invert(data_args, training_args, datasets, task_name, tokenizer)
    # if data_args.one_to_one_mapping:
    #     datasets = modify_inputs_one_to_one_mapping(data_args, training_args, datasets, task_name, tokenizer)
    # if data_args.permute_words:
    #     datasets = modify_inputs_permute_sentence(data_args, training_args, datasets, task_name, tokenizer)
    # pdb.set_trace()
    datasets = apply_modifications(data_args, training_args, datasets, task_name, tokenizer)

    # If we need to sample only a part of the dataset, handle it separately
    if 'target_dataset_ratio' in dir(data_args) and data_args.target_dataset_ratio is not None:
        # Subsample the original dataset
        for key in datasets.keys():
            if key == 'train':
                num_ele = len(datasets[key])
                select_indices = random.sample(range(num_ele), int(data_args.target_dataset_ratio * num_ele))
                datasets[key] = datasets[key].select(select_indices)
        
        # Combine with original datasets
        if original_word_modification == 'replace':
            return datasets
        elif original_word_modification == 'add':
            if 'keys' in dir(datasets):
                # Concatenate the two datasets
                combined_dataset = {}

                for key in datasets.keys():
                    combined_dataset[key] = concatenate_datasets([original_datasets[key], datasets[key]])
                
                return combined_dataset   

    return datasets

def modify_config(data_args, training_args, config):
    if data_args.one_to_one_mapping:
        config.vocab_size = config.vocab_size * 2
        return config
    else:
        return config