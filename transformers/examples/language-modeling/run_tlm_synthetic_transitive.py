# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import numpy as np
import pdb

from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainerWordModifications,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

# Synthetic languages
from transformers.synthetic_utils import modify_config
from transformers.synthetic_utils import modify_inputs_synthetic

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    train_synthetic_file: Optional[str] = field(default=None, metadata={"help": "The syntheticinput training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_synthetic_file: Optional[str] = field(
        default=None,
        metadata={"help": "A synthetic validation file."},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache for training and validation data."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # Permute the vocabulary
    permute_vocabulary: bool = field(
        default=False,
        metadata={
            "help": "Whether to make the word based synthetic language which permutes the vocabulary."
        },
    )
    vocab_permutation_file: str = field(
        default=None,
        metadata={
            "help": "File which contains the mapping from the old vocabulary file to the new one. Global names are preferred"
        },
    )
    word_modification: str = field(
        default='all',
        metadata={
            "help": "all/random||add/replace"
        },
    )
    # Add, delete, or modify words
    modify_words: bool = field(
        default=False,
        metadata={
            "help": "Randomly replace words with a random word."
        },
    )
    modify_words_probability: float = field(
        default=0.15,
        metadata={
            "help": "The probability with which a word in the sentence needs to be replaced"
        },
    )
    modify_words_range: str = field(
        default='100-50000',
        metadata={
            "help": "Vocab range to sample from."
        },
    )
    # Invert the word-order
    invert_word_order: bool = field(
        default=False,
        metadata={
            "help": "Invert each sentence"
        },
    )
    # One-to-one mapping to a new vocabulary
    one_to_one_mapping: bool = field(
        default=False,
        metadata={
            "help": "Create a vocabulary with a one-to-one mapping with the new vocab, like in K. et al."
        },
    )
    one_to_one_file: str = field(
        default=None,
        metadata={
            "help": "File which contains indices in the vocabulary to ignore."
        },
    )
    shift_special: bool = field(
        default=False,
        metadata={
            "help": "When used with one-to-one mapping, also changes the [CLS] and [SEP] token. Does not change the PAD token."
        },
    )
    # Permutation
    permute_words: bool = field(
        default=False,
        metadata={
            "help": "Permute the words of the sentence randomly. Different permutation for each sentence."
        },
    )
    # Dataset ratio
    target_dataset_ratio: float = field(
        default=None,
        metadata={
            "help": "When less than one, use target_dataset_ratio * original dataset size."
        },
    )   
    # Ratio of TLM data generated during data generation
    tlm_generation_rate: float = field(
        default=1,
        metadata={
            "help": "Percentage of original sentences we use/sample to generate TLM data instances"
        }
    ) 

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.train_synthetic_file is not None:
            data_files["train_synthetic"] = data_args.train_synthetic_file
        if data_args.validation_synthetic_file is not None:
            data_files["validation_synthetic"] = data_args.validation_synthetic_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        ######################################### DATASET LOADED HERE #########################################
        datasets = load_dataset(extension, data_files=data_files)
        ######################################### DATASET LOADED HERE #########################################

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # pdb.set_trace()
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Make modifications to config if necessary
    config = modify_config(data_args, training_args, config)

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    if not data_args.one_to_one_mapping:
        # Don't resize the model token embeddings if we are making the one-to-one mapping modification
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warn(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            tokens_per_batch = data_args.max_seq_length
            pad_index = tokenizer.pad_token_id

            num_elements = len(examples['input_ids'])
            concatenated = {k: [] for k in examples.keys()}
            lengths = []

            for i in range(num_elements):
                if len(examples['input_ids'][i]) < tokens_per_batch:
                    for k in examples.keys():
                        concatenated[k].extend(examples[k][i])
                    lengths.append(len(examples['input_ids'][i]))
            
            num_pairs = len(lengths)
            lengths = np.array(lengths)

            indices = np.cumsum(lengths)
            # Append 0 to beginning of indices
            indices = np.insert(indices, 0, 0)

            result = {
                k: []
                for k, t in examples.items()
            }
            result['position_ids'] = []

            if np.shape(lengths)[0] == 0:
                return result

            cur_sum = 0
            prev_idx = 0
            for i in range(0, num_pairs+1):
                if i == num_pairs or cur_sum + lengths[i] > tokens_per_batch:
                    for k, t in concatenated.items():
                        # pdb.set_trace()
                        if k != 'attention_mask':
                            data = [pad_index for _ in range(tokens_per_batch)]
                        else:
                            data = [0 for _ in range(tokens_per_batch)] #0 means no attention

                        # We are assuming both things are of the same length
                        distance = indices[i] - indices[prev_idx]

                        data[0:distance] = t[indices[prev_idx]:indices[i]]
                        result[k].append(data)
                    
                    prev_idx = i
                    cur_sum = 0
                
                if i != num_pairs:
                    cur_sum += lengths[i]

            # pdb.set_trace()
            an_array = np.array(result['input_ids'])
            mask = np.not_equal(an_array, tokenizer.pad_token_id)
            incremental_indices = np.cumsum(mask, axis=1) * mask
            result['position_ids'] = (incremental_indices.astype(int) + tokenizer.pad_token_id).tolist()

            return result

        def make_tlm(examples):
            ### IMPORTANT PARAMETER ###
            selected_indices = np.random.choice(len(examples['input_ids']), int(data_args.tlm_generation_rate*len(examples['input_ids'])), replace=False)
            selected_indices = selected_indices.tolist()
            selected_indices.sort()

            # print("STOP!")
            # pdb.set_trace()

            tmp_examples = {
                k: [] for k in examples.keys()
            }
            for k in examples.keys():
                tmp_examples[k] = [examples[k][index] for index in selected_indices]
            
            examples = deepcopy(tmp_examples)

            tokens_per_batch = data_args.max_seq_length
            pad_index = tokenizer.pad_token_id

            num_elements = len(examples['input_ids'])
            concatenated = {k: [] for k in examples.keys()}
            lengths1 = []
            lengths2 = []

            # pdb.set_trace()
            for i in range(num_elements):
                if len(examples['input_ids'][i]) < tokens_per_batch//2 and len(examples['input_ids_syn'][i]) < tokens_per_batch//2:
                    for k in examples.keys():
                        concatenated[k].extend(examples[k][i])
                    lengths1.append(len(examples['input_ids'][i]))
                    lengths2.append(len(examples['input_ids_syn'][i]))
            
            num_pairs = len(lengths1)
            num_tokens = len(concatenated['input_ids'])

            lengths1 = np.array(lengths1)
            lengths2 = np.array(lengths2)

            indices1 = np.cumsum(lengths1)
            indices2 = np.cumsum(lengths2)
            # Append 0 to beginning of indices
            indices1 = np.insert(indices1, 0, 0)
            indices2 = np.insert(indices2, 0, 0)

            result = {
                k: []
                for k in col_names
            }
            result['position_ids'] = []

            if np.shape(lengths1)[0] == 0 or np.shape(lengths2)[0] == 0:
                # pdb.set_trace()
                return result

            # pdb.set_trace()

            cur_sum1 = 0
            cur_sum2 = 0
            prev_idx = 0
            for i in range(0, num_pairs+1):
                if i == num_pairs or cur_sum1 + lengths1[i] > tokens_per_batch//2 or cur_sum2 + lengths2[i] > tokens_per_batch//2:
                    # for k, t in concatenated.items():
                    for k in col_names:
                        # pdb.set_trace()
                        if k != 'attention_mask':
                            data = [pad_index for _ in range(tokens_per_batch)]
                        else:
                            data = [0 for _ in range(tokens_per_batch)] #0 means no attention
                        
                        # We are assuming both things are of the same length
                        distance1 = indices1[i] - indices1[prev_idx]
                        distance2 = indices2[i] - indices2[prev_idx]

                        assert distance1 <= tokens_per_batch//2
                        assert distance2 <= tokens_per_batch//2

                        # pdb.set_trace()
                        data[0:distance1] = concatenated[k][indices1[prev_idx]:indices1[i]]
                        data[tokens_per_batch//2:tokens_per_batch//2+distance2] = concatenated[f"{k}_syn"][indices2[prev_idx]:indices2[i]]
                        result[k].append(data)
                    
                    prev_idx = i
                    cur_sum1 = 0
                    cur_sum2 = 0
                
                if i != num_pairs:
                    cur_sum1 += lengths1[i]
                    cur_sum2 += lengths2[i]
        
            # pdb.set_trace()
            input_array = np.array(result['input_ids'])

            array1 = input_array[:,0:tokens_per_batch//2]
            mask1 = np.not_equal(array1, tokenizer.pad_token_id)
            incremental_indices1 = np.cumsum(mask1, axis=1) * mask1
            positions1 = incremental_indices1.astype(int) + tokenizer.pad_token_id

            array2 = input_array[:,tokens_per_batch//2:]
            mask2 = np.not_equal(array2, tokenizer.pad_token_id)
            incremental_indices2 = np.cumsum(mask2, axis=1) * mask2
            positions2 = incremental_indices2.astype(int) + tokenizer.pad_token_id
            result['position_ids'] = np.concatenate((positions1, positions2), axis = 1).tolist()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        lang1_id = tokenizer.pad_token_id+1
        lang2_id = tokenizer.pad_token_id+2

        data_args.word_modification = 'replace'
        for key in tokenized_datasets.keys():
            if key == 'train_synthetic' or key == 'validation_synthetic':
                tokenized_datasets[key] = modify_inputs_synthetic(data_args, training_args, tokenized_datasets[key], tokenizer=tokenizer)
                # modify_inputs_one_to_one_mapping(data_args, training_args, tokenized_datasets[key], "Transliteration", tokenizer)

        # pdb.set_trace()
        for key in tokenized_datasets.keys():
            nrows = len(tokenized_datasets[key]['input_ids'])
            temp_data = tokenized_datasets[key]['input_ids']
            if key == 'train' or key == 'validation':
                tokenized_datasets[key] = tokenized_datasets[key].add_column(
                    "lang_type_ids",
                    [[lang1_id for _ in range(len(temp_data[i]))] for i in range(nrows)]
                )
            else:
                tokenized_datasets[key] = tokenized_datasets[key].add_column(
                    "lang_type_ids",
                    [[lang2_id for _ in range(len(temp_data[i]))] for i in range(nrows)]
                )
        

        assert len(tokenized_datasets['train']['input_ids']) == len(tokenized_datasets['train_synthetic']['input_ids'])
        assert len(tokenized_datasets['validation']['input_ids']) == len(tokenized_datasets['validation_synthetic']['input_ids'])
        # pdb.set_trace()

        col_names = tokenized_datasets['train'].column_names
        col_names_syn = [f"{c}_syn" for c in col_names]
        # pdb.set_trace()
        tlm_datasets = {'train': deepcopy(tokenized_datasets['train']), 'validation': deepcopy(tokenized_datasets['validation'])}

        # pdb.set_trace()
        for key in tlm_datasets.keys():
            # key = 'train' or 'validation'
            for col in col_names:
                tlm_datasets[key] = tlm_datasets[key].add_column(
                    f"{col}_syn",
                    deepcopy(tokenized_datasets[f"{key}_synthetic"][col])
                )

        # pdb.set_trace()

        for k in tlm_datasets:
            # pdb.set_trace()
            tlm_datasets[k] = tlm_datasets[k].map(
                make_tlm,
                batched=True,
                remove_columns=col_names_syn,
                # batch_size=None,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        
        # pdb.set_trace()

    # Make synthetic language modifications if necessary
    # tokenized_datasets = modify_inputs_synthetic(data_args, training_args, tokenized_datasets, tokenizer=tokenizer)
    ######################################### Modified #########################################
    # tokenized_datasets_new = modify_inputs_synthetic(data_args, training_args, tokenized_datasets["transitive_file"], tokenizer=tokenizer)
    ######################################### Modified #########################################
    for k in tokenized_datasets:
        tokenized_datasets[k] = tokenized_datasets[k].map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # pdb.set_trace()

    # Combine the two datasets
    tokenized_datasets = {'train': concatenate_datasets([tokenized_datasets['train'], tokenized_datasets['train_synthetic'], tlm_datasets['train']]), 
    'validation': concatenate_datasets([tokenized_datasets['validation'], tokenized_datasets['validation_synthetic'], tlm_datasets['validation']])}

    # pdb.set_trace()
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # Initialize our Trainer
    trainer = TrainerWordModifications(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
