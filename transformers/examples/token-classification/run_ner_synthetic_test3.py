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
This file is to be used only for inverted order and permutation NER Evaluation.
Use run_ner_synthetic.py for everything else.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import copy
import pdb

import numpy as np
from datasets import ClassLabel, load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerWordModifications,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

# Synthetic languages		
from transformers import modify_inputs_synthetic		
from transformers.synthetic_utils import modify_config


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache for training and validation data."},
    )    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
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
    # Dataset is in a synthetic language	
    is_synthetic: bool = field(	
        default=False,	
        metadata={	
            "help": "True if the dataset is in a synthetic (as opposed to the original base) language."	
        },	
    ) 

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


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
    set_seed(training_args.seed+2)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        # pdb.set_trace()
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        # pdb.set_trace()
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        an_array = np.array(tokenized_inputs["input_ids"])
        mask = np.not_equal(an_array, tokenizer.pad_token_id)
        incremental_indices = np.cumsum(mask, axis=1) * mask
        positions = incremental_indices.astype(int) + tokenizer.pad_token_id

        tokenized_inputs["position_ids"] = positions.tolist()

        nrows = len(tokenized_inputs['input_ids'])
        ncols = len(tokenized_inputs["input_ids"][0])
        lang_labels = np.full((nrows, ncols), tokenizer.pad_token_id)
        lang_id = tokenizer.pad_token_id+1 if (data_args.is_synthetic == False) else tokenizer.pad_token_id+2
        lang_labels[mask] = lang_id
        tokenized_inputs["lang_type_ids"] = lang_labels.tolist()
        return tokenized_inputs

    # pdb.set_trace()

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # pdb.set_trace()

    # Make synthetic language modifications if necessary
    tokenized_datasets = modify_inputs_synthetic(data_args, training_args, tokenized_datasets, tokenizer=tokenizer, task_name=data_args.task_name, task_type=data_args.task_name)
    
    def make_labels_consistent(examples):
        inv_label_to_id = {v: k for k, v in label_to_id.items()}
        B_labels = [label_to_id[key] for key in label_to_id.keys() if 'B' in key]
        I_labels = [label_to_id[key] for key in label_to_id.keys() if 'I' in key]
        B_to_I = dict([(label_to_id['B-'+suffix.split('-')[-1]], label_to_id['I-'+suffix.split('-')[-1]]) for suffix in label_to_id.keys() if 'B' in suffix])
        I_to_B = dict([(label_to_id['I-'+suffix.split('-')[-1]], label_to_id['B-'+suffix.split('-')[-1]]) for suffix in label_to_id.keys() if 'B' in suffix])
        for i in range(len(examples['labels'])):
            new_labels = copy.deepcopy(examples['labels'][i])
            flag = False
            prev_suffix = None
            for j in range(len(new_labels)):
                if examples['labels'][i][j] < 0:
                    continue
                if examples['labels'][i][j] not in I_labels and examples['labels'][i][j] not in B_labels:
                    flag = False
                    prev_suffix = None
                elif not flag and examples['labels'][i][j] in I_labels:
                    examples['labels'][i][j] = I_to_B[examples['labels'][i][j]]
                    flag = True
                    prev_suffix = inv_label_to_id[examples['labels'][i][j]].split('-')[-1]
                elif not flag and examples['labels'][i][j] in B_labels:
                    flag = True
                    prev_suffix = inv_label_to_id[examples['labels'][i][j]].split('-')[-1]
                elif flag and examples['labels'][i][j] in B_labels and prev_suffix == inv_label_to_id[examples['labels'][i][j]].split('-')[-1]:
                    examples['labels'][i][j] = B_to_I[examples['labels'][i][j]]
                elif flag and examples['labels'][i][j] in B_labels and prev_suffix != inv_label_to_id[examples['labels'][i][j]].split('-')[-1]:
                    prev_suffix = inv_label_to_id[examples['labels'][i][j]].split('-')[-1]
                elif flag and examples['labels'][i][j] in I_labels and prev_suffix != inv_label_to_id[examples['labels'][i][j]].split('-')[-1]:
                    examples['labels'][i][j] = I_to_B[examples['labels'][i][j]]
                    prev_suffix = inv_label_to_id[examples['labels'][i][j]].split('-')[-1]                    
        return examples

    # pdb.set_trace()
    # Make sure the NER labels are consistent
    # pdb.set_trace()
    # if data_args.task_name == "ner":
    #     # pdb.set_trace()
    #     for key in tokenized_datasets.keys():
    #         tokenized_datasets[key] = tokenized_datasets[key].map(
    #             make_labels_consistent,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #         )
    # pdb.set_trace()
    # pdb.set_trace()

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    # Initialize our Trainer
    trainer = TrainerWordModifications(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_ner.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_dataset = tokenized_datasets["test"]
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
