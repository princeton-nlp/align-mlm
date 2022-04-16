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

from datasets import load_dataset
import pdb
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
    TrainerWordModificationsTLM,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

# Synthetic languages
from transformers.synthetic_utils_tlm import modify_inputs_synthetic
from transformers.synthetic_utils_tlm import modify_config

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
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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

    # Ratio of sampling TLM data during training
    tlm_sample_rate: float = field(
        default=0.3,
        metadata={
            "help": "Percentage of time we sample a TLM token stream. For pure TLM, set to 1."
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
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)

    # pdb.set_trace()
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

        # pdb.set_trace()
        # datasets & tokenized_datasets is a DatasetDict
        # <class 'datasets.dataset_dict.DatasetDict'>
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        # pdb.set_trace()
        # for i in tokenized_datasets['train']['input_ids'][0]:
        #     print(tokenizer.decode([i]))
        # asdf = tokenizer.decode(tokenized_datasets['train']['input_ids'][2])
        # pdb.set_trace()

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
        # def group_texts(examples):
        #     # Concatenate all texts.
        #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        #     total_length = len(concatenated_examples[list(examples.keys())[0]])
        #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        #     # customize this part to your needs.
        #     total_length = (total_length // max_seq_length) * max_seq_length
        #     # Split by chunks of max_len.
        #     result = {
        #         k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        #         for k, t in concatenated_examples.items()
        #     }
        #     return result

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            # for k in examples.keys():
            #     print(k)
            # pdb.set_trace()
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        # pdb.set_trace()

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # pdb.set_trace()
    # tokenized datasets has become a dict after this point
    # Train and test are of type <class 'datasets.arrow_dataset.Dataset'>
    # BEFORE ->
    # DatasetDict({
    #     train: Dataset({
    #         features: ['input_ids', 'attention_mask', 'special_tokens_mask'],
    #         num_rows: 4
    #     })
    #     validation: Dataset({
    #         features: ['input_ids', 'attention_mask', 'special_tokens_mask'],
    #         num_rows: 1
    #     })
    # })

    # AFTER ->
    # {'train': Dataset({
    #     features: ['input_ids', 'attention_mask', 'special_tokens_mask'],
    #     num_rows: 8
    # }), 'validation': Dataset({
    #     features: ['input_ids', 'attention_mask', 'special_tokens_mask'],
    #     num_rows: 2
    # })}

    # tokenized_datasets["train"]['input_ids'] -> list of size (1, 512)

    # DataTrainingArguments(dataset_name=None, dataset_config_name=None, train_file='data/train.txt', validation_file='data/valid.txt', data_cache_dir=None, overwrite_cache=False, validation_split_percentage=5, max_seq_length=512, preprocessing_num_workers=None, mlm_probability=0.15, line_by_line=False, pad_to_max_length=False, permute_vocabulary=False, vocab_permutation_file=None, word_modification='add', modify_words=False, modify_words_probability=0.15, modify_words_range='100-50000', invert_word_order=True, one_to_one_mapping=False, one_to_one_file=None, shift_special=False, permute_words=False, target_dataset_ratio=None)
    #TrainingArguments(output_dir=data/output_dir, overwrite_output_dir=True, do_train=True, do_eval=True, do_predict=False, model_parallel=False, evaluation_strategy=EvaluationStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=0.0001, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=5000, warmup_steps=1000, logging_dir=runs/Feb06_00-38-27_adroit5, logging_first_step=False, logging_steps=50, save_steps=-1, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=50, dataloader_num_workers=0, past_index=-1, run_name=adroit_test, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, fp16_backend=auto, sharded_ddp=False)
    # Make synthetic language modifications if necessary
    tokenized_datasets = modify_inputs_synthetic(data_args, training_args, tokenized_datasets, tokenizer=tokenizer)
    # pdb.set_trace()

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    
    # print(training_args)
    # Initialize our Trainer
    trainer = TrainerWordModificationsTLM(
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
