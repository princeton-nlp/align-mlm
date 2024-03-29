#!/bin/bash
# Author: Ameet Deshpande

# Define some global variables
LANG="en"
TARGET="fr"
EVAL="_zero"
MODEL_DIR="syntax_en_fr_ratio_1_10"
MODEL="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/mlm"
OUTPUT_DIR="../../../../bucket/model_outputs/${LANG}/${MODEL_DIR}/"
RUN_NAME="${LANG}_${MODEL_DIR}_"
ZERO_SHOT_ADD=""

##### XNLI #####
TASK='xnli'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dev_$LANG.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

# Zero-shot eval
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/xnli/$LANG/train_$LANG.json --validation_file ../../../../bucket/supervised_data/xnli/$LANG/dep/synthetic_dep_flattened_dev_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD


##### NER #####
TASK='ner'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dev.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

# Zero-shot eval
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name $TASK --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/$TASK/$LANG/train.json --validation_file ../../../../bucket/supervised_data/$TASK/$LANG/dep/synthetic_dep_flattened_dev-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD

##### POS #####
TASK='pos'

# Train
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dev-$LANG.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK --model_name_or_path $MODEL

# Zero-shot eval
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name pos --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../../../bucket/supervised_data/pos/$LANG/train-$LANG.json --validation_file ../../../../bucket/supervised_data/pos/$LANG/dep/synthetic_dep_flattened_dev-$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK$EVAL --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $OUTPUT_DIR$TASK $ZERO_SHOT_ADD

##### TATOEBA #####
TASK='tatoeba'

# Eval
python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/sentence_retrieval/run_sentence_retrieval_synthetic.py --max_seq_length 128 --pool_type middle --logging_steps 50 --overwrite_output_dir --do_train --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --bilingual --train_file ../../../../bucket/supervised_data/tatoeba/$LANG/dep/synthetic_dep_flattened_$LANG-$LANG~$TARGET\@N~$TARGET\@V.json --output_dir $OUTPUT_DIR$TASK --run_name $RUN_NAME$TASK$EVAL --model_name_or_path $MODEL $ZERO_SHOT_ADD