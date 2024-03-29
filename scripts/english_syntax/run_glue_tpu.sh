export TASK_NAME=mnli
export RUN_NO=_english_syntax

python ../../transformers/examples/xla_spawn.py --num_cores 1 ../../transformers/examples/text-classification/run_glue_synthetic.py \
  --model_name_or_path ../../../../bucket/model_outputs/syntax_modif/english/english_synth_en_fr_hi_120/ \
  --cache_dir=../../../../bucket/cache \
  --train_file ../../../../bucket/syntax_modified_data/english/en_hi_N_hi_V/GLUE/MNLI/train.json \
  --validation_file ../../../../bucket/syntax_modified_data/english/en_hi_N_hi_V/GLUE/MNLI/mono_dep_flattened_dev_matched-en~hi\@N~hi\@V.json \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --output_dir ../../../../bucket/model_outputs/glue/$TASK_NAME$RUN_NO/ \
  --run_name glue_$TASK_NAME \