#################################################### Pretraining ####################################################
# Transliteration 100K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 100000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --validation_file ../../bucket/pretrain_data/en/valid.txt --output_dir ../../bucket/henry_model_outputs/en/transliteration_100K/tlm --run_name one_to_one_en_100K_tlm --one_to_one_mapping --word_modification add --tlm_sample_rate 0.5 --tlm_generation_rate 1.0 &

# Transliteration, Invert 100K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 100000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --validation_file ../../bucket/pretrain_data/en/valid.txt --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/tlm --run_name one_to_one_invert_en_100K_tlm --one_to_one_mapping --invert_word_order --word_modification add --tlm_sample_rate 0.5 --tlm_generation_rate 1.0 &

# Transliteration 500K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --validation_file ../../bucket/pretrain_data/en/valid.txt --output_dir ../../bucket/henry_model_outputs/en/transliteration_500K/tlm --run_name one_to_one_en_500K_tlm --one_to_one_mapping --word_modification add --tlm_sample_rate 0.5 --tlm_generation_rate 1.0 &

# Transliteration, Invert 500K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --validation_file ../../bucket/pretrain_data/en/valid.txt --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_500K/tlm --run_name one_to_one_invert_en_500K_tlm --one_to_one_mapping --invert_word_order --word_modification add --tlm_sample_rate 0.5 --tlm_generation_rate 1.0 &



python transformers/examples/language-modeling/run_tlm_synthetic.py --warmup_steps 1000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 50 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 5000 --per_device_train_batch_size 6 --per_device_eval_batch_size 16 --train_file data/train.txt --validation_file data/valid.txt --output_dir data/test_run --run_name adroit_test --invert_word_order --word_modification add --tlm_sample_rate 0.5 --tlm_generation_rate 0.3



--max_steps 100000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --validation_file ../../bucket/pretrain_data/en/valid.txt --output_dir ../../bucket/henry_model_outputs/en/transliteration_100K/tlm --run_name inverted_en_500K_mlm --one_to_one_mapping --word_modification add &



--tlm_sample_rate 0.5 --tlm_generation_rate 1.0



#################################################### Finetuning ####################################################
# Transliteration 100K
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_100K/xnli --run_name one_to_one_en_100K_xnli --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_100K/tlm &

# Transliteration, Invert 100K
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/xnli --run_name one_to_one_invert_en_100K_xnli --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/tlm &

# Transliteration 500K - orig
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_500K/xnli_orig --run_name one_to_one_en_500K_xnli_orig --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_500K/tlm &

# Transliteration 500K - deriv
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_500K/xnli_deriv --run_name one_to_one_en_500K_xnli_deriv --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_500K/tlm --one_to_one_mapping --word_modification replace &

# Transliteration, Invert 500K - orig
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_500K/xnli_orig --run_name one_to_one_invert_en_500K_xnli_orig --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_inverted_order_500K/tlm &

# Transliteration, Invert 500K - deriv
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_500K/xnli_deriv --run_name one_to_one_invert_en_500K_xnli_deriv --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_inverted_order_500K/tlm --one_to_one_mapping --word_modification replace &





python transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file files/supervised_data_xnli_en_train_en.json --validation_file files/supervised_data_xnli_en_dev_en.json --output_dir data/finetune_dir --run_name finetune_test --model_name_or_path data/pretrain_dir



#################################################### Zero-shot ####################################################
# Transliteration 100K
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_100K/xnli_zero --run_name one_to_one_en_100K_xnli_zero --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_100K/xnli --one_to_one_mapping --word_modification replace &

# Transliteration, Invert 100K
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/xnli_zero --run_name one_to_one_invert_en_100K_xnli_zero --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/xnli --one_to_one_mapping --invert_word_order --word_modification replace &

# Transliteration 500K - orig
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_500K/xnli_orig_zero --run_name one_to_one_en_500K_xnli_orig_zero --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_500K/xnli_orig --one_to_one_mapping --word_modification replace &

# Transliteration 500K - deriv
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_500K/xnli_deriv_zero --run_name one_to_one_en_500K_xnli_deriv_zero --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_500K/xnli_deriv --one_to_one_mapping --word_modification replace &

python transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file files/small_train_en.json --validation_file files/small_dev_en.json --output_dir data/zeroshot_dir --run_name zero-shot_test --model_name_or_path data/finetune_dir --one_to_one_mapping --word_modification replace



# Syntax Transformation
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_mlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 50 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --transitive_file ../../bucket/analysis_data/en/en_syntax_fr_train.txt --validation_file ../../bucket/analysis_data/en/en_syntax_fr_valid.txt --output_dir ../../bucket/model_outputs/en/syntax_en_fr_one_to_one/mlm --run_name syntax_en_fr_one_to_one --one_to_one_mapping --word_modification add &


nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_mlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 50 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --transitive_file ../../bucket/analysis_data/en/en_syntax_fr_train.txt --validation_file ../../bucket/analysis_data/en/en_syntax_fr_valid.txt --output_dir ../../bucket/model_outputs/en/syntax_en_fr_one_to_one/mlm --run_name syntax_en_fr_one_to_one &





#################################################### Transitive #####################################################
#################################################### Pretraining ####################################################
# Transliteration, 500K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/pretrain_data/en/train.txt --train_synthetic_file ../../bucket/pretrain_data/en/train.txt --validation_file ../../bucket/pretrain_data/en/valid.txt --validation_synthetic_file ../../bucket/pretrain_data/en/valid.txt --output_dir ../../bucket/henry_model_outputs/en/transl_en_500K/tlm --run_name transl_en_500K_tlm --one_to_one_mapping --word_modification replace &

# Transliteration + Syntax French, 500K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/henry_syntax_data/en/mono_dep_train-en~fr@N~fr@V.txt --train_synthetic_file ../../bucket/henry_syntax_data/en/synthetic_dep_train-en~fr@N~fr@V.txt --validation_file ../../bucket/henry_syntax_data/en/mono_dep_valid-en~fr@N~fr@V.txt --validation_synthetic_file ../../bucket/henry_syntax_data/en/synthetic_dep_valid-en~fr@N~fr@V.txt --output_dir ../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/tlm --run_name transl_synt_en_fr_500K_tlm --one_to_one_mapping --word_modification replace &

# Transliteration + Syntax Arabic, 500K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/henry_syntax_data/en/mono_dep_train-en~ar@N~ar@V.txt --train_synthetic_file ../../bucket/henry_syntax_data/en/synthetic_dep_train-en~ar@N~ar@V.txt --validation_file ../../bucket/henry_syntax_data/en/mono_dep_valid-en~ar@N~ar@V.txt --validation_synthetic_file ../../bucket/henry_syntax_data/en/synthetic_dep_valid-en~ar@N~ar@V.txt --output_dir ../../bucket/henry_model_outputs/en/transl_synt_en_ar_500K/tlm --run_name transl_synt_en_ar_500K_tlm --one_to_one_mapping --word_modification replace &

# Transliteration + Invert, 500K
nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/henry_invert_data/pretrain/en/train_orig.txt --train_synthetic_file ../../bucket/henry_invert_data/pretrain/en/train_inv.txt --validation_file ../../bucket/henry_invert_data/pretrain/en/valid_orig.txt --validation_synthetic_file ../../bucket/henry_invert_data/pretrain/en/valid_inv.txt --output_dir ../../bucket/henry_model_outputs/en/transl_inv_en_500K/tlm --run_name transl_inv_en_500K_tlm --one_to_one_mapping --word_modification replace &


#################################################### XNLI Finetuning ####################################################
# Transliteration, 500K - XNLI, orig
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transl_en_500K/xnli/orig --run_name one_to_one_en_500K_xnli_orig --model_name_or_path ../../bucket/henry_model_outputs/en/transl_en_500K/tlm &


# Transliteration, 500K - XNLI, deriv
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_500K/xnli/deriv --run_name one_to_one_en_500K_xnli_deriv --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_500K/tlm --one_to_one_mapping --word_modification replace --is_synthetic &


python transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 100 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file files/small_train_en.json --validation_file files/small_dev_en.json --output_dir data/finetune_dir --run_name finetune_test --model_name_or_path data/pretrain_dir --one_to_one_mapping --word_modification replace --is_synthetic



# python transformers/examples/language-modeling/run_tlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file sample_data/valid_orig_full.txt --train_synthetic_file sample_data/valid_inv_full.txt --validation_file sample_data/valid_orig_full.txt --validation_synthetic_file sample_data/valid_inv_full.txt --output_dir data/test --run_name transl_inv_en_500K_tlm --one_to_one_mapping --word_modification replace




# nohup python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_tlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 100 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/config_tlm.json --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 5000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../bucket/henry_syntax_data/en/mono_dep_train-en~fr@N~fr@V.txt --train_synthetic_file ../../bucket/henry_syntax_data/en/mono_dep_train-en~fr@N~fr@V.txt --validation_file ../../bucket/henry_syntax_data/en/mono_dep_valid-en~fr@N~fr@V.txt --validation_synthetic_file ../../bucket/henry_syntax_data/en/mono_dep_valid-en~fr@N~fr@V.txt --output_dir ../../bucket/henry_model_outputs/en/transl_en_500K/tlm --run_name transl_en_500K_tlm --one_to_one_mapping --word_modification replace &







# ALL FILES
../../bucket/henry_syntax_data/en/mono_dep_train-en~fr@N~fr@V.txt
../../bucket/henry_syntax_data/en/synthetic_dep_train-en~fr@N~fr@V.txt
../../bucket/henry_syntax_data/en/mono_dep_valid-en~fr@N~fr@V.txt
../../bucket/henry_syntax_data/en/synthetic_dep_valid-en~fr@N~fr@V.txt

# ../../bucket/henry_syntax_data/en/mono_dep_train-en~ar@N~ar@V.txt
# ../../bucket/henry_syntax_data/en/synthetic_dep_train-en~ar@N~ar@V.txt
# ../../bucket/henry_syntax_data/en/mono_dep_valid-en~ar@N~ar@V.txt
# ../../bucket/henry_syntax_data/en/synthetic_dep_valid-en~ar@N~ar@V.txt

../../bucket/pretrain_data/en/train.txt
../../bucket/pretrain_data/en/valid.txt

../../bucket/henry_invert_data/pretrain/en/train_orig.txt
../../bucket/henry_invert_data/pretrain/en/train_inv.txt
../../bucket/henry_invert_data/pretrain/en/valid_orig.txt
../../bucket/henry_invert_data/pretrain/en/valid_inv.txt

../../bucket/henry_model_outputs/en/transl_en_500K/tlm
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/tlm
../../bucket/henry_model_outputs/en/transl_inv_en_500K/tlm


# XNLI
../../bucket/supervised_data/xnli/en/train_en.json
../../bucket/supervised_data/xnli/en/dev_en.json

../../bucket/supervised_data/xnli/en/dep/synthetic_dep_flattened_train_en-en~fr@N~fr@V.json
../../bucket/supervised_data/xnli/en/dep/synthetic_dep_flattened_dev_en-en~fr@N~fr@V.json

# ../../bucket/supervised_data/xnli/en/dep/synthetic_dep_flattened_train_en-en~ar@N~ar@V.json
# ../../bucket/supervised_data/xnli/en/dep/synthetic_dep_flattened_dev_en-en~ar@N~ar@V.json

../../bucket/henry_invert_data/xnli/en/train_en_orig.json
../../bucket/henry_invert_data/xnli/en/train_en_inv.json
../../bucket/henry_invert_data/xnli/en/dev_en_orig.json
../../bucket/henry_invert_data/xnli/en/dev_en_inv.json

../../bucket/henry_model_outputs/en/transl_en_500K/xnli/orig
../../bucket/henry_model_outputs/en/transl_en_500K/xnli/deriv
../../bucket/henry_model_outputs/en/transl_inv_en_500K/xnli/orig
../../bucket/henry_model_outputs/en/transl_inv_en_500K/xnli/deriv
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/xnli/orig
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/xnli/deriv


# NER
../../bucket/supervised_data/ner/en/train.json
../../bucket/supervised_data/ner/en/dev.json

../../bucket/supervised_data/ner/en/dep/synthetic_dep_flattened_train-en~fr@N~fr@V.json
../../bucket/supervised_data/ner/en/dep/synthetic_dep_flattened_dev-en~fr@N~fr@V.json

# ../../bucket/supervised_data/ner/en/dep/synthetic_dep_flattened_train-en~ar@N~ar@V.json
# ../../bucket/supervised_data/ner/en/dep/synthetic_dep_flattened_dev-en~ar@N~ar@V.json

../../bucket/henry_invert_data/ner/en/train_en_orig.json
../../bucket/henry_invert_data/ner/en/train_en_inv.json
../../bucket/henry_invert_data/ner/en/dev_en_orig.json
../../bucket/henry_invert_data/ner/en/dev_en_inv.json

../../bucket/henry_model_outputs/en/transl_en_500K/ner/orig
../../bucket/henry_model_outputs/en/transl_en_500K/ner/deriv
../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner/orig
../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner/deriv
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/ner/orig
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/ner/deriv

# POS
../../bucket/supervised_data/pos/en/train-en.json
../../bucket/supervised_data/pos/en/dev-en.json

../../bucket/supervised_data/pos/en/dep/synthetic_dep_flattened_train-en-en~fr@N~fr@V.json
../../bucket/supervised_data/pos/en/dep/synthetic_dep_flattened_dev-en-en~fr@N~fr@V.json

# ../../bucket/supervised_data/pos/en/dep/synthetic_dep_flattened_train-en-en~ar@N~ar@V.json
# ../../bucket/supervised_data/pos/en/dep/synthetic_dep_flattened_dev-en-en~ar@N~ar@V.json

../../bucket/henry_invert_data/pos/en/train_en_orig.json
../../bucket/henry_invert_data/pos/en/train_en_inv.json
../../bucket/henry_invert_data/pos/en/dev_en_orig.json
../../bucket/henry_invert_data/pos/en/dev_en_inv.json

../../bucket/henry_model_outputs/en/transl_en_500K/pos/orig
../../bucket/henry_model_outputs/en/transl_en_500K/pos/deriv
../../bucket/henry_model_outputs/en/transl_inv_en_500K/pos/orig
../../bucket/henry_model_outputs/en/transl_inv_en_500K/pos/deriv
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/pos/orig
../../bucket/henry_model_outputs/en/transl_synt_en_fr_500K/pos/deriv