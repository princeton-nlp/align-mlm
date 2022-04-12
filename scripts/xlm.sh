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


python transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file files/supervised_data_xnli_en_train_en.json --validation_file files/supervised_data_xnli_en_dev_en.json --output_dir data/finetune_dir --run_name finetune_test --model_name_or_path data/pretrain_dir



#################################################### Zero-shot ####################################################
# Transliteration 100K
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_100K/xnli_zero --run_name one_to_one_en_100K_xnli_zero --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_100K/xnli --one_to_one_mapping --word_modification replace &

# Transliteration, Invert 100K
nohup python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ../../bucket/supervised_data/xnli/en/train_en.json --validation_file ../../bucket/supervised_data/xnli/en/dev_en.json --output_dir ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/xnli_zero --run_name one_to_one_invert_en_100K_xnli_zero --model_name_or_path ../../bucket/henry_model_outputs/en/transliteration_inverted_order_100K/xnli --one_to_one_mapping --invert_word_order --word_modification replace &


python transformers/examples/text-classification/run_glue_synthetic_tlm.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file files/small_train_en.json --validation_file files/small_dev_en.json --output_dir data/zeroshot_dir --run_name zero-shot_test --model_name_or_path data/finetune_dir --one_to_one_mapping --word_modification replace
