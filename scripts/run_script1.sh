inputs=("../../bucket/supervised_data/xnli/en/train_en.json" "../../bucket/supervised_data/xnli/en/dev_en.json" "../../bucket/supervised_data/ner/en/train.json" "../../bucket/supervised_data/ner/en/dev.json" "../../bucket/supervised_data/pos/en/train-en.json" "../../bucket/supervised_data/pos/en/dev-en.json")

pretrain_model="../../bucket/henry_model_outputs/en/transl_en_500K/tlm"

outputs=("../../bucket/henry_model_outputs/en/transl_en_500K/xnli/orig" "../../bucket/henry_model_outputs/en/transl_en_500K/xnli/deriv"
"../../bucket/henry_model_outputs/en/transl_en_500K/ner/orig" "../../bucket/henry_model_outputs/en/transl_en_500K/ner/deriv" "../../bucket/henry_model_outputs/en/transl_en_500K/pos/orig" "../../bucket/henry_model_outputs/en/transl_en_500K/pos/deriv")

synthetic_info=("" "--one_to_one_mapping --word_modification replace --is_synthetic")

task=("xnli" "ner" "pos")

run_name="transl_en_500K_tlm"
state=("_orig" "_deriv")

#################################################### Finetuning ####################################################
for i in {0..5}
do
    python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]} --run_name ${run_name}_${task[$((i%2))]}${state[$((i/3))]} --model_name_or_path ${pretrain_model} ${synthetic_info[$((i%2))]}
done

#################################################### Zero-shot ####################################################
for i in {0..5}
do
    python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]}_zero --run_name ${run_name}_${task[$((i%2))]}${state[$((i/3))]}_zero --model_name_or_path ${outputs[i]} ${synthetic_info[1]}
done