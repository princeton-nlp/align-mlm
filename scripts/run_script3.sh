inputs=("../../bucket/henry_invert_data/xnli/en/train_en_inv.json" "../../bucket/henry_invert_data/xnli/en/dev_en_inv.json" "../../bucket/henry_invert_data/ner/en/train_en_inv.json" "../../bucket/henry_invert_data/ner/en/dev_en_inv.json" "../../bucket/henry_invert_data/pos/en/train_en_inv.json" "../../bucket/henry_invert_data/pos/en/dev_en_inv.json")

pretrain_model="../../bucket/henry_model_outputs/en/transl_inv_en_500K/tlm"

outputs=("../../bucket/henry_model_outputs/en/transl_inv_en_500K/xnli/orig" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/xnli/deriv"
"../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner/orig" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner/deriv" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/pos/orig" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/pos/deriv")

synthetic_info=("" "--one_to_one_mapping --word_modification replace --is_synthetic")

task=("xnli" "ner" "pos")

run_name="transl_inv_en_500K"
state=("_orig" "_deriv")

#################################################### Finetuning ####################################################
for i in {0..5}
do
    if ((i < 2));
    then
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]} --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]} --model_name_or_path ${pretrain_model} ${synthetic_info[$((i%2))]}
    else 
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/token-classification/run_ner_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --task_name ${task[$((i/2))]} --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]} --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]} --model_name_or_path ${pretrain_model} ${synthetic_info[$((i%2))]}
    fi
done

#################################################### Zero-shot ####################################################
for i in {0..5}
do
    if ((i < 2));
    then
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]}_zero --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]}_zero --model_name_or_path ${outputs[i]} ${synthetic_info[1]}
    else 
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/token-classification/run_ner_synthetic_transitive.py --learning_rate 2e-5 --save_steps -1 --task_name ${task[$((i/2))]} --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]}_zero --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]}_zero --model_name_or_path ${outputs[i]} ${synthetic_info[1]}
    fi
done