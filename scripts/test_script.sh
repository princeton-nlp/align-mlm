inputs=("../../bucket/henry_invert_data/ner/en/train_en_inv.json" "../../bucket/henry_invert_data/ner/en/dev_en_inv.json" "../../bucket/henry_invert_data/ner/en/train_en_inv.json" "../../bucket/henry_invert_data/ner/en/dev_en_inv.json" "../../bucket/henry_invert_data/ner/en/train_en_inv.json" "../../bucket/henry_invert_data/ner/en/dev_en_inv.json" "../../bucket/henry_invert_data/ner/en/train_en_inv.json" "../../bucket/henry_invert_data/ner/en/dev_en_inv.json")

pretrain_model="../../bucket/henry_model_outputs/en/transl_inv_en_500K/tlm"

outputs=("../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/orig1-normal" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/deriv1-normal" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/orig2-fixed" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/deriv2-fixed" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/orig3-normal" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/deriv3-normal" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/orig4-fixed" "../../bucket/henry_model_outputs/en/transl_inv_en_500K/ner_test/deriv4-fixed")

synthetic_info=("" "--one_to_one_mapping --word_modification replace --is_synthetic")

task=("1-normal" "2-fixed" "3-normal" "4-fixed")

run_name="transl_inv_en_500K_tlm"
state=("_orig" "_deriv")

#################################################### Finetuning ####################################################
for i in {0..7}
do
    python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/token-classification/run_ner_synthetic_test$((i/2+1)).py --learning_rate 2e-5 --save_steps -1 --task_name ner --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]} --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]} --model_name_or_path ${pretrain_model} ${synthetic_info[$((i%2))]}
done

#################################################### Zero-shot ####################################################
for i in {0..7}
do
    python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/token-classification/run_ner_synthetic_test$((i/2+1)).py --learning_rate 2e-5 --save_steps -1 --task_name ner --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*2))]} --validation_file ${inputs[$((i/2*2+1))]} --output_dir ${outputs[i]}_zero --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]}_zero --model_name_or_path ${outputs[i]} ${synthetic_info[1]}
done