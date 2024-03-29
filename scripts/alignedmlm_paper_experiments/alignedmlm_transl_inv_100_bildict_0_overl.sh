inputs=("../../../bucket/supervised_data/xnli/en/train_en.json" "../../../bucket/supervised_data/xnli/en/dev_en.json"
"../../../bucket/henry_invert_data/xnli/en/train_en_inv.json" "../../../bucket/henry_invert_data/xnli/en/dev_en_inv.json" 
"../../../bucket/supervised_data/ner/en/train.json" "../../../bucket/supervised_data/ner/en/dev.json"
"../../../bucket/henry_invert_data/ner/en/train_en_inv.json" "../../../bucket/henry_invert_data/ner/en/dev_en_inv.json" 
"../../../bucket/supervised_data/pos/en/train-en.json" "../../../bucket/supervised_data/pos/en/dev-en.json"
"../../../bucket/henry_invert_data/pos/en/train_en_inv.json" "../../../bucket/henry_invert_data/pos/en/dev_en_inv.json")

suffix="alignedmlm_100_bildict_0_overl"
base_dir="transl_inv_en_500K_$suffix"
config_name="config_alignedmlm.json"

pretrain_model="../../../bucket/henry_model_outputs/en/$base_dir/alignedmlm"

outputs=("../../../bucket/henry_model_outputs/en/$base_dir/xnli_$suffix/orig" "../../../bucket/henry_model_outputs/en/$base_dir/xnli_$suffix/deriv"
"../../../bucket/henry_model_outputs/en/$base_dir/ner_$suffix/orig" "../../../bucket/henry_model_outputs/en/$base_dir/ner_$suffix/deriv" "../../../bucket/henry_model_outputs/en/$base_dir/pos_$suffix/orig" "../../../bucket/henry_model_outputs/en/$base_dir/pos_$suffix/deriv")

synthetic_info=("" "--one_to_one_mapping --word_modification replace")

task=("xnli" "ner" "pos")

run_name="${base_dir}"
state=("_orig" "_deriv")

export WANDB_PROJECT=$run_name
echo $WANDB_PROJECT

#################################################### Pretraining ####################################################
python transformers/examples/xla_spawn.py --num_cores 8 transformers/examples/language-modeling/run_alignedmlm_synthetic_transitive.py --warmup_steps 10000 --learning_rate 1e-4 --save_steps -1 --max_seq_length 512 --logging_steps 50 --overwrite_output_dir --model_type roberta --config_name config/en/roberta_8/${config_name} --tokenizer_name config/en/roberta_8/ --do_train --do_eval --max_steps 500000 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file ../../../bucket/pretrain_data/en/train.txt --transitive_file ../../../bucket/henry_invert_data/pretrain/en/train_inv.txt --validation_file ../../../bucket/pretrain_data/en/valid.txt --output_dir ${pretrain_model} --run_name ${base_dir} --one_to_one_mapping --word_modification replace --alignment_loss_weight 10 --bilingual_rate 1.0 


#################################################### Finetuning ####################################################
for i in {0..5}
do
    if ((i < 2));
    then
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_train --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i*2))]} --validation_file ${inputs[$((i*2+1))]} --output_dir ${outputs[i]} --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]} --model_name_or_path ${pretrain_model} ${synthetic_info[$((i%2))]}
    else
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name ${task[$((i/2))]} --logging_steps 500 --overwrite_output_dir --do_train --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i*2))]} --validation_file ${inputs[$((i*2+1))]} --output_dir ${outputs[i]} --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]} --model_name_or_path ${pretrain_model} ${synthetic_info[$((i%2))]}
    fi
done

#################################################### Zero-shot ####################################################
for i in {0..5}
do
    if ((i < 2));
    then
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/text-classification/run_glue_synthetic.py --learning_rate 2e-5 --save_steps -1 --max_seq_length 128 --logging_steps 50 --overwrite_output_dir --do_eval --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*4+2))]} --validation_file ${inputs[$((i/2*4+3))]} --output_dir ${outputs[i]}_zero --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]}_zero --model_name_or_path ${outputs[i]} ${synthetic_info[1]}
    else 
        python transformers/examples/xla_spawn.py --num_cores 1 transformers/examples/token-classification/run_ner_synthetic.py --learning_rate 2e-5 --save_steps -1 --task_name ${task[$((i/2))]} --logging_steps 500 --overwrite_output_dir --do_eval --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --train_file ${inputs[$((i/2*4+2))]} --validation_file ${inputs[$((i/2*4+3))]} --output_dir ${outputs[i]}_zero --run_name ${run_name}_${task[$((i/2))]}${state[$((i%2))]}_zero --model_name_or_path ${outputs[i]} ${synthetic_info[1]}
    fi
done