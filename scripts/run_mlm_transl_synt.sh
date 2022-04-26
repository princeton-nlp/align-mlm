inputs=("../../bucket/supervised_data/xnli/en/train_en.json" "../../bucket/supervised_data/xnli/en/dev_en.json"
"../../bucket/supervised_data/xnli/en/dep/synthetic_dep_flattened_train_en-en~fr@N~fr@V.json" "../../bucket/supervised_data/xnli/en/dep/synthetic_dep_flattened_dev_en-en~fr@N~fr@V.json"
"../../bucket/supervised_data/ner/en/train.json" "../../bucket/supervised_data/ner/en/dev.json"
"../../bucket/henry_syntax_data/ner/en/train_en_fr_synt.json" "../../bucket/henry_syntax_data/ner/en/dev_en_fr_synt.json"
"../../bucket/supervised_data/pos/en/train-en.json" "../../bucket/supervised_data/pos/en/dev-en.json"
"../../bucket/supervised_data/pos/en/dep/synthetic_dep_flattened_train-en-en~fr@N~fr@V.json" "../../bucket/supervised_data/pos/en/dep/synthetic_dep_flattened_dev-en-en~fr@N~fr@V.json")

suffix="mlm_0_overl"
base_dir="transl_synt_en_500K_$suffix"

pretrain_path="syntax_en_fr_one_to_one"
pretrain_model="../../bucket/model_outputs/en/$pretrain_path/mlm"

outputs=("../../bucket/henry_model_outputs/en/$base_dir/xnli_$suffix/orig" "../../bucket/henry_model_outputs/en/$base_dir/xnli_$suffix/deriv"
"../../bucket/henry_model_outputs/en/$base_dir/ner_$suffix/orig" "../../bucket/henry_model_outputs/en/$base_dir/ner_$suffix/deriv" "../../bucket/henry_model_outputs/en/$base_dir/pos_$suffix/orig" "../../bucket/henry_model_outputs/en/$base_dir/pos_$suffix/deriv")

synthetic_info=("" "--one_to_one_mapping --word_modification replace")

task=("xnli" "ner" "pos")

run_name="${base_dir}"
state=("_orig" "_deriv")

export WANDB_PROJECT="transl_synt_en_500K_mlm_0_overl"
echo $WANDB_PROJECT

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