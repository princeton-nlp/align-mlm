MODEL=~/bucket/henry_model_outputs/en/transl_inv_en_500K_alignedmlm_01_cost_50_alignper/alignedmlm
python learn_mapping_no_svd.py --model1 $MODEL --model2 $MODEL --train_fraction 0.05 --valid_fraction 0.9
MODEL=~/bucket/henry_model_outputs/en/transl_inv_en_500K_alignedmlm_1_cost_50_alignper/alignedmlm
python learn_mapping_no_svd.py --model1 $MODEL --model2 $MODEL --train_fraction 0.05 --valid_fraction 0.9
MODEL=~/bucket/henry_model_outputs/en/transl_inv_en_500K_alignedmlm_10_cost_50_alignper/alignedmlm
python learn_mapping_no_svd.py --model1 $MODEL --model2 $MODEL --train_fraction 0.05 --valid_fraction 0.9