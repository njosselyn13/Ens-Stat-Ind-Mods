#!/bin/bash
#SBATCH --job-name=train_1_layer_mod
#SBATCH --mem 70G
#####SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C A100|A100-80G

# Conf EBDA
python model_build_1_layer.py --epoch 20 --lr 0.001 --psty 'conf' --pseudo 'domainnet_single_pseudo_labels_using_conf_score_train_data_analysis.csv' --b 0.0 --ls 0.0 --exp_id 'NEW_CONF_SSDA_semi_pseudoweight0.1_numshots1_beta0.0_lr0.001_labelsmooth0.0' --model_train 'MLP' --semisup 'semi' --pseudo_weight 0.1 --gt_shots 1

# Mode EBDA
python model_build_1_layer.py --epoch 20 --lr 0.001 --psty 'mode' --pseudo 'domainnet_single_pseudo_labels_using_mode_train_data_analysis.csv' --b 0.0 --ls 0.0 --exp_id 'NEW_MODE_SSDA_semi_pseudoweight0.1_numshots1_beta0.0_lr0.001_labelsmooth0.0' --model_train 'MLP' --semisup 'semi' --pseudo_weight 0.1 --gt_shots 1

