# Ensembles of Statistically Independent Models: An Embarrassingly Simple yet Strong Method for Domain Adaptation

## Introduction 

Codes are provided for reproducibility of results. Included here will be:
- Model codes (DANN, JAN, CDAN, AFN, MCC)
- Ensemble codes
- Analysis python scripts to analyze results
- Samples of saved model files (.pth files)
- Sample file for running individual DA experiments
- Environment setup information
- Details and instructions on how to use this repo and codes 
- Data information


## Environment Setup

In this work, a miniconda environment was setup on a remote compute cluster running Linux. Experiments were run on A100 GPUs using python 3.7.6, CUDA 11.0, torch 1.7.1, and torchvision 0.8.2.

First, download miniconda and move download to compute cluster if applicable: 
- Download miniconda: https://docs.conda.io/en/latest/miniconda.html

Then, run the following commands in order:
- conda create -n my_env1
- source activate "path to environment" (e.x. /home/username/miniconda3/envs/my_env1/)
- conda install python=3.7.6
- pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
- pip install -r requirements_DA.txt


## Individual Domain Adaptation Models ran and how to run

In this work we make use of 5 domain adaptation models: DANN, JAN, CDAN, AFN, and MCC. These 5 model .py files are found in: Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification

BEFORE running any of these models, you must edit each .py file (dann.py, jan.py, cdan.py, afn.py, and mcc.py) to include the correct path to the TEST datasets for DomainNet and Office31.  

Replace: "../../../../../../copy_DomainNet_all_correct/DomainNet/cross_val_folds/"  and "../../../../../../Office31/Original_images/cross_val_folds/" with whatever path to the TEST data you have on your machine for EACH of the 5 model codes listed above. Change this path for both the DomainNet and Office31 test datasets. 

We include a sample of how to run experiments in sample_run.txt. 

In this sample_run.txt file are sample job/experiment submissions for each of the 5 models and both datasets. It requires you to change the path to the data to the necessary path on your machine. They include submissions with all the tuned hyperparameters to demonstrate how to set them. They are just for 1 adaptation task for each dataset (DomainNet: clipart to painting and Office31: amazon to webcam). The dataset directory is shown just for the first cross-validation fold. In reality you would run it 5 times for each model and dataset pair for all 5 cross-val fold data directories.
See at the end of each model .py file how to enable "test" and "analyze" modes for testing an already trained model and generating TSNE plots, respectively. 
Also refer to original code repository cited at bottom for more information on how to run. 

At the bottom of each .py model file is a set of args that include all hyperparameters you can tune. Refer to the supplementary PDF for final hyperparameters determined in our work. And see sample_run.txt for how to set these values when running experiments. 

## Ensemble Model Codes 

Please note, file paths/directories will need to be updated based on your own local setup throughout these codes. 

Please refer to the following link in necessary_files.zip to obtain the ground truth labels for each adaptation task used in each of these codes. Train data labels use: "afn_subset_exps.csv", Test data labels use: "gt_preds_each_da_model_approach1.csv"
https://drive.google.com/drive/folders/1wqk3Ubmk-7_lSQYkbhmaIwNd2vsGWZOQ?usp=sharing

The EBDA ensemble model codes included in this repository can be found in Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification and are as follows:
- model_build_1_layer.py
  - Use this for training single-source DomainNet EBDA experiments
<!--
[//]: # - model_build_1_layer_5DAs.py
[//]: #  - Use this for training single-source DomainNet EBDA of 5 DA models on fold 1 experiments
[//]: # - model_build_1_layer_5folds.py
[//]: #  - Use this for training single-source DomainNet EBDA of 5 fold models on DANN model experiments
-->
- model_build_1_layer_off31.py
  - Use this for training single-source Office-31 EBDA experiments
<!--
[//]: # - model_build_1_layer_multi_src.py
[//]: #  - Use this for training multi-source DomainNet EBDA experiments
[//]: # - model_build_1_layer_multi_src_off31.py
[//]: #  - Use this for training multi-source Office-31 EBDA experiments
[//]: # - model_build_1_layer_SIMPLEX.py
[//]: #  - Use this for training DomainNet EBDA experiments with regularization (simplex) as shown in the supplementary material
-->

## Analysis of results

Included in this repo are also 5 analysis scripts, one specific for each DA model, and anlysis scripts for t-SNE plotting and statistical independence calculations:
- results_analysis_dann.py
- results_analysis_jan.py
- results_analysis_cdan.py
- results_analysis_afn.py
- results_analysis_mcc.py
- tsne_ensemble.py
- tsne_ensemble_5s.py
- tsne_ensemble_our_model.py
- statistical_independence_test.py

Feel free to use and adapt these scripts to analyze results generated using the models. We ran models on a remote compute cluster and analyzed on our local machines, these analysis scripts allow the user to log in to the remote server, copy the necessary files (excel log file and training log txt file) from remote to local, and analyze them and create summaries (csv logs, plots) of training, validation, and testing results.

Near the top of each file, you can enter the paths to the remote folder with your results and where you want to have the results copied locally. 

For example:
- folder_name = 'off31_w2w'
- remote_exp_folder = '/home/user/remote_pth/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/logs/exp1/office31/office31_w2w/' + folder_name + '/'
- local_exp_folder = 'C:\\Users\\yourname\\Desktop\\Results\\exp1\\Office31\\office31_w2w\\' + folder_name + '\\'
- results_table_csv = 'off31_w2w_' + folder_name

Then, set up remote connection (if using a remote compute cluster) by entering your host, username, and password in each analysis script. Using the paramiko python package you will be able to connect directly from the code.

tsne_ensemble.py can be used to generate feature spaces for all 25 individual DA models individually and together and for all 5 DA models for 1 fold of data and for all 5 folds for 1 DA model as seen in the main paper.

tsne_ensemble_5s.py can be used to plot feature spaces of a subset of 5 classes for DomainNet for all 5 DA models for 1 fold of data and for all 5 folds for 1 DA model as seen in the supplementary document.

tsne_ensemble_our_model.py can be used to plot feature spaces of a subset of 5 classes for DomainNet for all 25 models as seen in the supplementary document.

statistical_independence_test.py can be used to calculate the statistical dependency between all pairs of models for all data domains for DomainNet. Please refer to the link here for accessing data_pool_matrix.xlsx files in necessary_files.zip for each adaptation task.
https://drive.google.com/drive/folders/1wqk3Ubmk-7_lSQYkbhmaIwNd2vsGWZOQ?usp=sharing

## Saved models

A few sample saved models can be found in sample_model_files directory. These model .pth checkpoint files are from the epoch with highest validation accuracy and first cross-validation fold for each experiment identified below. We include these few just as an example of what an output model looks like after training (not guaranteed to be models with best performance results, just examples). 

Included 5 model files (.pth checkpoint files):
- DANN for DomainNet adaptation task clipart to painting (file too large for GitHub...)
- AFN for DomainNet adaptation task clipart to painting (file too large for GitHub...)
- MCC for DomainNet adaptation task clipart to painting (file too large for GitHub...)
- CDAN for Office31 adaptation task amazon to webcam
- JAN for Office31 adaptation task amazon to webcam

See link for all 5 model files (too large for GitHub >100MB): https://drive.google.com/drive/folders/1sprX_Dwxl4_zVtX_C6Fbuw-vKNSpDx_M?usp=sharing 

## Data

All data is ~217GB. Provided here are a link to the original DomainNet data release website and accompanying excel files we generate such that you may reproduce the datasets we use for DomainNet (all 5 cross-val folds; and train/val/test sets). If you have difficulty reconstructing the datasets from the provided excel files, please contact us and we can attempt to share the raw data folders (it is fairly large to put all here). 

Original DomainNet data release link (cleaned version): http://ai.bu.edu/M3SDA/ 

For Office-31 data, please see the Google drive link here for all raw data (~1GB) and cite the original work: https://drive.google.com/drive/folders/173BSlSeuaLnhi9HIpMU5UfR7e0zAq4fJ?usp=sharing

This download will contain all the raw data cross-val folds and test set for you already. No need to reconstruct. 


To reconstruct the DomainNet datasets, first download the cleaned version of DomainNet data from the above link. Then look at our folder of excel files named "DomainNet_data". In this folder there are 11 excel files. 

10 of the excel file names follow the structure: DomainNet_full_fold_x_split.xlsx 
x is either 1, 2, 3, 4, 5 for each of the 5 folds.
Split is either train or val.

The 11th file is for the TEST set of DomainNet data: DomainNet_test.xlsx

Reconstruct the following folder structure: 
- full
  - fold_1
    - train
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
    - val
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
  - fold_2
    - train
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
    - val
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
  - fold_3
    - train
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
    - val
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
  - fold_4
    - train
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
    - val
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
  - fold_5
    - train
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch
    - val
      - clipart
      - infograph
      - painting
      - quickdraw
      - real
      - sketch

In each domain folder (clipart, infograph, painting, quickdraw, real, and sketch) there will be 345 folders for each class with each of their respective images in these folders.

In each of the excel files provided, there are 6 sheets corresponding to each of the 6 domains. In each sheet there are 345 columns corresponding to each of the 345 classes. For each class the list of image names are provided. Using the excel file name, sheet names, column names, and image file names you should be able to reconstruct the above folder structures. 

For the DomainNet TEST set, use DomainNet_test.xlsx to reconstruct the following folder structure:

- test
  - clipart
  - infograph
  - painting
  - quickdraw
  - real
  - sketch

Where in each domain folder there will be 345 class folders with respective images in each of these 345 folders. 


This repository is adapted and modified from the below work. Please cite their work in addition to ours if you use this repository.

```latex
@misc{dalib,
  author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
  title = {Transfer-Learning-library},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}
```



## License

See license file

MIT License

Copyright (c) 2022 xxxx xxxxxx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

