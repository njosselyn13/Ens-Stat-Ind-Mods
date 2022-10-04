# An Empirical Study of Domain Adaptation: Are We Really Learning Transferable Representations?

Nicholas Josselyn

Link to paper: pending ... 

## Introduction

This code was used to produce results for the above paper title. Codes are provided for reproducibility of results. Included here will be:
- Model codes (DANN, JAN, CDAN, AFN, MCC)
- Analysis python scripts to analyze results
- Samples of saved model files (.pth files)
- Sample file for running experiments
- Environment setup information
- Details and instructions on how to use this repo and codes 
- Supplementary document (PDF) with additional results, details, and hyperparameter information
- Data information
- Citation and license information

## Environment Setup

In this work, a miniconda environment was setup on a remote compute cluster running Linux. Experiments were run on A100 GPUs using python 3.7.6, CUDA 11.0, torch 1.7.1, and torchvision 0.8.2.

First, download miniconda and move download to compute cluster if applicable: 
- Download miniconda: https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html 

Then, run the following commands in order:
- conda create -n my_env1
- source activate "path to environment" (e.x. /home/username/miniconda3/envs/my_env1/)
- conda install python=3.7.6
- pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
- pip install -r requirements_DA.txt


## Models ran and how to run

In this work we make use of 5 domain adaptation models: DANN, JAN, CDAN, AFN, and MCC. These 5 model .py files are found in: Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification

BEFORE running any of these models, you must edit each .py file (dann.py, jan.py, cdan.py, afn.py, and mcc.py) to include the correct path to the TEST datasets for DomainNet and Office31.  

Replace: "../../../../../../copy_DomainNet_all_correct/DomainNet/cross_val_folds/"  and "../../../../../../Office31/Original_images/cross_val_folds/" with whatever path to the TEST data you have on your machine for EACH of the 5 model codes listed above. Change this path for both the DomainNet and Office31 test datasets. 

We include a sample of how to run experiments in sample_run.txt. 

In this sample_run.txt file are sample job/experiment submissions for each of the 5 models and both datasets. It requires you to change the path to the data to the necessary path on your machine. They include submissions with all the tuned hyperparameters to demonstrate how to set them. They are just for 1 adaptation task for each dataset (DomainNet: clipart to painting and Office31: amazon to webcam) and only the "full" dataset for DomainNet. The dataset directory is shown just for the first cross-validation fold. In reality you would run it 5 times for each model and dataset pair for all 5 cross-val fold data directories.
See at the end of each model .py file how to enable "test" and "analyze" modes for testing an already trained model and generating TSNE plots, respectively. 
Also refer to original code repository cited at bottom for more information on how to run. 

At the bottom of each .py model file is a set of args that include all hyperparameters you can tune. Refer to the included supplementary PDF here for final hyperparameters determined in our work. And see sample_run.txt for how to set these values when running experiments. 


## Analysis of results

Included in this repo are also 5 analysis scripts, one specific for each model:
- results_analysis_dann.py
- results_analysis_jan.py
- results_analysis_cdan.py
- results_analysis_afn.py
- results_analysis_mcc.py

Feel free to use and adapt these scripts to analyze results generated using the models. We ran models on a remote compute cluster and analyzed on our local machines, these analysis scripts allow the user to log in to the remote server, copy the necessary files (excel log file and training log txt file) from remote to local, and analyze them and create summaries (csv logs, plots) of training, validation, and testing results.

Near the top of each file, you can enter the paths to the remote folder with your results and where you want to have the results copied locally. 

For example:
- folder_name = 'off31_w2w'
- remote_exp_folder = '/home/user/remote_pth/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/logs/exp1/office31/office31_w2w/' + folder_name + '/'
- local_exp_folder = 'C:\\Users\\yourname\\Desktop\\Results\\exp1\\Office31\\office31_w2w\\' + folder_name + '\\'
- results_table_csv = 'off31_w2w_' + folder_name

Then, set up remote connection (if using a remote compute cluster) by entering your host, username, and password in each analysis script. Using the paramiko pyton package you will be able to connect directly from the code.


## Saved models

A few sample saved models can be found in sample_model_files directory. These model .pth checkpoint files are from the epoch with highest validation accuracy and first cross-validation fold for each experiment identified below. We include these few just as an example of what an output model looks like after training (not guaranteed to be models with best performance results, just examples). 

Included 5 model files (.pth checkpoint files):
- DANN for DomainNet adaptation task clipart to painting
- AFN for DomainNet adaptation task clipart to painting
- MCC for DomainNet adaptation task clipart to painting
- CDAN for Office31 adaptation task amazon to webcam
- JAN for Office31 adaptation task amazon to webcam

## Data

Access to data splits for reproducibility is pending.

## Citation

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

Our paper citation is pending ... 


## License

See license file

