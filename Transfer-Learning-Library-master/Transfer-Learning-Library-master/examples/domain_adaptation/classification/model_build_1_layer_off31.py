import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn import random_projection
import shutil
from statistics import mode, mean
from matplotlib import pyplot as plt
import math
import torch
# import torchvision
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim
# from sklearn.utils import shuffle
import argparse
from clid_loss import clid_loss
import time
import sys
from torch.utils.checkpoint import checkpoint
from typing import List
from collections import defaultdict


start_time = time.time()

parser = argparse.ArgumentParser(description='Ensemble Code')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
parser.add_argument('--b', '--beta', default=0.0, type=float)
parser.add_argument('--ls', '--label_smooth', default=0.0, type=float)
parser.add_argument('--epoch', '--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--w', default=False, type=bool) #, action='store_true'
parser.add_argument('--pseudo', default='domainnet_single_pseudo_labels_using_conf_score_train_fold1_data.csv', type=str)
parser.add_argument('--exp_id', default='pseudo1', type=str)
parser.add_argument('--psty', default='conf', type=str)
parser.add_argument('--model_train', default='MLP', type=str)
parser.add_argument('--semisup', default='unsup', type=str) # 'semi' to make true (do semi-sup training)
parser.add_argument('--frac_gt', default=0.25, type=float)
parser.add_argument('--gt_shots', default=1, type=int)
parser.add_argument('--pseudo_weight', default=0.3, type=float)
args = parser.parse_args()

pseudo_train_file = args.pseudo
pseudo_type = args.psty

semi_sup = args.semisup
frac_gt = args.frac_gt
pseudo_weight = args.pseudo_weight
gt_shot = args.gt_shots

parent_dir = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/Ensemble_DA_conf/mean_stdev_mode/Ensemble_DomainNet/ORIGINAL_SPLIT/'


adapt_tasks = ['a2d', 'a2w', 'd2a', 'd2w', 'w2a', 'w2d']


# args variables: adapt task, device, num epochs, learning rate, batch size (train), non_neg, hidden dim size,
    # model_struct, num workers, weight decay,
# use these to also build the csv saving files and model saving file

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_epochs = args.epoch #100
learning_rate = args.lr
beta = args.b
# non_neg = args.w
exp_id = args.exp_id

batch_size = args.batch_size

sftmx = False

feat_extract_exps = True
gaussian_projecting = False

model_train = args.model_train


desired_feat_size = 256
num_models = 5
num_folds = 5
input_dimension = (1000*5)+(desired_feat_size*4*5) #desired_feat_size*num_models*num_folds #10120 # 8625 (logit, softmax inputs), 10120 (features input) (desired feat size*num models (6400)
                        # desired_feat_size*num_models*num_folds
output_dimension = 31 #345
hidden_dimension = 1000

label_smooth = args.ls

if learning_rate == 0.01:
    lr_str = '1e2'
elif learning_rate == 0.001:
    lr_str = '1e3'
else:
    lr_str = str(learning_rate)

# file_save = 'Office31_' + 'Gaussian_' + str(gaussian_projecting) + '_Feat_' + str(feat_extract_exps) + '_SOFTMAX_' + str(sftmx) + '_stacked_linear_1layer_lr' + lr_str + '_epochs' + str(num_epochs)
# file_save = exp_id + '_pseudo_off31_single_src_' + pseudo_type + '_' + str(gaussian_projecting) + '_Feat_' + str(feat_extract_exps) + '_SOFTMAX_' + str(sftmx) + '_stacked_linear_1layer_lr' + lr_str + '_epochs' + str(num_epochs)
file_save = exp_id #+ '_pseudo_off31_single_src_' + pseudo_type + '_lr' + lr_str + '_epochs' + str(num_epochs) + '_beta_' + str(beta)

# folder_save_name = parent_dir + file_save
folder_save_name = 'NEW_EXPS/pseudolabel_exps/off31/single/' + file_save
if not os.path.isdir(folder_save_name):
    os.makedirs(folder_save_name)

averages_file_save = 'AVERAGES_' + file_save
task_file_save = 'TASKS_' + file_save

# if non_neg == True:
#     print('Running with Non-Negative Weights Constraint........')


class SemiSupervisedDatasetLoader(Dataset):
    def __init__(self, data_tensor, label_tensor, label_type_list):
        assert len(data_tensor) == len(label_tensor) == len(label_type_list)
        self.data = data_tensor
        self.labels = label_tensor
        self.label_types = label_type_list  # list of strings like ['gt', 'pseudo', ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        label_type = self.label_types[idx]  # string: 'gt' or 'pseudo'
        return x, y, label_type





# # Guarantee a flip to GT from the pseudolabel
# def SemiSupervisedDataset(
#     ground_truth_labels: List[int],
#     pseudo_labels: List[int],
#     n_shots: int = 1,
#     random_seed: int = 42
# ):
#     ids = [f"sample{i}" for i in range(len(ground_truth_labels))]
#     np.random.seed(random_seed)
#
#     # Step 1: Group indices by class (based on GT labels)
#     class_to_indices = defaultdict(list)
#     for idx, label in enumerate(ground_truth_labels):
#         class_to_indices[label].append(idx)
#
#     # Track GT selections and reasons
#     gt_indices = []
#     gt_reasons = {}
#
#     # Step 2: For each class, pick exactly n_shots GT indices
#     for label, indices in class_to_indices.items():
#         mismatch_indices = [i for i in indices if ground_truth_labels[i] != pseudo_labels[i]]
#
#         chosen = []
#         reasons = {}
#
#         # First pick mismatches (preferred)
#         if mismatch_indices:
#             take = min(n_shots, len(mismatch_indices))
#             picked = np.random.choice(mismatch_indices, size=take, replace=False)
#             chosen.extend(picked)
#             reasons.update({i: "mismatch" for i in picked})
#
#         # If still fewer than n_shots, top up with fallback picks
#         if len(chosen) < n_shots:
#             remaining_needed = n_shots - len(chosen)
#             remaining_pool = list(set(indices) - set(chosen))
#             picked = np.random.choice(remaining_pool, size=remaining_needed, replace=False)
#             chosen.extend(picked)
#             reasons.update({i: "fallback" for i in picked})
#
#         # Store results
#         gt_indices.extend(chosen)
#         gt_reasons.update(reasons)
#
#     # Step 3: Assign GT or pseudo label + reason
#     final_labels = []
#     label_types = []
#     reasons = []
#     for i in range(len(ground_truth_labels)):
#         if i in gt_indices:
#             final_labels.append(ground_truth_labels[i])
#             label_types.append("gt")
#             reasons.append(gt_reasons[i])
#         else:
#             final_labels.append(pseudo_labels[i])
#             label_types.append("pseudo")
#             reasons.append(None)
#
#     # Step 4: Build dataframe
#     df = pd.DataFrame({
#         "id": ids,
#         "label": final_labels,
#         "label_type": label_types,
#         "gt_reason": reasons
#     })
#
#     return df




#######################################################################################################
# THIS ONE
def SemiSupervisedDataset(
    ground_truth_labels: List[int],
    pseudo_labels: List[int],
    n_shots: int = 1,
    random_seed: int = 42
):
    ids = [f"sample{i}" for i in range(len(ground_truth_labels))]
    np.random.seed(random_seed)

    # Step 1: Group indices by class (based on GT labels)
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(ground_truth_labels):
        class_to_indices[label].append(idx)

    # Step 2: For each class, randomly pick n_shots indices to keep as GT
    gt_indices = []
    for label, indices in class_to_indices.items():
        chosen = np.random.choice(indices, size=min(n_shots, len(indices)), replace=False)
        gt_indices.extend(chosen)

    # Step 3: Assign GT or pseudo label
    final_labels = []
    label_types = []
    for i in range(len(ground_truth_labels)):
        if i in gt_indices:
            final_labels.append(ground_truth_labels[i])
            label_types.append("gt")
        else:
            final_labels.append(pseudo_labels[i])
            label_types.append("pseudo")

    # Step 4: Build dataframe
    df = pd.DataFrame({
        "id": ids,
        "label": final_labels,
        "label_type": label_types
    })

    return df
#######################################################################################################



def semi_supervised_loss(logits, labels, label_types, pseudo_weight=0.3):
    """
    logits: [B, num_classes]
    labels: [B] (tensor of class indices)
    label_types: list or tensor of strings/ints indicating 'gt' or 'pseudo'
    pseudo_weight: how much to weight pseudo-label loss (0 to ignore them)
    """
    ce = nn.CrossEntropyLoss(reduction="none")  # compute per-sample loss

    # Convert label_types to tensor mask: 1 for gt, 0 for pseudo
    if isinstance(label_types[0], str):
        gt_mask = torch.tensor([lt == "gt" for lt in label_types], dtype=torch.float32, device=logits.device)
    else:
        gt_mask = (label_types == 1).float()  # if you store as 1 for GT, 0 for pseudo

    pseudo_mask = 1 - gt_mask

    # Per-sample CE loss
    per_sample_loss = ce(logits, labels)

    # Weighted sum: GT full weight, pseudo scaled
    weighted_loss = per_sample_loss * (gt_mask + pseudo_weight * pseudo_mask)

    return weighted_loss.mean()


# Transformer Variable Tokens
class VariableTokenTransformerNet(nn.Module):
    def __init__(self, num_tokens=25, output_dim=31, d_model=256, nhead=8, num_layers=3): # d_model=256, nhead=8
        super(VariableTokenTransformerNet, self).__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.d_model = d_model

        self.input_proj = None  # will be set dynamically
        self.pos_embedding = None  # will be set dynamically

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.transformer = torch.utils.checkpoint.checkpoint_sequential(self.transformer, chunks=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, return_features=False):
        B, total_dim = x.shape
        token_dim = total_dim // self.num_tokens

        # Create projection + positional embedding if not yet created or token count changed
        if (self.input_proj is None) or (self.input_proj.in_features != token_dim):
            self.input_proj = nn.Linear(token_dim, self.d_model).to(x.device)
        if (self.pos_embedding is None) or (self.pos_embedding.shape[1] != self.num_tokens):
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.d_model, device=x.device))

        # Reshape to tokens and project
        x = x.view(B, self.num_tokens, token_dim)  # [B, num_tokens, token_dim]
        x = self.input_proj(x) + self.pos_embedding  # [B, num_tokens, d_model]

        # Transformer: [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        # Pool and classify
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)

        return (pooled, logits) if return_features else logits


# Transformer 25 tokens
class TokenTransformerNet(nn.Module):
    def __init__(self, num_tokens=25, feat_dim=256, output_dim=31, d_model=256, nhead=8, num_layers=3):
        super(TokenTransformerNet, self).__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.output_dim = output_dim

        # Optional projection to d_model if feat_dim != d_model
        if feat_dim != d_model:
            self.input_proj = nn.Linear(feat_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # Learnable positional encoding
        # Z = 6400 // self.num_tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, Z))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) #, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling over token dimension (e.g., mean or [CLS]-style)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x, return_features=False):
        # x: [B, num_tokens * feat_dim]
        B = x.size(0)

        # Z = (self.feat_dim*25)/self.num_tokens

        # x = x.view(B, self.num_tokens, Z)  # [B, 25, feat_dim]

        # token_dim = x.shape[1] // self.num_tokens  # auto-compute size
        # x = x.view(B, self.num_tokens, token_dim)

        x = x.view(B, self.num_tokens, self.feat_dim)  # [B, 25, feat_dim]

        # Apply linear projection + positional encoding: [B, 25, d_model]
        x = self.input_proj(x) + self.pos_embedding  # still [B, 25, d_model]

        # Transpose for Transformer: [B, 25, d_model] → [25, B, d_model]
        x = x.transpose(0, 1)

        # Transformer expects input as [seq_len, batch, d_model]
        x = self.transformer(x)  # [25, B, d_model]

        # Transpose back: [25, B, d_model] → [B, 25, d_model]
        x = x.transpose(0, 1)

        # Pool over token dimension
        pooled = x.mean(dim=1)  # [B, d_model]

        # x = self.input_proj(x) + self.pos_embedding[:, :self.num_tokens, :]  # [B, 25, d_model]
        # x = self.transformer(x)  # [B, 25, d_model]
        #
        # pooled = x.mean(dim=1)  # global average pooling over tokens

        logits = self.classifier(pooled)

        if return_features:
            return pooled, logits
        else:
            return logits

class TransformerEnsembleNet(nn.Module):
    def __init__(self, input_dim=8625, output_dim=31, d_model=512, nhead=8, num_layers=3):
        super(TransformerEnsembleNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Project flat input to sequence of tokens (e.g., reshape to [B, T, d_model])
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding (optional if ordering matters)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))  # or use sin/cos positional encodings

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x, return_features=False):
        # Reshape from [B, input_dim] → [B, 1, d_model]
        x = self.embedding(x).unsqueeze(1) + self.pos_encoder

        # Pass through Transformer
        x = self.transformer_encoder(x)  # [B, 1, d_model]

        features = x.squeeze(1)  # back to [B, d_model]
        logits = self.classifier(features)

        if return_features:
            return features, logits
        else:
            return logits

# VisionTransformer
class ViTFeatureAdapter(nn.Module):
    def __init__(self, vit_model, num_tokens=25, feat_dim=256, output_dim=31, d_model=768):
        super().__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # self.vit_encoder = vit_model.encoder  # Use pretrained encoder blocks from ViT
        self.vit_encoder = vit_model.encoder.layers  # NOT the whole encoder

        # Project your feature dimension into ViT's d_model space
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Learnable positional embedding (like ViT uses)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model))

        # Classifier hed
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x, return_features=False):
        # x: [B, num_tokens * feat_dim]
        B = x.size(0)
        x = x.view(B, self.num_tokens, self.feat_dim)  # [B, num_tokens, feat_dim]

        # Project + add positional encoding
        x = self.input_proj(x) + self.pos_embedding  # [B, num_tokens, d_model]

        # ViT expects shape: [sequence_length, batch_size, d_model]
        x = x.transpose(0, 1)  # [num_tokens, B, d_model]

        # Transformer encoder
        # x = self.vit_encoder(x)  # [num_tokens, B, d_model]
        for blk in self.vit_encoder:
            x = blk(x)

        # Token pooling — mean over tokens
        features = x.mean(dim=0)  # [B, d_model]

        logits = self.classifier(features)  # [B, output_dim]

        if return_features:
            return features, logits
        else:
            return logits


class IdentityExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return x  # No change

# MLP
class ens_net(nn.Module):
    def __init__(self, input_dim=8625, hidden_dim=1000, output_dim=31, model_struct='sequential', beta=0.0):
        super(ens_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_struct = model_struct
        # self.beta = beta

        # sequential way
        # single layer
        # self.net = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=True)) # can add activation functions here when more layers following

        # self.net = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim, bias=True),
        #                     nn.ReLU(),
        #                     nn.Linear(self.hidden_dim, self.output_dim, bias=True))

        # Split net into encoder and classifier
        # if self.beta != 0.0:
        self.feature_extractor = IdentityExtractor(self.input_dim)
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        # )

        self.classifier = nn.Linear(self.input_dim, self.output_dim)

        # # Split net into encoder and classifier
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(self.input_dim, 5000),
        #     nn.ReLU(),
        #     nn.Linear(5000, 3000),
        #     nn.ReLU(),
        #     nn.Linear(3000, 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, 512),
        #     nn.ReLU()
        # )
        #
        # self.classifier = nn.Linear(512, self.output_dim)

        # not sequential way
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer3 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.activation = nn.ReLU()

    def forward(self, X, return_features=False):
        if self.model_struct == 'sequential':
            # sequential
            # return self.net(X)
            features = self.feature_extractor(X)
            logits = self.classifier(features)

            if return_features:
                # print('Returning features and logits')
                return features, logits  # 512-dim feature vector, then logits
            else:
                # print('returning only logits')
                return logits

        else:
            # not sequential
            hidden1 = self.activation( self.layer1(X) )
            hidden2 = self.activation( self.layer2(hidden1) )
            out = self.layer3(hidden2)
            return out



# class ens_net(nn.Module):
#     def __init__(self, input_dim=8625, hidden_dim=1000, output_dim=345, model_struct='sequential'):
#         super(ens_net, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.model_struct = model_struct
#
#         # sequential way
#         # self.net = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=True)) # can add activation functions here when more layers following
#         # self.net = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim, bias=True),
#         #                     nn.ReLU(),
#         #                     nn.Linear(self.hidden_dim, self.output_dim, bias=True))
#
#         # Split net into encoder and classifier
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(self.input_dim, 5000),
#             nn.ReLU(),
#             nn.Linear(5000, 3000),
#             nn.ReLU(),
#             nn.Linear(3000, 1000),
#             nn.ReLU(),
#             nn.Linear(1000, 512),
#             nn.ReLU()
#         )
#
#         self.classifier = nn.Linear(512, self.output_dim)
#
#         # not sequential way
#         self.layer1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
#         self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#         self.layer3 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
#         self.activation = nn.ReLU()
#
#     def forward(self, X, return_features=False):
#         if self.model_struct == 'sequential':
#             # sequential
#             # return self.net(X)
#             features = self.feature_extractor(X)
#             logits = self.classifier(features)
#
#             if return_features:
#                 # print('Returning features and logits')
#                 return features, logits  # 512-dim feature vector, then logits
#             else:
#                 # print('returning only logits')
#                 return logits
#
#         else:
#             # not sequential
#             hidden1 = self.activation( self.layer1(X) )
#             hidden2 = self.activation( self.layer2(hidden1) )
#             out = self.layer3(hidden2)
#             return out


# shuffling, done in dataloader?

# def initialize(p):
#     if isinstance(p, nn.Linear):
#         torch.nn.init.xavier_uniform(p.weight) # can change the initialization type here
#         p.bias.data.fill_(0.01)

all_accs = []
all_f1s = []
all_bal_accs = []

# adapt_tasks_feat_train_npys = []
# adapt_tasks_feat_test_npys = []

task_counter = 0
for adapt_task in adapt_tasks:

    print(task_counter, adapt_task)

    loss_acc_logger_csv = pd.DataFrame()
    task_metrics_csv = pd.DataFrame()


    if feat_extract_exps == True:
        print('Feature Extracting experiments...')

        tasks = ['amazon_dslr', 'amazon_webcam', 'dslr_amazon', 'dslr_webcam', 'webcam_amazon', 'webcam_dslr']

        print(tasks[task_counter])
        models_concat = []
        gauss_dict = {}

        models = ['feature_lists/Office31/DANN/', 'feature_lists/Office31/JAN/', 'feature_lists/Office31/CDAN/', 'feature_lists/Office31/AFN/', 'feature_lists/Office31/MCC/']
        for m in models:
            print(m)
            pth1 = m + tasks[task_counter] + '_' + '1' + '_feature_list.npy'
            pth2 = m + tasks[task_counter] + '_' + '2' + '_feature_list.npy'
            pth3 = m + tasks[task_counter] + '_' + '3' + '_feature_list.npy'
            pth4 = m + tasks[task_counter] + '_' + '4' + '_feature_list.npy'
            pth5 = m + tasks[task_counter] + '_' + '5' + '_feature_list.npy'

            npy_load1 = np.load(pth1)
            npy_load2 = np.load(pth2)
            npy_load3 = np.load(pth3)
            npy_load4 = np.load(pth4)
            npy_load5 = np.load(pth5)
            print('before gaussian transform', npy_load1.shape)

            # if npy_load1.shape[1] != desired_feat_size: # assuming the model feature size is larger than desired feature size
            #     print('Found model to reduce feature dimension size:', m)
            #     gauss_dict[m] = random_projection.GaussianRandomProjection(n_components=desired_feat_size)
            #     # transformer = random_projection.GaussianRandomProjection(n_components=desired_feat_size)
            #     npy_load1 = gauss_dict[m].fit_transform(npy_load1)
            #     npy_load2 = gauss_dict[m].transform(npy_load2)
            #     npy_load3 = gauss_dict[m].transform(npy_load3)
            #     npy_load4 = gauss_dict[m].transform(npy_load4)
            #     npy_load5 = gauss_dict[m].transform(npy_load5)
            #     print('after gaussian transform', npy_load1.shape)

            np_hstack = np.hstack((npy_load1, npy_load2, npy_load3, npy_load4, npy_load5))
            print(np_hstack.shape)
            models_concat.append(np_hstack)
        all_models_concat = np.hstack(
            (models_concat[0], models_concat[1], models_concat[2], models_concat[3], models_concat[4]))
        print(all_models_concat.shape)
        # adapt_tasks_feat_train_npys.append(all_models_concat)
        train_data_feat = all_models_concat
        print(gauss_dict)
        # [*gauss_dict.keys()]

        # Concat Test Data #
        print()
        print('TEST')
        # for t in tasks:
        models_concat = []
        for m in models:
            print(m)
            pth1 = m + 'TEST/' + tasks[task_counter] + '_' + '1' + '_feature_list.npy'
            pth2 = m + 'TEST/' + tasks[task_counter] + '_' + '2' + '_feature_list.npy'
            pth3 = m + 'TEST/' + tasks[task_counter] + '_' + '3' + '_feature_list.npy'
            pth4 = m + 'TEST/' + tasks[task_counter] + '_' + '4' + '_feature_list.npy'
            pth5 = m + 'TEST/' + tasks[task_counter] + '_' + '5' + '_feature_list.npy'
            # pth_dict = {}
            # npy_dict = {}
            # for fld in range(1, num_folds+1):
            #     pth_dict[fld] = m + 'TEST/' + tasks[task_counter] + '_' + str(fld) + '_feature_list.npy'
            #     npy_dict[fld] = np.load(pth_dict[fld])

            npy_load1 = np.load(pth1)
            npy_load2 = np.load(pth2)
            npy_load3 = np.load(pth3)
            npy_load4 = np.load(pth4)
            npy_load5 = np.load(pth5)
            print('before gaussian transform', npy_load1.shape)

            # if npy_load1.shape[1] != desired_feat_size: # assuming the model feature size is larger than desired feature size
            #     print('Found model to reduce feature dimension size:', m)
            #     # transformer = random_projection.GaussianRandomProjection(n_components=desired_feat_size)
            #     npy_load1 = gauss_dict[m].transform(npy_load1)
            #     npy_load2 = gauss_dict[m].transform(npy_load2)
            #     npy_load3 = gauss_dict[m].transform(npy_load3)
            #     npy_load4 = gauss_dict[m].transform(npy_load4)
            #     npy_load5 = gauss_dict[m].transform(npy_load5)
            #     print('after gaussian transform', npy_load1.shape)

            np_hstack = np.hstack((npy_load1, npy_load2, npy_load3, npy_load4, npy_load5))
            print(np_hstack.shape)
            models_concat.append(np_hstack)
        all_models_concat = np.hstack(
            (models_concat[0], models_concat[1], models_concat[2], models_concat[3], models_concat[4]))
        print(all_models_concat.shape)
        # adapt_tasks_feat_test_npys.append(all_models_concat)
        test_data_feat = all_models_concat

    # data_npy_train = np.load(parent_dir + adapt_task + '/' + adapt_task + '.npy') # THIS IS TEST DATA, just for debugging
    if sftmx == True:
        print('Using SOFTMAX train data...')
        data_npy_train = np.load(parent_dir + adapt_task + '/' + adapt_task + '_SOFTMAX_train_set.npy')
    elif feat_extract_exps == True:
        print('Using feature extracted train data...')
        data_npy_train = train_data_feat
    else:
        print('Using LOGIT train data ... ')
        data_npy_train = np.load(parent_dir + adapt_task + '/' + adapt_task + '_train_set.npy')
    print(data_npy_train.shape)
    print()
    # task_csv_train = pd.read_csv(parent_dir + adapt_task + '/' + 'gt_preds_each_da_model_approach1.csv') # THIS IS TEST DATA, just for debugging

    # adapt_tasks = ['a2d', 'a2w', 'd2a', 'd2w', 'w2a', 'w2d']
    if adapt_task == 'd2a' or adapt_task == 'w2a':
        adt1 = 'amazon'
    elif adapt_task == 'a2d' or adapt_task == 'w2d':
        adt1 = 'dslr'
    elif adapt_task == 'a2w' or adapt_task == 'd2w':
        adt1 = 'webcam'

    if adapt_task == 'd2a':
        adt = 'Office31_d2a'
    elif adapt_task == 'w2a':
        adt = 'Office31_w2a'
    elif adapt_task == 'a2d':
        adt = 'Office31_a2d'
    elif adapt_task == 'w2d':
        adt = 'Office31_w2d'
    elif adapt_task == 'a2w':
        adt = 'Office31_a2w'
    elif adapt_task == 'd2w':
        adt = 'Office31_d2w'


    # task_csv_train = pd.read_csv(
    #     '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/off31_gt_target_labels/' + 'off31_train_set_gt_pred_ratings_' + adt + '.csv')  # NEED TO FIND PATH TO CORRECT TRAIN DATA LABELS, for 1 adapt task, the images are the same across each DA model so only need to load from 1 model and just chose afn to use
    # # print(task_csv)

    if semi_sup == 'semi':
        # use semi-supervised (fraction of gt, rest pseudo)
        task_csv_train_gt = pd.read_csv(
           '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/off31_gt_target_labels/' + 'off31_train_set_gt_pred_ratings_' + adt1 + '.csv')  # NEED TO FIND PATH TO CORRECT TRAIN DATA LABELS, for 1 adapt task, the images are the same across each DA model so only need to load from 1 model and just chose afn to use
        gtlist = task_csv_train_gt['Ignore_Ground Truth'].dropna().tolist()

        task_csv_train_pseudo = pd.read_csv(
            '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/off31_eval_models/train_fold1_data/' + pseudo_train_file)
        # pseudolist = task_csv_train_pseudo[adapt_task].dropna().tolist()
        pseudolist = task_csv_train_pseudo[adt].dropna().tolist()

        df_semi = SemiSupervisedDataset(ground_truth_labels=gtlist, pseudo_labels=pseudolist, n_shots=gt_shot)

        image_gt_labels = df_semi['label'].dropna().tolist()

        label_types = df_semi['label_type'].dropna().tolist()

        X_tensor = torch.Tensor(data_npy_train)
        Y_tensor = torch.Tensor(image_gt_labels)

        train_dset = SemiSupervisedDatasetLoader(X_tensor, Y_tensor, label_types)
    else:
        # use pseudo

        task_csv_train = pd.read_csv('/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/off31_eval_models/train_fold1_data/' + pseudo_train_file)

    # image_names = task_csv_train['Image Name'].dropna().tolist()
    # # print(image_names)
    # print('Number images:', len(image_names))
    # image_gt_labels = task_csv_train['Ignore_Ground Truth'].dropna().tolist()
        image_gt_labels = task_csv_train[adt].dropna().tolist()
    # print(image_gt_labels)
    # print(len(image_gt_labels))
        print()


        X_tensor = torch.Tensor(data_npy_train)
        Y_tensor = torch.Tensor(image_gt_labels)

        train_dset = TensorDataset(X_tensor, Y_tensor) # are rows read as samples? Is my Y_temsor the right axis, each row? -- Yes yes

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=2) # define other params in here like batch size... # drop_last=True if batch size not divisible
                        # num_workers, need to request at least however many i put here in .sh file when submitting job (--ntasks)

    # MLP
    if model_train == 'MLP':
        model = ens_net(input_dim=input_dimension, hidden_dim=hidden_dimension, output_dim=output_dimension,
                        model_struct='sequential').to(
            device)

    # Transformer 1 token
    elif model_train == 'Transformer_1token':
        model = TransformerEnsembleNet(input_dim=input_dimension, output_dim=output_dimension).to(device)

    # Transformer 25 tokens
    elif model_train == 'Transformer_25tokens':
        model = TokenTransformerNet(
            num_tokens=25,
            feat_dim=desired_feat_size,  # should be 256
            output_dim=output_dimension  #
        ).to(device)

    # Transformer 100 tokens
    elif model_train == 'Transformer_100tokens':
        model = VariableTokenTransformerNet(
            num_tokens=100, d_model=64, nhead=4,
            # feat_dim=desired_feat_size,  # should be 256
            output_dim=output_dimension  #
        ).to(device)

    # ViT Model 25 tokens
    # elif model_train == 'ViT_25tokens':
    #     # Load pretrained ViT base model
    #     vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    #     # Wrap it
    #     model = ViTFeatureAdapter(vit_model=vit, num_tokens=25, feat_dim=desired_feat_size, output_dim=output_dimension)
    #     model = model.to(device)

    # model = ens_net(input_dim=input_dimension, hidden_dim=hidden_dimension, output_dim=output_dimension, model_struct='sequential').to(device) # input_dim=8625, hidden_dim=1000, output_dim=345, model_struct='sequential'

    # torch.manual_seed(32)
    # model.apply(initialize)

    # print('Initialized Weights:')
    # for name, pp in model.parameters():
    #     print(name, pp.data)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    clid_losses = []
    ce_losses = []
    accs = []
    for epoch in range(0, num_epochs):
        clid_loss_epoch = []
        ce_loss_epoch = []
        losses_epoch = []
        accs_epoch = []

        if semi_sup == 'semi':
            for (x, y, l) in train_loader:
                x, y = x.to(device), y.to(device)  # , l.to(device)

                optimizer.zero_grad()

                features, logit = model(x, return_features=True)
                loss = semi_supervised_loss(logits=logit, labels=y.long(), label_types=l, pseudo_weight=pseudo_weight)
                losses_epoch.append(loss.item())

                # NOT USED for semi-supervised
                ce_loss_epoch.append(0)
                clid_loss_epoch.append(0)

                loss.backward()  # calculates the gradients

                optimizer.step()  # applies the updates to the weights based on the gradients

                acc1 = accuracy_score(y.cpu(), torch.argmax(logit.cpu(), dim=1))
                accs_epoch.append(acc1)

        else:

            for (x,y) in train_loader:
                x,y = x.to(device), y.to(device)

                optimizer.zero_grad()

                features, logit = model(x, return_features=True)

                # beta = 0
                if label_smooth != 0.0:
                    ce_loss = nn.functional.cross_entropy(logit, y.long(),
                                                          label_smoothing=label_smooth)  # target y needed to be long tensor--different for diff loss functions
                else:
                    ce_loss = nn.functional.cross_entropy(logit, y.long())

                # beta = 0
                # ce_loss = nn.functional.cross_entropy(logit,
                #                                       y.long())  # target y needed to be long tensor--different for diff loss functions

                clid_l = clid_loss(features, logit)

                loss = ce_loss + beta * clid_l

                losses_epoch.append(loss.item())
                ce_loss_epoch.append(ce_loss.item())
                clid_loss_epoch.append(clid_l.item())

                # loss = # cross entropy loss function we can define outside loop take in logit and y
                loss.backward() # calculates the gradients

                optimizer.step() # applies the updates to the weights based on the gradients

                acc1 = accuracy_score(y.cpu(), torch.argmax(logit.cpu(), dim=1))
                accs_epoch.append(acc1)

                # if non_neg == True:
                #     for p in model.parameters():
                #         p.data.clamp_(0.0, 1e10)
                #print('Min model weights:', min(model.parameters()))
                        #print('Params min:', torch.min(p.data))

        loss_epoch_avg = mean(losses_epoch)
        losses.append(loss_epoch_avg)

        ce_loss_epoch_avg = mean(ce_loss_epoch)
        ce_losses.append(ce_loss_epoch_avg)

        clid_loss_epoch_avg = mean(clid_loss_epoch)
        clid_losses.append(clid_loss_epoch_avg)

        acc_epoch_avg = mean(accs_epoch)
        accs.append(acc_epoch_avg)

    # print(losses)
    # print(len(losses))
    loss_acc_logger_csv['loss'] = pd.Series(losses)
    loss_acc_logger_csv['ce_loss'] = pd.Series(ce_losses)
    loss_acc_logger_csv['clid_loss'] = pd.Series(clid_losses)
    loss_acc_logger_csv['acc'] = pd.Series(accs)


    ###################################
    ###################################
    ###################################


    ### TESTING ###

    print()
    print('------------------------')
    print()
    print('TESTING......')

    if sftmx == True:
        print('Using SOFTMAX test data...')
        data_npy_test = np.load(parent_dir + adapt_task + '/' + adapt_task + '_SOFTMAX_test_set.npy')  # THIS IS TEST DATA
    elif feat_extract_exps == True:
        print('Using feature extracted test data...')
        data_npy_test = test_data_feat
    else:
        print('Using LOGIT test data ... ')
        data_npy_test = np.load(parent_dir + adapt_task + '/' + adapt_task + '.npy') # THIS IS TEST DATA
    print(data_npy_test.shape)
    print()
    if adapt_task == 'd2a' or adapt_task == 'w2a':
        adt = 'amazon'
    elif adapt_task == 'a2d' or adapt_task == 'w2d':
        adt = 'dslr'
    elif adapt_task == 'a2w' or adapt_task == 'd2w':
        adt = 'webcam'
    task_csv_test = pd.read_csv('/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/off31_gt_target_labels/' + 'off31_test_set_gt_pred_ratings_' + adt + '.csv') # THIS IS TEST DATA
    # print(task_csv)

    image_names_test = task_csv_test['Image Name'].dropna().tolist()
    # print(image_names)
    print('Number images:', len(image_names_test))
    image_gt_labels_test = task_csv_test['Ignore_Ground Truth'].dropna().tolist()
    # print(image_gt_labels)
    # print(len(image_gt_labels))
    print()

    X_test = torch.Tensor(data_npy_test)
    Y_test = torch.Tensor(image_gt_labels_test)

    test_dset = TensorDataset(X_test, Y_test)

    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=2) # define other params in here like batch size...

    with torch.no_grad():
        model.eval().cpu()
        # for (x_test,y_test) in test_loader:
        y_pred = model(X_test)
        # print(y_pred)
        # print(y_pred[0:2])
        # print(y_pred[0].size())
        print(y_pred.size())

        acc = accuracy_score(Y_test, torch.argmax(y_pred, dim=1))
        f1_weighted = f1_score(Y_test, torch.argmax(y_pred, dim=1), average='weighted')
        bal_acc = balanced_accuracy_score(Y_test, torch.argmax(y_pred, dim=1))


        task_metrics_csv['Accuracy'] = pd.Series(acc)
        task_metrics_csv['F1_Weighted'] = pd.Series(f1_weighted)
        task_metrics_csv['Balanced_Accuracy'] = pd.Series(bal_acc)

        print(adapt_task)
        print('Accuracy:', acc)
        print('F1-Weighted:', f1_weighted)
        print('Balanced Accuracy:', bal_acc)
        print()

        all_accs.append(acc)
        all_f1s.append(f1_weighted)
        all_bal_accs.append(bal_acc)


    task_metrics_csv.to_csv(folder_save_name + '/' + adapt_task + '_' + task_file_save + '.csv')

    loss_acc_logger_csv.to_csv(folder_save_name + '/' + adapt_task + '_logger_' + file_save + '.csv')

    task_counter = task_counter + 1

mean_acc = mean(all_accs)
mean_f1 = mean(all_f1s)
mean_bal_acc = mean(all_bal_accs)

summary_csv = pd.DataFrame()
summary_csv['Average Accuracy'] = pd.Series(mean_acc)
summary_csv['Average F1-Weighted'] = pd.Series(mean_f1)
summary_csv['Average Balanced Accuracy'] = pd.Series(mean_bal_acc)

summary_csv.to_csv(folder_save_name + '/' + averages_file_save + '.csv')

print()
print('AVERAGES:')
print('Average Accuracy', mean_acc)
print('Average F1-Weighted', mean_f1)
print('Average Balanced Accuracy', mean_bal_acc)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
