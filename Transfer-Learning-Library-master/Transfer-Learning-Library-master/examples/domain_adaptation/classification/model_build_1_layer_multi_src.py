import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn import random_projection
import shutil
from statistics import mode, mean, stdev, multimode, median, median_high, median_low
from matplotlib import pyplot as plt
import math
import torch
# import torchvision
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
# from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description='Ensemble Code')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
parser.add_argument('--w', default=False, type=bool) #, action='store_true'
parser.add_argument('--pseudo', default='domainnet_multi_pseudo_labels_using_conf_score_train_data_analysis.csv', type=str)
parser.add_argument('--psty', default='conf', type=str)
args = parser.parse_args()

pseudo_train_file = args.pseudo
pseudo_type = args.psty


parent_dir = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/Ensemble_DA_conf/mean_stdev_mode/Ensemble_DomainNet/ORIGINAL_SPLIT/'


# adapt_tasks = ['DomainNet_c2i'] #, 'DomainNet_c2p']

# adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p', 'DomainNet_c2q', 'DomainNet_c2r', 'DomainNet_c2s',
#               'DomainNet_i2c', 'DomainNet_i2p', 'DomainNet_i2q', 'DomainNet_i2r', 'DomainNet_i2s',
#               'DomainNet_p2c', 'DomainNet_p2i','DomainNet_p2q', 'DomainNet_p2r', 'DomainNet_p2s',
#               'DomainNet_q2c', 'DomainNet_q2i','DomainNet_q2p','DomainNet_q2r', 'DomainNet_q2s',
#               'DomainNet_r2c', 'DomainNet_r2i','DomainNet_r2p', 'DomainNet_r2q', 'DomainNet_r2s',
#               'DomainNet_s2c', 'DomainNet_s2i','DomainNet_s2p', 'DomainNet_s2q', 'DomainNet_s2r']

# adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p', 'DomainNet_c2q', 'DomainNet_c2r', 'DomainNet_c2s', 'DomainNet_i2c']
# adapt_tasks = ['DomainNet_i2p', 'DomainNet_i2q', 'DomainNet_i2r', 'DomainNet_i2s', 'DomainNet_p2c', 'DomainNet_p2i']
# adapt_tasks = ['DomainNet_p2q', 'DomainNet_p2r', 'DomainNet_p2s', 'DomainNet_q2c', 'DomainNet_q2i','DomainNet_q2p']
# adapt_tasks = ['DomainNet_q2r', 'DomainNet_q2s', 'DomainNet_r2c', 'DomainNet_r2i','DomainNet_r2p', 'DomainNet_r2q']
# adapt_tasks = ['DomainNet_r2s', 'DomainNet_s2c', 'DomainNet_s2i', 'DomainNet_s2p']
# adapt_tasks = ['DomainNet_s2q', 'DomainNet_s2r']


# adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p', 'DomainNet_c2q', 'DomainNet_c2r', 'DomainNet_c2s',
#               'DomainNet_i2c', 'DomainNet_i2p', 'DomainNet_i2q', 'DomainNet_i2r', 'DomainNet_i2s',
#               'DomainNet_p2c', 'DomainNet_p2i','DomainNet_p2q']

# adapt_tasks = ['DomainNet_p2r', 'DomainNet_p2s', 'DomainNet_q2c', 'DomainNet_q2i','DomainNet_q2p','DomainNet_q2r', 'DomainNet_q2s',
#               'DomainNet_r2c', 'DomainNet_r2i','DomainNet_r2p', 'DomainNet_r2q', 'DomainNet_r2s',
#               'DomainNet_s2c', 'DomainNet_s2i','DomainNet_s2p']
#
# adapt_tasks = ['DomainNet_s2q', 'DomainNet_s2r']

# args variables: adapt task, device, num epochs, learning rate, batch size (train), non_neg, hidden dim size,
    # model_struct, num workers, weight decay,
# use these to also build the csv saving files and model saving file

# adapt_tasks = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

adapt_tasks = ['c', 'i', 'p', 'q', 'r', 's'] # the 6 possible target domains in this case for multi-source code
# adapt_tasks = ['c'] #, 'i', 'p', 'q', 'r', 's']

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_epochs = 100
learning_rate = args.lr
non_neg = args.w

sftmx = False

feat_extract_exps = True
gaussian_projecting = True


desired_feat_size = 256
num_models = 5
num_folds = 5
# input_dimension = desired_feat_size*num_models*num_folds #10120 # 8625 (logit, softmax inputs), 10120 (features input) (desired feat size*num models (6400)
#                         # desired_feat_size*num_models*num_folds
input_dimension = 5*num_folds*num_models*desired_feat_size # the 5 is the number of domains-1 (5 srcs for 1 target in multi scenario) 
output_dimension = 345
hidden_dimension = 1000

if learning_rate == 0.01:
    lr_str = '1e2'
elif learning_rate == 0.001:
    lr_str = '1e3'
else:
    lr_str = str(learning_rate)

# file_save = 'MULTI_SOURCE_small_dnet_' + 'Gaussian_' + str(gaussian_projecting) + '_Feat_' + str(feat_extract_exps) + '_SOFTMAX_' + str(sftmx) + '_stacked_linear_1layer_lr' + lr_str + '_epochs' + str(num_epochs) + '_weights_' + str(non_neg)
file_save = 'pseudo_dnet_multi_src' + pseudo_type + '_' + 'Gaussian_' + str(gaussian_projecting) + '_Feat_' + str(feat_extract_exps) + '_SOFTMAX_' + str(sftmx) + '_stacked_linear_1layer_lr' + lr_str + '_epochs' + str(num_epochs) + '_weights_' + str(non_neg)

# folder_save_name = parent_dir + file_save
folder_save_name = 'pseudolabel_exps/dnet/multi/' + file_save
if not os.path.isdir(folder_save_name):
    os.makedirs(folder_save_name)

averages_file_save = 'AVERAGES_' + file_save
task_file_save = 'TASKS_' + file_save

if non_neg == True:
    print('Running with Non-Negative Weights Constraint........')

# Train Labels


# parent_train_lbls = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/train_data_analysis/Ensemble_DomainNet/ORIGINAL_SPLIT/'
# task_csv_train_dict = {'c': parent_train_lbls + 'DomainNet_i2c' + '/afn_subset_exps/' + 'afn_subset_exps.csv',
#                       'i': parent_train_lbls + 'DomainNet_c2i' + '/afn_subset_exps/' + 'afn_subset_exps.csv',
#                       'p': parent_train_lbls + 'DomainNet_i2p' + '/afn_subset_exps/' + 'afn_subset_exps.csv',
#                       'q': parent_train_lbls + 'DomainNet_i2q' + '/afn_subset_exps/' + 'afn_subset_exps.csv',
#                       'r': parent_train_lbls + 'DomainNet_i2r' + '/afn_subset_exps/' + 'afn_subset_exps.csv',
#                       's': parent_train_lbls + 'DomainNet_i2s' + '/afn_subset_exps/' + 'afn_subset_exps.csv'}

# parent_train_lbls = '/home/njjosselyn/ARL/domain_adaptation/DomainNet_all_correct/DomainNet_small/'
# task_csv_train_dict = {'c': parent_train_lbls + 'c_DomainNet_small_labels.csv', 'i': parent_train_lbls + 'i_DomainNet_small_labels.csv', 'p': parent_train_lbls + 'p_DomainNet_small_labels.csv', 'q': parent_train_lbls + 'q_DomainNet_small_labels.csv', 'r': parent_train_lbls + 'r_DomainNet_small_labels.csv', 's': parent_train_lbls + 's_DomainNet_small_labels.csv'}


# Test Labels

# task_csv_test = pd.read_csv(parent_dir + adapt_task + '/' + 'gt_preds_each_da_model_approach1.csv') # THIS IS TEST DATA

paren_test_lbls = parent_dir
task_csv_test_dict = {'c': paren_test_lbls + 'DomainNet_i2c' + '/' + 'gt_preds_each_da_model_approach1.csv',
                       'i': paren_test_lbls + 'DomainNet_c2i' + '/' + 'gt_preds_each_da_model_approach1.csv',
                       'p': paren_test_lbls + 'DomainNet_i2p' + '/' + 'gt_preds_each_da_model_approach1.csv',
                       'q': paren_test_lbls + 'DomainNet_i2q' + '/' + 'gt_preds_each_da_model_approach1.csv',
                       'r': paren_test_lbls + 'DomainNet_i2r' + '/' + 'gt_preds_each_da_model_approach1.csv',
                       's': paren_test_lbls + 'DomainNet_i2s' + '/' + 'gt_preds_each_da_model_approach1.csv'}


class ens_net(nn.Module):
    def __init__(self, input_dim=8625, hidden_dim=1000, output_dim=345, model_struct='sequential'):
        super(ens_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_struct = model_struct

        # sequential way
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=True)) # can add activation functions here when more layers following
        # self.net = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim, bias=True),
        #                     nn.ReLU(),
        #                     nn.Linear(self.hidden_dim, self.output_dim, bias=True))

        # not sequential way
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer3 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.activation = nn.ReLU()

    def forward(self, X):
        if self.model_struct == 'sequential':
            # sequential
            return self.net(X)

        else:
            # not sequential
            hidden1 = self.activation( self.layer1(X) )
            hidden2 = self.activation( self.layer2(hidden1) )
            out = self.layer3(hidden2)
            return out

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

        # tasks = ['clipart_infograph', 'clipart_painting', 'clipart_quickdraw', 'clipart_real', 'clipart_sketch',
        #          'infograph_clipart', 'infograph_painting', 'infograph_quickdraw', 'infograph_real', 'infograph_sketch',
        #          'painting_clipart', 'painting_infograph', 'painting_quickdraw', 'painting_real', 'painting_sketch',
        #          'quickdraw_clipart', 'quickdraw_infograph', 'quickdraw_painting', 'quickdraw_real', 'quickdraw_sketch',
        #          'real_clipart', 'real_infograph', 'real_painting', 'real_quickdraw', 'real_sketch',
        #          'sketch_clipart', 'sketch_infograph', 'sketch_painting', 'sketch_quickdraw', 'sketch_real']

        tasks = [ ['infograph_clipart', 'painting_clipart', 'quickdraw_clipart', 'real_clipart', 'sketch_clipart'],
                ['clipart_infograph', 'painting_infograph', 'quickdraw_infograph', 'real_infograph', 'sketch_infograph'],
                ['clipart_painting', 'infograph_painting', 'quickdraw_painting', 'real_painting', 'sketch_painting'],
                ['clipart_quickdraw', 'infograph_quickdraw', 'painting_quickdraw', 'real_quickdraw', 'sketch_quickdraw'],
                ['clipart_real', 'infograph_real', 'painting_real', 'quickdraw_real', 'sketch_real'],
                ['clipart_sketch', 'infograph_sketch', 'painting_sketch', 'quickdraw_sketch', 'real_sketch'] ]


        print(tasks[task_counter])
        models_concat = []
        gauss_dict = {}

        models = ['feature_lists/dataset/DANN/', 'feature_lists/dataset/JAN/', 'feature_lists/dataset/CDAN/', 'feature_lists/dataset/AFN/', 'feature_lists/dataset/MCC/']

        for m in models:
            print(m)
            tasks_concat = []
            for ts in tasks[task_counter]:
                print('Task:', ts)
                pth1 = m + ts + '_' + '1' + '_feature_list.npy'
                pth2 = m + ts + '_' + '2' + '_feature_list.npy'
                pth3 = m + ts + '_' + '3' + '_feature_list.npy'
                pth4 = m + ts + '_' + '4' + '_feature_list.npy'
                pth5 = m + ts + '_' + '5' + '_feature_list.npy'

                npy_load1 = np.load(pth1)
                npy_load2 = np.load(pth2)
                npy_load3 = np.load(pth3)
                npy_load4 = np.load(pth4)
                npy_load5 = np.load(pth5)
                print('before gaussian transform', npy_load1.shape)

                if npy_load1.shape[1] != desired_feat_size: # assuming the model feature size is larger than desired feature size
                    print('Found model to reduce feature dimension size:', m)
                    gauss_dict[m] = random_projection.GaussianRandomProjection(n_components=desired_feat_size)
                    # transformer = random_projection.GaussianRandomProjection(n_components=desired_feat_size)
                    npy_load1 = gauss_dict[m].fit_transform(npy_load1)
                    npy_load2 = gauss_dict[m].transform(npy_load2)
                    npy_load3 = gauss_dict[m].transform(npy_load3)
                    npy_load4 = gauss_dict[m].transform(npy_load4)
                    npy_load5 = gauss_dict[m].transform(npy_load5)
                    print('after gaussian transform', npy_load1.shape)

                np_hstack = np.hstack((npy_load1, npy_load2, npy_load3, npy_load4, npy_load5)) # stack 5 folds, 1 target, 1 model
                print('np_stack 5 folds shape:', np_hstack.shape) # 1280
                tasks_concat.append(np_hstack) # len = 5 after loop

            print('length tasks_concat:', len(tasks_concat)) # len = 5, each element 256*5=1280
            print('shape tasks_concat element:', tasks_concat[0].shape)
            # models_concat.append(tasks_concat) # len = 25
            tasks_stack = np.concatenate(tasks_concat, axis=1)
            print('tasks_stack shape:', tasks_stack.shape)
            models_concat.append(tasks_stack)

        all_models_concat = np.concatenate(models_concat, axis=1)
        # all_models_concat = np.hstack(
        #     (models_concat[0], models_concat[1], models_concat[2], models_concat[3], models_concat[4]))
        print('all_models_concat shape:', all_models_concat.shape)

        # task_counter = task_counter + 1

        # adapt_tasks_feat_train_npys.append(all_models_concat)
        train_data_feat = all_models_concat
        print(gauss_dict)
        # [*gauss_dict.keys()]

        # Concat Test Data #
        print()
        print()
        print('TEST')
        # for t in tasks:
        models_concat = []
        for m in models:
            print(m)
            tasks_concat = []
            for ts in tasks[task_counter]:
                print('Task:', ts)
                pth1 = m + 'TEST/' + ts + '_' + '1' + '_feature_list.npy'
                pth2 = m + 'TEST/' + ts + '_' + '2' + '_feature_list.npy'
                pth3 = m + 'TEST/' + ts + '_' + '3' + '_feature_list.npy'
                pth4 = m + 'TEST/' + ts + '_' + '4' + '_feature_list.npy'
                pth5 = m + 'TEST/' + ts + '_' + '5' + '_feature_list.npy'
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

                if npy_load1.shape[1] != desired_feat_size: # assuming the model feature size is larger than desired feature size
                    print('Found model to reduce feature dimension size:', m)
                    # transformer = random_projection.GaussianRandomProjection(n_components=desired_feat_size)
                    npy_load1 = gauss_dict[m].transform(npy_load1)
                    npy_load2 = gauss_dict[m].transform(npy_load2)
                    npy_load3 = gauss_dict[m].transform(npy_load3)
                    npy_load4 = gauss_dict[m].transform(npy_load4)
                    npy_load5 = gauss_dict[m].transform(npy_load5)
                    print('after gaussian transform', npy_load1.shape)

                np_hstack = np.hstack((npy_load1, npy_load2, npy_load3, npy_load4, npy_load5))
                print('np_stack 5 folds shape:', np_hstack.shape)  # 1280
                tasks_concat.append(np_hstack)  # len = 5 after loop

            print('length tasks_concat:', len(tasks_concat))  # len = 5, each element 256*5=1280
            print('shape tasks_concat element:', tasks_concat[0].shape)
            # models_concat.append(tasks_concat) # len = 25
            tasks_stack = np.concatenate(tasks_concat, axis=1)
            print('tasks_stack shape:', tasks_stack.shape)
            models_concat.append(tasks_stack)

        all_models_concat = np.concatenate(models_concat, axis=1)
        print('all_models_concat shape:', all_models_concat.shape)

        # all_models_concat = np.hstack(
        #     (models_concat[0], models_concat[1], models_concat[2], models_concat[3], models_concat[4]))
        # print(all_models_concat.shape)
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



    # task_csv_train = pd.read_csv(task_csv_train_dict[adapt_task])
    task_csv_train = pd.read_csv('/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/train_data_analysis/Ensemble_DomainNet/ORIGINAL_SPLIT/' + pseudo_train_file)
    # task_csv_train = pd.read_csv(
    #     '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/train_data_analysis/Ensemble_DomainNet/ORIGINAL_SPLIT/' + adapt_task + '/afn_subset_exps/' + 'afn_subset_exps.csv')  # NEED TO FIND PATH TO CORRECT TRAIN DATA LABELS, for 1 adapt task, the images are the same across each DA model so only need to load from 1 model and just chose afn to use
    # print(task_csv)

    # image_names = task_csv_train['Image_Name'].dropna().tolist()
    # print(image_names)
    # print('Number images:', len(image_names))
    # image_gt_labels = task_csv_train['Ground_Truth'].dropna().tolist()
    image_gt_labels = task_csv_train[adapt_task].dropna().tolist()
    # print(image_gt_labels)
    # print(len(image_gt_labels))
    print()


    X_tensor = torch.Tensor(data_npy_train)
    Y_tensor = torch.Tensor(image_gt_labels)

    train_dset = TensorDataset(X_tensor, Y_tensor) # are rows read as samples? Is my Y_temsor the right axis, each row? -- Yes yes

    train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=2) # define other params in here like batch size... # drop_last=True if batch size not divisible
                        # num_workers, need to request at least however many i put here in .sh file when submitting job (--ntasks)
    model = ens_net(input_dim=input_dimension, hidden_dim=hidden_dimension, output_dim=output_dimension, model_struct='sequential').to(device) # input_dim=8625, hidden_dim=1000, output_dim=345, model_struct='sequential'
    # torch.manual_seed(32)
    # model.apply(initialize)

    # print('Initialized Weights:')
    # for name, pp in model.parameters():
    #     print(name, pp.data)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    losses = []
    accs = []
    for epoch in range(0, num_epochs):
        losses_epoch = []
        accs_epoch = []
        for (x,y) in train_loader:
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logit = model(x)

            loss = nn.functional.cross_entropy(logit, y.long()) # target y needed to be long tensor--different for diff loss functions
            losses_epoch.append(loss.item())

            # loss = # cross entropy loss function we can define outside loop take in logit and y
            loss.backward() # calculates the gradients

            optimizer.step() # applies the updates to the weights based on the gradients

            acc1 = accuracy_score(y.cpu(), torch.argmax(logit.cpu(), dim=1))
            accs_epoch.append(acc1)

            if non_neg == True:
                for p in model.parameters():
                    p.data.clamp_(0.0, 1e10)
            #print('Min model weights:', min(model.parameters()))
                    #print('Params min:', torch.min(p.data))



        loss_epoch_avg = mean(losses_epoch)
        losses.append(loss_epoch_avg)
        acc_epoch_avg = mean(accs_epoch)
        accs.append(acc_epoch_avg)

    # print(losses)
    # print(len(losses))
    loss_acc_logger_csv['loss'] = pd.Series(losses)
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
    task_csv_test = pd.read_csv(task_csv_test_dict[adapt_task])
    # task_csv_test = pd.read_csv(parent_dir + adapt_task + '/' + 'gt_preds_each_da_model_approach1.csv') # THIS IS TEST DATA
    # print(task_csv)

    image_names_test = task_csv_test['Image_Names'].dropna().tolist()
    # print(image_names)
    print('Number images:', len(image_names_test))
    image_gt_labels_test = task_csv_test['Ground Truth'].dropna().tolist()
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

