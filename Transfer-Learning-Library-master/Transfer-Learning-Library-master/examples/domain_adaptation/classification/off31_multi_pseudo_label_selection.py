from statistics import mean, stdev, multimode, median, median_high, median_low
from scipy.stats import mode
from itertools import zip_longest
import torch
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score



# parent_dir = 'C:\\Users\\Nick\\Desktop\\WPI\\ARL\\Domain_Adaptation\\Codes\\Results\\ARL_DA\\journal_extension\\DA_models\\necessary_files\\'

train_or_test = 'test_data' # train_fold1_data or test_data

parent_dir = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/off31_eval_models/' + train_or_test + '/'

# adapt_tasks = ['Office31_a2d', 'Office31_a2w', 'Office31_d2w', 'Office31_d2a', 'Office31_w2a', 'Office31_w2d']

targets = ['a', 'd', 'w'] # the 3 possible target domains in this case for multi-source code

adapt_tasks = {'a': ['Office31_w2a', 'Office31_d2a'],
                 'd': ['Office31_a2d', 'Office31_w2d'],
                 'w': ['Office31_a2w', 'Office31_d2w']}

# adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p', 'DomainNet_c2q', 'DomainNet_c2r', 'DomainNet_c2s',
#               'DomainNet_i2c', 'DomainNet_i2p', 'DomainNet_i2q', 'DomainNet_i2r', 'DomainNet_i2s',
#               'DomainNet_p2c', 'DomainNet_p2i','DomainNet_p2q', 'DomainNet_p2r', 'DomainNet_p2s',
#               'DomainNet_q2c', 'DomainNet_q2i','DomainNet_q2p','DomainNet_q2r', 'DomainNet_q2s',
#               'DomainNet_r2c', 'DomainNet_r2i','DomainNet_r2p', 'DomainNet_r2q', 'DomainNet_r2s',
#               'DomainNet_s2c', 'DomainNet_s2i','DomainNet_s2p', 'DomainNet_s2q', 'DomainNet_s2r']

# adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p']
# adapt_tasks = ['DomainNet_c2i']

data_pool_excel_file = 'data_pool_matrix.xlsx'

dann_keep_preds = ['dann_subset_exps_fold1', 'dann_subset_exps_fold2', 'dann_subset_exps_fold3', 'dann_subset_exps_fold4', 'dann_subset_exps_fold5']
afn_keep_preds = ['afn_subset_exps_fold1', 'afn_subset_exps_fold2', 'afn_subset_exps_fold3', 'afn_subset_exps_fold4', 'afn_subset_exps_fold5']
cdan_keep_preds = ['cdan_subset_exps_fold1', 'cdan_subset_exps_fold2', 'cdan_subset_exps_fold3', 'cdan_subset_exps_fold4', 'cdan_subset_exps_fold5']
jan_keep_preds = ['jan_subset_exps_fold1', 'jan_subset_exps_fold2', 'jan_subset_exps_fold3', 'jan_subset_exps_fold4', 'jan_subset_exps_fold5']
mcc_keep_preds = ['mcc_subset_exps_fold1', 'mcc_subset_exps_fold2', 'mcc_subset_exps_fold3', 'mcc_subset_exps_fold4', 'mcc_subset_exps_fold5']

dann_keep_conf = ['dann_subset_exps_fold1_conf', 'dann_subset_exps_fold2_conf', 'dann_subset_exps_fold3_conf', 'dann_subset_exps_fold4_conf', 'dann_subset_exps_fold5_conf']
afn_keep_conf = ['afn_subset_exps_fold1_conf', 'afn_subset_exps_fold2_conf', 'afn_subset_exps_fold3_conf', 'afn_subset_exps_fold4_conf', 'afn_subset_exps_fold5_conf']
cdan_keep_conf = ['cdan_subset_exps_fold1_conf', 'cdan_subset_exps_fold2_conf', 'cdan_subset_exps_fold3_conf', 'cdan_subset_exps_fold4_conf', 'cdan_subset_exps_fold5_conf']
jan_keep_conf = ['jan_subset_exps_fold1_conf', 'jan_subset_exps_fold2_conf', 'jan_subset_exps_fold3_conf', 'jan_subset_exps_fold4_conf', 'jan_subset_exps_fold5_conf']
mcc_keep_conf = ['mcc_subset_exps_fold1_conf', 'mcc_subset_exps_fold2_conf', 'mcc_subset_exps_fold3_conf', 'mcc_subset_exps_fold4_conf', 'mcc_subset_exps_fold5_conf']


print()

pseudo_labels_tasks = {}
mode_pseudo_labels_tasks = {}
gt_tasks = {}
for t in targets:
    print('TARGET TASK: ', t)
    print()
    adapt_tasks_t = adapt_tasks[t]
    # preds_init_array = np.empty((0, 25))
    # conf_init_array = np.empty((0, 25))

    data_pool_all_preds_list = []
    data_pool_all_confs_list = []
    for adapt_task in adapt_tasks_t:
        print('ADAPT TASK:', adapt_task)
        print()
        data_pool_pth = parent_dir + adapt_task + '/' + data_pool_excel_file

        data_pool_dann_ = pd.read_excel(data_pool_pth, sheet_name='dann_subset_exps') #, index_col=0)
        data_pool_jan_ = pd.read_excel(data_pool_pth, sheet_name='jan_subset_exps') #, index_col=0)
        data_pool_cdan_ = pd.read_excel(data_pool_pth, sheet_name='cdan_subset_exps') #, index_col=0)
        data_pool_afn_ = pd.read_excel(data_pool_pth, sheet_name='afn_subset_exps') #, index_col=0)
        data_pool_mcc_ = pd.read_excel(data_pool_pth, sheet_name='mcc_subset_exps') #, index_col=0)

        gt_label_ = data_pool_dann_['GT Number'].dropna().tolist()
        # gt_name = data_pool_dann_['Image Names'].dropna().tolist()

        gt_label = list(map(int, gt_label_))

        gt_tasks[t] = gt_label

        # data_pool_dann_preds = data_pool_dann_.drop(columns=['Image Names', 'GT Number', 'dann_subset_exps_fold1_conf', 'dann_subset_exps_fold2_conf', 'dann_subset_exps_fold3_conf', 'dann_subset_exps_fold4_conf', 'dann_subset_exps_fold5_conf']).to_numpy()
        # data_pool_jan_preds = data_pool_jan_.drop(columns=['Image Names', 'GT Number', 'jan_subset_exps_fold1_conf', 'jan_subset_exps_fold2_conf', 'jan_subset_exps_fold3_conf', 'jan_subset_exps_fold4_conf', 'jan_subset_exps_fold5_conf']).to_numpy()
        # data_pool_cdan_preds = data_pool_cdan_.drop(columns=['Image Names', 'GT Number', 'cdan_subset_exps_fold1_conf', 'cdan_subset_exps_fold2_conf', 'cdan_subset_exps_fold3_conf', 'cdan_subset_exps_fold4_conf', 'cdan_subset_exps_fold5_conf']).to_numpy()
        # data_pool_afn_preds = data_pool_afn_.drop(columns=['Image Names', 'GT Number', 'afn_subset_exps_fold1_conf', 'afn_subset_exps_fold2_conf', 'afn_subset_exps_fold3_conf', 'afn_subset_exps_fold4_conf', 'afn_subset_exps_fold5_conf']).to_numpy()
        # data_pool_mcc_preds = data_pool_mcc_.drop(columns=['Image Names', 'GT Number', 'mcc_subset_exps_fold1_conf', 'mcc_subset_exps_fold2_conf', 'mcc_subset_exps_fold3_conf', 'mcc_subset_exps_fold4_conf', 'mcc_subset_exps_fold5_conf']).to_numpy()

        data_pool_dann_preds = data_pool_dann_[dann_keep_preds].to_numpy()
        data_pool_jan_preds = data_pool_jan_[jan_keep_preds].to_numpy()
        data_pool_cdan_preds = data_pool_cdan_[cdan_keep_preds].to_numpy()
        data_pool_afn_preds = data_pool_afn_[afn_keep_preds].to_numpy()
        data_pool_mcc_preds = data_pool_mcc_[mcc_keep_preds].to_numpy()

        # data_pool_dann_conf = data_pool_dann_.drop(columns=['Image Names', 'GT Number', 'dann_subset_exps_fold1', 'dann_subset_exps_fold2', 'dann_subset_exps_fold3', 'dann_subset_exps_fold4', 'dann_subset_exps_fold5']).to_numpy()
        # data_pool_jan_conf = data_pool_jan_.drop(columns=['Image Names', 'GT Number', 'jan_subset_exps_fold1', 'jan_subset_exps_fold2', 'jan_subset_exps_fold3', 'jan_subset_exps_fold4', 'jan_subset_exps_fold5']).to_numpy()
        # data_pool_cdan_conf = data_pool_cdan_.drop(columns=['Image Names', 'GT Number', 'cdan_subset_exps_fold1', 'cdan_subset_exps_fold2', 'cdan_subset_exps_fold3', 'cdan_subset_exps_fold4', 'cdan_subset_exps_fold5']).to_numpy()
        # data_pool_afn_conf = data_pool_afn_.drop(columns=['Image Names', 'GT Number', 'afn_subset_exps_fold1', 'afn_subset_exps_fold2', 'afn_subset_exps_fold3', 'afn_subset_exps_fold4', 'afn_subset_exps_fold5']).to_numpy()
        # data_pool_mcc_conf = data_pool_mcc_.drop(columns=['Image Names', 'GT Number', 'mcc_subset_exps_fold1', 'mcc_subset_exps_fold2', 'mcc_subset_exps_fold3', 'mcc_subset_exps_fold4', 'mcc_subset_exps_fold5']).to_numpy()

        data_pool_dann_conf = data_pool_dann_[dann_keep_conf].to_numpy()
        data_pool_jan_conf = data_pool_jan_[jan_keep_conf].to_numpy()
        data_pool_cdan_conf = data_pool_cdan_[cdan_keep_conf].to_numpy()
        data_pool_afn_conf = data_pool_afn_[afn_keep_conf].to_numpy()
        data_pool_mcc_conf = data_pool_mcc_[mcc_keep_conf].to_numpy()

        # append each of the 5 df side by side, for conf and pred dfs
        data_pool_all_preds1 = np.hstack(( data_pool_dann_preds, data_pool_jan_preds, data_pool_cdan_preds, data_pool_afn_preds, data_pool_mcc_preds ))
        data_pool_all_preds_list.append(data_pool_all_preds1)
        # print(data_pool_all_preds.shape)

        data_pool_all_confs1 = np.hstack(( data_pool_dann_conf, data_pool_jan_conf, data_pool_cdan_conf, data_pool_afn_conf, data_pool_mcc_conf ))
        data_pool_all_confs_list.append(data_pool_all_confs1)
        # print(data_pool_all_confs.shape)
        # print()

        print()
        print('------------------------------------------------------------------------')
        print('Print Sizes:')
        print('data_pool_dann_preds', data_pool_dann_preds.shape)
        print('data_pool_jan_preds', data_pool_jan_preds.shape)
        print('data_pool_cdan_preds', data_pool_cdan_preds.shape)
        print('data_pool_afn_preds', data_pool_afn_preds.shape)
        print('data_pool_mcc_preds', data_pool_mcc_preds.shape)
        print('data_pool_all_preds1', data_pool_all_preds1.shape)
        print('data_pool_all_confs1', data_pool_all_confs1.shape)
        print('------------------------------------------------------------------------')
        print()


    # now loop through 'data_pool_all_preds_list' and 'data_pool_all_confs_list' and stack them
    # for dpl in range(0, len(data_pool_all_preds_list)):
    data_pool_all_preds = np.hstack(data_pool_all_preds_list)
    data_pool_all_confs = np.hstack(data_pool_all_confs_list)

    print()
    print('------------------------------------------------------------------------')
    print('data_pool_all_preds_list', len(data_pool_all_preds_list))
    print('data_pool_all_confs_list', len(data_pool_all_confs_list))
    print('------------------------------------------------------------------------')
    print('data_pool_all_preds_list 0 shape', data_pool_all_preds_list[0].shape)
    print('data_pool_all_confs_list 0 shape', data_pool_all_confs_list[0].shape)
    print('data_pool_all_preds_list 1 shape', data_pool_all_preds_list[1].shape)
    print('data_pool_all_confs_list 1 shape', data_pool_all_confs_list[1].shape)
    # print('data_pool_all_preds_list 2 shape', data_pool_all_preds_list[2].shape)
    # print('data_pool_all_confs_list 2 shape', data_pool_all_confs_list[2].shape)
    # print('data_pool_all_preds_list 3 shape', data_pool_all_preds_list[3].shape)
    # print('data_pool_all_confs_list 3 shape', data_pool_all_confs_list[3].shape)
    # print('data_pool_all_preds_list 4 shape', data_pool_all_preds_list[4].shape)
    # print('data_pool_all_confs_list 4 shape', data_pool_all_confs_list[4].shape)
    print('------------------------------------------------------------------------')
    print('Print Sizes:')
    print('data_pool_all_preds', data_pool_all_preds.shape)
    print('data_pool_all_confs', data_pool_all_confs.shape)
    print('------------------------------------------------------------------------')
    print()


    # then hstack these across all 5 sources, column names will overlap though...
    # need to hstack these to themselves cuz the variable will be overwritten, need to initialize an empty array to stack onto
    # hstack across the lists!!!!!

    # i think bump these back one for loop
    # update names to the new hstack-ed array of all 5 sources
    # then get highest conf and mode over all 125

    # The resulting mode_values array will contain the mode of each row. If there are multiple modes for a row, it will return the smallest mode value.
    mode_values_pseudo_preds = mode(data_pool_all_preds, axis=1).mode

    max_conf = np.amax(data_pool_all_confs, axis=1)
    # print(max_conf)
    # print(len(max_conf))

    max_conf_idx = np.argmax(data_pool_all_confs, axis=1)
    # print(max_conf_idx)
    # print(len(max_conf_idx))
    # print()
    # print('------')
    # print()

    # get corresponding pred value for max_conf_idx
    pseudo_preds = [] # pseudo labels list
    for i in range(0, len(max_conf_idx)):
        idx = max_conf_idx[i]
        # print(idx)
        pred_row = data_pool_all_preds[i, idx]
        # print(pred_row)
        # print(len(pred_row))
        # print()
        pseudo_preds.append(pred_row) # pseudo labels list
    # print(pseudo_labels_tasks)
    # print()
    # print(pseudo_preds) # pseudo labels list
    # print(len(pseudo_preds))
    # print()
    # print(gt_label)
    # print(len(gt_label))
    # print()

    pseudo_preds = list(map(int, pseudo_preds))
    pseudo_labels_tasks[t] = pseudo_preds

    mode_values_pseudo_preds = list(map(int, mode_values_pseudo_preds))
    mode_pseudo_labels_tasks[t] = mode_values_pseudo_preds




print('Dictionary:')
print('Conf. Score based')
print(pseudo_labels_tasks.keys())
print('Mode based')
print(mode_pseudo_labels_tasks.keys())
print()


# calculate metrics for each task comparing GT against pseudo label (pred corresponding to highest logit conf score)
print()
print('calculate metrics for each task comparing GT against pseudo label (pred corresponding to highest logit conf score)')
accs = []
f1s = []
bal_accs = []
for k in pseudo_labels_tasks.keys():
    print(k)
    gt = gt_tasks[k]
    y_hat_pred = pseudo_labels_tasks[k]
    # gt = [int(i) for i in gt_tasks[k]]
    # y_hat_pred = [int(i) for i in pseudo_labels_tasks[k]]
    # gt = list(map(int, gt_tasks[k]))
    # y_hat_pred = list(map(int, pseudo_labels_tasks[k]))

    acc = accuracy_score(gt, y_hat_pred)
    f1_weighted = f1_score(gt, y_hat_pred, average='weighted')
    bal_acc = balanced_accuracy_score(gt, y_hat_pred)
    accs.append(acc)
    f1s.append(f1_weighted)
    bal_accs.append(bal_acc)
    print('Accuracy:', acc)
    print('F1:', f1_weighted)
    print('Balanced Accuracy:', bal_acc)
    print()


mean_acc = mean(accs)
mean_f1 = mean(f1s)
mean_bal_acc = mean(bal_accs)

stdev_acc = stdev(accs)
stdev_f1 = stdev(f1s)
stdev_bal_acc = stdev(bal_accs)

print()
print('AVERAGES over all targets:')
print('Average Accuracy', mean_acc)
print('Average F1-Weighted', mean_f1)
print('Average Balanced Accuracy', mean_bal_acc)

print()
print('STDEV over all targets:')
print('STDEV Accuracy', stdev_acc)
print('STDEV F1-Weighted', stdev_f1)
print('STDEV Balanced Accuracy', stdev_bal_acc)




print()
print()
print('calculate metrics for each task comparing GT against pseudo label (pred corresponding to mode)')
accs = []
f1s = []
bal_accs = []
for k in mode_pseudo_labels_tasks.keys():
    print(k)
    gt = gt_tasks[k]
    y_hat_pred = mode_pseudo_labels_tasks[k]
    # gt = [int(i) for i in gt_tasks[k]]
    # y_hat_pred = [int(i) for i in pseudo_labels_tasks[k]]
    # gt = list(map(int, gt_tasks[k]))
    # y_hat_pred = list(map(int, pseudo_labels_tasks[k]))

    acc = accuracy_score(gt, y_hat_pred)
    f1_weighted = f1_score(gt, y_hat_pred, average='weighted')
    bal_acc = balanced_accuracy_score(gt, y_hat_pred)
    accs.append(acc)
    f1s.append(f1_weighted)
    bal_accs.append(bal_acc)
    print('Accuracy:', acc)
    print('F1:', f1_weighted)
    print('Balanced Accuracy:', bal_acc)
    print()


mean_acc = mean(accs)
mean_f1 = mean(f1s)
mean_bal_acc = mean(bal_accs)

stdev_acc = stdev(accs)
stdev_f1 = stdev(f1s)
stdev_bal_acc = stdev(bal_accs)

print()
print('AVERAGES over all targets:')
print('Average Accuracy', mean_acc)
print('Average F1-Weighted', mean_f1)
print('Average Balanced Accuracy', mean_bal_acc)

print()
print('STDEV over all targets:')
print('STDEV Accuracy', stdev_acc)
print('STDEV F1-Weighted', stdev_f1)
print('STDEV Balanced Accuracy', stdev_bal_acc)



print()
print('saving...')
# save GT and pseudo labels for training ensemble

# zip all the values together
zl_pseudo = list(zip_longest(*pseudo_labels_tasks.values()))
# create dataframe
df_pseudo = pd.DataFrame(zl_pseudo, columns=pseudo_labels_tasks.keys())


# zip all the values together
zl_pseudo_mode = list(zip_longest(*mode_pseudo_labels_tasks.values()))
# create dataframe
df_pseudo_mode = pd.DataFrame(zl_pseudo_mode, columns=mode_pseudo_labels_tasks.keys())

# zip all the values together
zl_gt = list(zip_longest(*gt_tasks.values()))
# create dataframe
df_gt = pd.DataFrame(zl_gt, columns=gt_tasks.keys())

# pseudo_labels_tasks_df = pd.DataFrame.from_dict(pseudo_labels_tasks)
# gt_tasks_df = pd.DataFrame.from_dict(gt_tasks)

# print()
# print()
# print(df_pseudo)
# print()
# print(df_gt)
# print()

# saving files
df_pseudo.to_csv(parent_dir + 'off31_multi_pseudo_labels_using_conf_score_' + train_or_test + '.csv')
df_gt.to_csv(parent_dir + 'off31_multi_gt_labels_' + train_or_test + '.csv')

df_pseudo_mode.to_csv(parent_dir + 'off31_multi_pseudo_labels_using_mode_' + train_or_test + '.csv')



