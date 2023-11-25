import pandas as pd
import os
import numpy as np
import math
from statistics import mode, mean, stdev, multimode, median, median_high, median_low
from sklearn.metrics import accuracy_score
from itertools import combinations

adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p', 'DomainNet_c2q', 'DomainNet_c2r', 'DomainNet_c2s',
              'DomainNet_i2c', 'DomainNet_i2p', 'DomainNet_i2q', 'DomainNet_i2r', 'DomainNet_i2s',
              'DomainNet_p2c', 'DomainNet_p2i','DomainNet_p2q', 'DomainNet_p2r', 'DomainNet_p2s',
              'DomainNet_q2c', 'DomainNet_q2i','DomainNet_q2p','DomainNet_q2r', 'DomainNet_q2s',
              'DomainNet_r2c', 'DomainNet_r2i','DomainNet_r2p', 'DomainNet_r2q', 'DomainNet_r2s',
              'DomainNet_s2c', 'DomainNet_s2i','DomainNet_s2p', 'DomainNet_s2q', 'DomainNet_s2r']

# adapt_tasks = ['DomainNet_c2i', 'DomainNet_c2p']

models = ['afn_subset_exps', 'cdan_subset_exps', 'dann_subset_exps', 'jan_subset_exps', 'mcc_subset_exps']
# models = ['afn_subset_exps', 'cdan_subset_exps']

data_type = 'train_data_analysis'
num_folds = 5

parent_pth = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/' + data_type + '/Ensemble_DomainNet/ORIGINAL_SPLIT/'


excel_file = 'data_pool_matrix.xlsx'

# df_save = pd.DataFrame()

sub_avg_dict_list = {}

for adapt_task in adapt_tasks:

    df_save = pd.DataFrame()

    print(adapt_task)
    pth_load = parent_pth + adapt_task + '/' + excel_file
    mod_folds = {}
    for mod in models:
        print(mod)
        df = pd.read_excel(pth_load, sheet_name=mod)
        # print(mod)
        # print(df)
        # print()
        image_names = df['Image Names'].dropna().tolist()
        gt = df['GT Number'].dropna().tolist()
        # mod_folds['image_names'] = df['Image Names'].dropna().tolist()
        # mod_folds['GT'] = df['GT Number'].dropna().tolist()
        for fld in range(1, num_folds+1):
            mod_folds[mod+'_fold_'+str(fld)] = df[mod+'_fold'+str(fld)].dropna().tolist()



    # probabilities of each individual model
    m_acc_1 = {}
    m_acc_0 = {}
    for md in models:
        # print(md)
        for fld in range(1, num_folds + 1):
            # print(fld)
            m_acc_1[md+'_fold_'+str(fld)] = accuracy_score(gt, mod_folds[md+'_fold_'+str(fld)]) # p(m1=1))
            m_acc_0[md + '_fold_' + str(fld)] = 1 - accuracy_score(gt, mod_folds[md + '_fold_' + str(fld)])  # p(m1=0))
    # print(m_acc_1)
    # print(m_acc_0)
    # print()

    # # pairs of models, all 4 combos per pair
    # print(type(mod_folds.keys()))
    # print(mod_folds.keys())
    # print(type(list(mod_folds.keys())))
    # print(list(mod_folds.keys()))
    mod_keys_list = list(mod_folds.keys())
    # print("The original list : " + str(mod_keys_list))
    mod_pairs_list = list(combinations(mod_keys_list, 2))
    # print("All possible pairs : " + str(mod_pairs_list))
    # print("Number pairs:", len(mod_pairs_list))
    # print()

    mod_pair_results = []

    m1_1_m2_1_dict = {}
    m1_1_m2_0_dict = {}
    m1_0_m2_1_dict = {}
    m1_0_m2_0_dict = {}

    prod_m1_1_m2_1_dict = {}
    prod_m1_1_m2_0_dict = {}
    prod_m1_0_m2_1_dict = {}
    prod_m1_0_m2_0_dict = {}

    sub_pairs_avg_dict = {}

    for pair in mod_pairs_list:
        print('Pair:', pair)
        m1 = pair[0]
        m2 = pair[1]
        print('Model 1:', m1)
        print('Model 2:', m2)

        m_acc_1_m1 = m_acc_1[m1] # model 1's True (1) acc (i.e. acc)
        m_acc_0_m1 = m_acc_0[m1] # model 1's False (0) acc (i.e. 1-acc)

        m_acc_1_m2 = m_acc_1[m2] # model 2's True (1) acc (i.e. acc)
        m_acc_0_m2 = m_acc_0[m2] # model 2's False (0) acc (i.e. 1-acc)

        m1_vals = mod_folds[m1]
        # print('Length m1_vals:', len(m1_vals))
        m2_vals = mod_folds[m2]
        # print('Length m2_vals:', len(m2_vals))

        prod_m1_1_m2_1 = m_acc_1_m1 * m_acc_1_m2
        prod_m1_1_m2_0 = m_acc_1_m1 * m_acc_0_m2
        prod_m1_0_m2_1 = m_acc_0_m1 * m_acc_1_m2
        prod_m1_0_m2_0 = m_acc_0_m1 * m_acc_0_m2

        # now do the 4 joint prob combos for this model pair
        m1_1_m2_1 = 0
        m1_1_m2_0 = 0
        m1_0_m2_1 = 0
        m1_0_m2_0 = 0

        for i in range(0, len(gt)):
            num_images = len(gt)
            gtv = gt[i]
            m1v = m1_vals[i]
            m2v = m2_vals[i]
            if gtv == m1v and gtv == m2v:
                m1_1_m2_1 += 1
                # prod_m1_1_m2_1 = m_acc_1_m1*m_acc_1_m2
            elif gtv == m1v and gtv != m2v:
                m1_1_m2_0 += 1
                # prod_m1_1_m2_0 = m_acc_1_m1*m_acc_0_m2
            elif gtv != m1v and gtv == m2v:
                m1_0_m2_1 += 1
                # prod_m1_0_m2_1 = m_acc_0_m1*m_acc_1_m2
            elif gtv != m1v and gtv != m2v:
                m1_0_m2_0 += 1
                # prod_m1_0_m2_0 = m_acc_0_m1*m_acc_0_m2
            else:
                print('No rules satisfied...')
                print('gt:', gtv)
                print('m1:', m1v)
                print('m2:', m2v)
        m1_1_m2_1_dict[m1+'_'+m2] = m1_1_m2_1/num_images
        m1_1_m2_0_dict[m1+'_'+m2] = m1_1_m2_0/num_images
        m1_0_m2_1_dict[m1+'_'+m2] = m1_0_m2_1/num_images
        m1_0_m2_0_dict[m1+'_'+m2] = m1_0_m2_0/num_images
        # print(m1_1_m2_1/num_images)
        # print(m1_1_m2_0/num_images)
        # print(m1_0_m2_1/num_images)
        # print(m1_0_m2_0/num_images)
        # print()
        prod_m1_1_m2_1_dict[m1+'_'+m2] = prod_m1_1_m2_1
        prod_m1_1_m2_0_dict[m1+'_'+m2] = prod_m1_1_m2_0
        prod_m1_0_m2_1_dict[m1+'_'+m2] = prod_m1_0_m2_1
        prod_m1_0_m2_0_dict[m1+'_'+m2] = prod_m1_0_m2_0

    print()
    print('p(m1=1), p(m2=1):', m1_1_m2_1_dict)
    print('p(m1=1), p(m2=0):', m1_1_m2_0_dict)
    print('p(m1=0), p(m2=1):', m1_0_m2_1_dict)
    print('p(m1=0), p(m2=0):', m1_0_m2_0_dict)
    print()
    print('prod p(m1=1)*p(m2=1):', prod_m1_1_m2_1_dict)
    print('prod p(m1=1)*p(m2=0):', prod_m1_1_m2_0_dict)
    print('prod p(m1=0)*p(m2=1):', prod_m1_0_m2_1_dict)
    print('prod p(m1=0)*p(m2=0):', prod_m1_0_m2_0_dict)
    print()
    print('m_acc_1:', m_acc_1)
    print('m_acc_0:', m_acc_0)
    print()

    for pr in mod_pairs_list:
        print('Pair:', pr)
        m1 = pr[0]
        m2 = pr[1]
        print('Model 1:', m1)
        print('Model 2:', m2)
        m1_m2_pair = m1 + '_' + m2
        print('Model 1 and Model 2 Pair:', m1_m2_pair)
        sub_11 = abs(m1_1_m2_1_dict[m1_m2_pair] - prod_m1_1_m2_1_dict[m1_m2_pair])
        sub_10 = abs(m1_1_m2_0_dict[m1_m2_pair] - prod_m1_1_m2_0_dict[m1_m2_pair])
        sub_01 = abs(m1_0_m2_1_dict[m1_m2_pair] - prod_m1_0_m2_1_dict[m1_m2_pair])
        sub_00 = abs(m1_0_m2_0_dict[m1_m2_pair] - prod_m1_0_m2_0_dict[m1_m2_pair])

        sub_avg = (sub_11 + sub_10 + sub_01 + sub_00)/4
        print(sub_avg)

        sub_pairs_avg_dict[m1_m2_pair] = sub_avg

    sub_avg_dict_list[adapt_task] = sub_pairs_avg_dict
    print()
    print('Average Difference each model pair:', sub_pairs_avg_dict)

    df_save = pd.DataFrame.from_dict(sub_pairs_avg_dict, orient='index')
    df_save.to_csv(parent_pth + 'stat_ind_res/' + adapt_task + '.csv')

    print()
    print('num images:', num_images)
    print()




