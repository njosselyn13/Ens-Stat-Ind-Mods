# import torch
import matplotlib
from sklearn import random_projection
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='TSNE Plotting')
parser.add_argument('--tk', '--task', default='clipart_infograph', type=str)
parser.add_argument('--d', '--data', default='test', type=str) # 'test', 'train'
args = parser.parse_args()

num_folds = 5 #2
data_use = args.d
task = args.tk
# models = ['DANN']
models = ['DANN'] #, 'JAN', 'CDAN', 'AFN', 'MCC']
# models = ['DANN', 'AFN']
# models = ['DANN', 'JAN']
print(data_use)
print()

# tasks = ['clipart_infograph', 'clipart_painting', 'clipart_quickdraw', 'clipart_real', 'clipart_sketch',
#                  'infograph_clipart', 'infograph_painting', 'infograph_quickdraw', 'infograph_real', 'infograph_sketch',
#                  'painting_clipart', 'painting_infograph', 'painting_quickdraw', 'painting_real', 'painting_sketch',
#                  'quickdraw_clipart', 'quickdraw_infograph', 'quickdraw_painting', 'quickdraw_real', 'quickdraw_sketch',
#                  'real_clipart', 'real_infograph', 'real_painting', 'real_quickdraw', 'real_sketch',
#                  'sketch_clipart', 'sketch_infograph', 'sketch_painting', 'sketch_quickdraw', 'sketch_real']

model_npys = {}
pth_dict = {}
# model_names = {}
# gauss_dict = {}
print(task)
mdls_all = []
gauss_proj = random_projection.GaussianRandomProjection(n_components=256)
for m in models:
    print()
    print(m)
    for fld in range(1, num_folds+1):
        print(fld)
        if data_use == 'test':
            pth_dict[fld] = 'feature_lists/dataset/' + m + '/TEST/' + task + '_' + str(fld) + '_feature_list.npy'
        elif data_use == 'train':
            pth_dict[fld] = 'feature_lists/dataset/' + m + '/' + task + '_' + str(fld) + '_feature_list.npy'

        if m == 'AFN' and fld == 1:
            print('Found AFN and fold1 model', fld)
            h = np.load(pth_dict[fld])
            h = gauss_proj.fit_transform(h)
            print('projected afn:', h.shape)
        elif m == 'AFN':
            print('Found AFN model', fld)
            h = np.load(pth_dict[fld])
            h = gauss_proj.transform(h)
            print('projected afn:', h.shape)
        else:
            h = np.load(pth_dict[fld])

        model_npys[m+'_'+str(fld)] = h
        # model_names[m+'_'+str(fld)] = m+'_'+str(fld)

        mdls_all.append(m+'_'+str(fld))

        print(model_npys[m+'_'+str(fld)].shape)

num_images = h.shape[0]
print('num images:', num_images)
print('mdls_all list:', mdls_all)
# print('model names dict keys:', model_names.keys())
# print('model names dict vals:', model_names.values())
print(model_npys.keys())
# print(model_npys.values())
print()

# concat_model_names = np.concatenate(list(model_names.values()), axis=0) # this may error, np concat strings...?
concat_arrays = np.concatenate(list(model_npys.values()), axis=1) # what's the model order here???

# select subset of classes
concat_arrays = concat_arrays[0:160]
print('concat arrays shape', concat_arrays.shape)
print()


parent_dir = '/home/njjosselyn/ARL/domain_adaptation/JAN_CDAN/Transfer-Learning-Library-master/Transfer-Learning-Library-master/examples/domain_adaptation/classification/Ensemble_DA_conf/mean_stdev_mode/Ensemble_DomainNet/ORIGINAL_SPLIT/'

task_csv_test = pd.read_csv(parent_dir + 'DomainNet_c2i' + '/' + 'gt_preds_each_da_model_approach1.csv') # THIS IS TEST DATA
# print(task_csv)

image_names_test = task_csv_test['Image_Names'].dropna().tolist()
image_names_test = image_names_test[0:160]
print('image names', image_names_test)
print('Number images:', len(image_names_test))
# image_gt_labels_test = task_csv_test['Ground Truth'].dropna().tolist()

print('num classes using:', len(np.unique(np.array(image_names_test))))

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(concat_arrays) #, y=image_names_test)
print('tsne shape', X_tsne.shape)
# print(get_feature_names_out(X_tsne))

# NUM_COLORS = len(np.unique(np.array(image_names_test))) #345
# print('num colors', NUM_COLORS)
# cm = plt.get_cmap('gist_rainbow')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cc = ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# plt.scatter(x=X_tsne[:,0], y=X_tsne[:,1], c=cc)
# plt.savefig('feature_lists/'+'testtsne3.png')

import seaborn as sns
fig, ax = plt.subplots(1)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=image_names_test, ax=ax, s=25) #, legend=False)
# ax.legend()
plt.legend() #, bbox_to_anchor=(0.5, 1.35), borderaxespad=0)
plt.savefig('feature_lists/'+'tsne_5folds_5cls.png')#, bbox_inches='tight')