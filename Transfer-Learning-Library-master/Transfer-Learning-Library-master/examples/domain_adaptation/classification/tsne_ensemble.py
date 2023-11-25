# import torch
import matplotlib
from sklearn import random_projection
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import argparse

parser = argparse.ArgumentParser(description='TSNE Plotting')
parser.add_argument('--tk', '--task', default='clipart_infograph', type=str)
parser.add_argument('--d', '--data', default='test', type=str) # 'test', 'train'
args = parser.parse_args()

num_folds = 5 #2
data_use = args.d
task = args.tk
# models = ['DANN']
models = ['DANN', 'JAN', 'CDAN', 'AFN', 'MCC']
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

# concat_model_names = np.concatenate(list(model_names.values()), axis=0) # this may error, np concat strings...?
concat_arrays = np.concatenate(list(model_npys.values()), axis=0) # what's the model order here???
print(concat_arrays.shape)

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(concat_arrays)
print(X_tsne.shape)

# mdl_nm_chunks = np.split(concat_model_names, len(models)*num_folds)
np_chunks = np.split(X_tsne, len(models)*num_folds)
print('number of model chunks:', len(np_chunks))
print(np_chunks[0].shape)

# c list to subdivide the models over all DA models and fold models
NUM_COLORS = len(models)*num_folds
cm = plt.get_cmap('gist_rainbow')

fig = plt.figure()
ax = fig.add_subplot(111)
cc = ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# bb=0
for g in range(0, len(np_chunks)):
    Xt = np_chunks[g]
    mdl_nm = mdls_all[g]
    ax.scatter(Xt[:,0], Xt[:,1], label=mdl_nm) #f'{bb}'
    # bb+=1

plt.savefig('feature_lists/all_tsne/' + data_use + '_' + task + '_tsne_legend_none' + '.png')

ax.legend()

plt.savefig('feature_lists/all_tsne/' + data_use + '_' + task + '_tsne_legend_inside' + '.png')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('feature_lists/all_tsne/' + data_use + '_' + task + '_tsne_legend_outside' + '.png')


for g in range(0, len(np_chunks)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xt = np_chunks[g]
    mdl_nm = mdls_all[g]
    ax.scatter(Xt[:,0], Xt[:,1], label=mdl_nm) #f'{bb}'
    ax.legend()
    plt.savefig('feature_lists/individual_tsne/'+ data_use + '_' + task + '_' + mdl_nm + '.png')


if num_folds*len(models) ==25: # 4

    dann_chunks = [np_chunks[0], np_chunks[1], np_chunks[2], np_chunks[3], np_chunks[4]]
    jan_chunks = [np_chunks[5], np_chunks[6], np_chunks[7], np_chunks[8], np_chunks[9]]
    cdan_chunks = [np_chunks[10], np_chunks[11], np_chunks[12], np_chunks[13], np_chunks[14]]
    afn_chunks = [np_chunks[15], np_chunks[16], np_chunks[17], np_chunks[18], np_chunks[19]]
    mcc_chunks = [np_chunks[20], np_chunks[21], np_chunks[22], np_chunks[23], np_chunks[24]]
    # dann_chunks = [np_chunks[0], np_chunks[1]]
    # afn_chunks = [np_chunks[2], np_chunks[3]]

    model_chunks = [dann_chunks, jan_chunks, cdan_chunks, afn_chunks, mcc_chunks]
    # model_chunks = [dann_chunks, afn_chunks]

    # model order in eat list: DANN, JAN, CDAN, AFN, MCC
    fold_chunks_1 = [np_chunks[0], np_chunks[5], np_chunks[10], np_chunks[15], np_chunks[20]] # fold1 all DA models
    fold_chunks_2 = [np_chunks[1], np_chunks[6], np_chunks[11], np_chunks[16], np_chunks[21]] # fold2 all DA models
    fold_chunks_3 = [np_chunks[2], np_chunks[7], np_chunks[12], np_chunks[17], np_chunks[22]]
    fold_chunks_4 = [np_chunks[3], np_chunks[8], np_chunks[13], np_chunks[18], np_chunks[23]]
    fold_chunks_5 = [np_chunks[4], np_chunks[9], np_chunks[14], np_chunks[19], np_chunks[24]]
    # fold_chunks_1 = [np_chunks[0], np_chunks[2]]  # fold1 all DA models
    # fold_chunks_2 = [np_chunks[1], np_chunks[3]]  # fold2 all DA models

    fold_chunks = [fold_chunks_1, fold_chunks_2, fold_chunks_3, fold_chunks_4, fold_chunks_5]
    # fold_chunks = [fold_chunks_1, fold_chunks_2]

    fold_list = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    # fold_list = ['fold_1', 'fold_2']

    # plot 5 folds for 1 DA model, for each DA model
    for md_chunks in range(0, len(model_chunks)):
        md_chunk = model_chunks[md_chunks]
        what_model = models[md_chunks]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ffld = 1
        for mdc in md_chunk:
            fold_ffld = what_model + ' fold_'+str(ffld)
            ax.scatter(mdc[:, 0], mdc[:, 1], label=fold_ffld)  # f'{bb}'
            ffld += 1
        ax.legend()
        plt.savefig('feature_lists/sets5_tsne/' + data_use + '_' + task + '_5folds_' + what_model + '.png')

    # plot 5 DA models for 1 fold, for each fold
    for fd_chunks in range(0, len(fold_chunks)):
        fd_chunk = fold_chunks[fd_chunks]
        what_fold = fold_list[fd_chunks]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for mdcfs in range(0, len(fd_chunk)):
            mdcf = fd_chunk[mdcfs]
            mdlname = models[mdcfs]
            label_name = mdlname + ' ' + what_fold
            ax.scatter(mdcf[:, 0], mdcf[:, 1], label=label_name)
        ax.legend()
        plt.savefig('feature_lists/sets5_tsne/' + data_use + '_' + task + '_5DAmods_' + what_fold + '.png')

