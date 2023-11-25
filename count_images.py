import pandas as pd
import os
import numpy as np
from statistics import mode, mean, stdev

parent_dir = '/home/njjosselyn/ARL/domain_adaptation/DomainNet_all_correct/DomainNet/cross_val_folds/fold_1/train/'

domains = os.listdir(parent_dir)

for d in domains:
    print('Domain:', d)
    p1 = parent_dir + d
    classes = os.listdir(p1)
    num_imgs_avg_cls = []
    for c in classes:
        # print('Class:', c)
        p2 = p1 + '/' + c
        images = os.listdir(p2)
        num_imgs = len(images)
        num_imgs_avg_cls.append(num_imgs)
        # print('Num. Images:', num_imgs)
    avg_num_imgs = mean(num_imgs_avg_cls)
    print('Avg Num Images:', avg_num_imgs)
