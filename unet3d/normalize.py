import os

import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')
from dev_tools.my_tools import minmax_normalize,print_red
import pdb
from progressbar import *
import pickle



def normalize_data_storage(data_storage, offset=0.1, mul_factor=100, save_file='../data/mean_std.pkl'):
    '''
    data_storage is modality_storage_list
    1. -mean/std(all nonzero voxels(brain area) of all images for the same modality)
    2. minmax(each image individually)
    offset and mul_factor are used to make brain voxel distinct from background zero points.
    '''
#     pdb.set_trace()
    print('normalize_data_storage...')
    mean_std_values = {}
    for modality_storage in data_storage:
        means = []
        pbar = ProgressBar().start()
        print('calculate mean value...')
        n_subs = modality_storage.shape[0]
        for i in range(n_subs):
            means.append(np.mean(np.ravel(modality_storage[i])[np.flatnonzero(modality_storage[i])]))
            pbar.update(int(i*100/(n_subs-1)))
        pbar.finish()
        mean = np.mean(means)
        mean_std_values[modality_storage.name + '_mean'] = mean 
        print('mean=',mean)
        
        std_means = []
        pbar = ProgressBar().start()
        print('calculate std value...')
        for i in range(n_subs):
            std_means.append(np.mean(np.power(np.ravel(modality_storage[i])[np.flatnonzero(modality_storage[i])]-mean,2)))
            pbar.update(int(i*100/(n_subs-1)))
        pbar.finish()
        std = np.sqrt(np.mean(std_means))
        mean_std_values[modality_storage.name + '_std'] = std
        print('std=',std)
        
#         pdb.set_trace()
        for i in range(n_subs):
            brain_index = np.nonzero(modality_storage[i])
            temp_img = np.copy(modality_storage[i])
            temp_img[brain_index] = (minmax_normalize((modality_storage[i][brain_index]-mean)/std) + offset)*mul_factor
            modality_storage[i] = temp_img
    print('normalization FINISHED')
    with open(save_file,'wb') as f:
        pickle.dump(mean_std_values,f)
    return
    
def normalize_data_storage_val(data_storage, offset=0.1, mul_factor=100, save_file='../data/mean_std.pkl'):
    print('normalize validation data storage...')
    if not os.path.exists(save_file):
        print_red('There\'s no mean_std.pkl file.')
        return
    with open(save_file,'rb') as f:
        mean_std_values = pickle.load(f)
    for modality_storage in data_storage:
        n_subs = modality_storage.shape[0]
        mean = mean_std_values[modality_storage.name + '_mean']
        std = mean_std_values[modality_storage.name + '_std']
        
#         pdb.set_trace()
        for i in tqdm(range(n_subs)):
            brain_index = np.nonzero(modality_storage[i])
            temp_img = np.copy(modality_storage[i])
            temp_img[brain_index] = (minmax_normalize((modality_storage[i][brain_index]-mean)/std) + offset)*mul_factor
            modality_storage[i] = temp_img
    print('normalization FINISHED')
    return
