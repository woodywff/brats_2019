import os
import copy
from random import shuffle
import itertools

import numpy as np

from .utils import pickle_dump, pickle_load
from .patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data, compute_patch_indices_for_prediction
from .augment import augment_data, random_permutation_x_y

import pdb
from dev_tools.my_tools import print_red
from tqdm import tqdm
import time


def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_start_offset=None,
                                           validation_batch_size=None, skip_blank=True, permute=False,num_model=1,
                                           pred_specific=False, overlap_label=True,
                                           for_final_val=False):
#     pdb.set_trace()
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)
    if for_final_val:
        training_list = training_list + validation_list

    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=validation_patch_overlap,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute,
                                        num_model=num_model,
                                        pred_specific=pred_specific,
                                        overlap_label=overlap_label)
    
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,
                                          num_model=num_model,
                                          pred_specific=pred_specific,
                                          overlap_label=overlap_label)

    # Set the number of training and testing samples per epoch correctly
#     pdb.set_trace()
    if os.path.exists('num_patches_training.npy'):
        num_patches_training = int(np.load('num_patches_training.npy'))
    else:
        num_patches_training = get_number_of_patches(data_file, training_list, patch_shape,
                                                       skip_blank=skip_blank,
                                                       patch_start_offset=training_patch_start_offset,
                                                       patch_overlap=validation_patch_overlap,
                                                       pred_specific=pred_specific)
        np.save('num_patches_training', num_patches_training)
    num_training_steps = get_number_of_steps(num_patches_training, batch_size)
    print("Number of training steps in each epoch: ", num_training_steps)

    if os.path.exists('num_patches_val.npy'):
        num_patches_val = int(np.load('num_patches_val.npy'))
    else:
        num_patches_val = get_number_of_patches(data_file, validation_list, patch_shape,
                                                 skip_blank=skip_blank,
                                                 patch_overlap=validation_patch_overlap,
                                                 pred_specific=pred_specific)
        np.save('num_patches_val', num_patches_val)
    num_validation_steps = get_number_of_steps(num_patches_val, validation_batch_size)
    print("Number of validation steps in each epoch: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps



def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    '''
    Splits the data into the training and validation indices list.
    '''
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.truth.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing




def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                   shuffle_index_list=True, skip_blank=True, permute=False, num_model=1, pred_specific=False,overlap_label=False):
#     pdb.set_trace()

    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file, patch_shape,
                                                 patch_overlap, patch_start_offset,pred_specific=pred_specific)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
                     skip_blank=skip_blank, permute=permute)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels, num_model=num_model,overlap_label=overlap_label)
#                 convert_data(x_list, y_list, n_labels=n_labels, labels=labels, num_model=num_model)
                x_list = list()
                y_list = list()



def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True,pred_specific=False):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file, patch_shape, patch_overlap,
                                             patch_start_offset,pred_specific=pred_specific)
        count = 0
        for index in tqdm(index_list):
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)


def create_patch_index_list(index_list, data_file, patch_shape, patch_overlap, patch_start_offset=None, pred_specific=False):
    patch_index = list()
    for index in index_list:
        brain_width = data_file.root.brain_width[index]
        image_shape = brain_width[1] - brain_width[0] + 1
        if pred_specific:
            patches = compute_patch_indices_for_prediction(image_shape, patch_shape)
        else:
            if patch_start_offset is not None:
                random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
                patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
            else:
                patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False):
    '''
    add qualified x,y to the generator list
    '''
#     pdb.set_trace()
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    
    if np.sum(truth) == 0:
        return
    if augment:
        affine = np.load('affine.npy')
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)


def get_data_from_file(data_file, index, patch_shape=None):
#     pdb.set_trace()
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        brain_width = data_file.root.brain_width[index]
        x = np.array([modality_img[index,0,
                                   brain_width[0,0]:brain_width[1,0]+1,
                                   brain_width[0,1]:brain_width[1,1]+1,
                                   brain_width[0,2]:brain_width[1,2]+1] 
                      for modality_img in [data_file.root.t1,
                                           data_file.root.t1ce,
                                           data_file.root.flair,
                                           data_file.root.t2]])
        y = data_file.root.truth[index, 0,
                                 brain_width[0,0]:brain_width[1,0]+1,
                                 brain_width[0,1]:brain_width[1,1]+1,
                                 brain_width[0,2]:brain_width[1,2]+1]
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None, num_model=1,overlap_label=False):
#     pdb.set_trace()
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        if overlap_label:
            y = get_multi_class_labels_overlap(y, n_labels=n_labels, labels=labels)
        else:
            y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    if num_model == 1:
        return x, y
    else:
        return [x]*num_model, y


def get_multi_class_labels_overlap(data, n_labels=3, labels=(1,2,4)):
    """
    4: ET
    1+4: TC
    1+2+4: WT
    """
#     pdb.set_trace()
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    
    y[:,0][np.logical_or(data[:,0] == 1,data[:,0] == 4)] = 1    #1
    y[:,1][np.logical_or(data[:,0] == 1,data[:,0] == 2, data[:,0] == 4)] = 1 #2
    y[:,2][data[:,0] == 4] = 1    #4
    return y