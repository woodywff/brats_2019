# uc: unchanged
import os
import pdb
import numpy as np
import tables
import nibabel as nib
from tqdm import tqdm
from .normalize import normalize_data_storage

import sys
sys.path.append('..')
from dev_tools.my_tools import pad_image, print_red


def create_data_file(out_file, n_samples, image_shape, modality_names):
    '''
    create storage in data.h5 
    '''
#     pdb.set_trace()
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    modality_shape = tuple([0, 1] + list(image_shape))
    truth_shape =    tuple([0, 1] + list(image_shape))
    brain_width_shape = (0,2,3)
    
    
    modality_storage_list = [hdf5_file.create_earray(hdf5_file.root, modality_name, tables.Float32Atom(), shape=modality_shape,
                             filters=filters, expectedrows=n_samples) for modality_name in modality_names]
    
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    
    brain_width_storage = hdf5_file.create_earray(hdf5_file.root, 'brain_width', tables.UInt8Atom(), shape=brain_width_shape,
                                            filters=filters, expectedrows=n_samples)
    tumor_width_storage = hdf5_file.create_earray(hdf5_file.root, 'tumor_width', tables.UInt8Atom(), shape=brain_width_shape,
                                            filters=filters, expectedrows=n_samples)
    
    return hdf5_file, [modality_storage_list, truth_storage, brain_width_storage, tumor_width_storage]



def write_image_data_to_file(image_files, storage_list,
                             image_shape, modality_names, truth_dtype=np.uint8, trivial_check = True):
    '''
    trivial_check: to see if all images share the same affine info and pad_width, the incompliance file names 
                   would be printed in red lines.
                   Also to check the order of modalities when added to the .h5
    '''
#     pdb.set_trace()
    affine_0 = None
    save_affine = True
    print('write_image_data_to_file...')
    for set_of_files in tqdm(image_files):
        if trivial_check:
            if not [os.path.basename(img_file).split('.')[0] for img_file in set_of_files] == modality_names + ['truth']:
                print('wrong order of modalities')
                print_red(image_nii_path)
        subject_data = []
        brain_widths = []
        for i,image_nii_path in enumerate(set_of_files):
            img = nib.load(image_nii_path)
            affine = img.affine
            if affine_0 is None:
                affine_0 = affine
            if trivial_check:
                if np.sum(affine_0 - affine):
                    print('affine incompliance:')
                    print_red(image_nii_path)
                    save_affine = False
            img_npy = img.get_data()
            subject_data.append(img_npy)
            
            if i < len(set_of_files)-1: # we don't calculate brain_width for truth
                brain_widths.append(cal_outline(img_npy))
            else:
                tumor_width = cal_outline(img_npy)
                
        start_edge = np.min(brain_widths,axis=0)[0]
        end_edge = np.max(brain_widths,axis=0)[1]
        brain_width = np.vstack((start_edge,end_edge))
        
        if add_data_to_storage(storage_list,
                               subject_data, brain_width, tumor_width, truth_dtype, modality_names = modality_names):
            print_red('modality_storage.name != modality_name')
            print_red(set_of_files)
    print('write_image_data_to_file...FINISHED')
    if save_affine:
        np.save('affine',affine_0)
    return 


def cal_outline(img_npy):
    '''
    return a (2,3) array indicating the outline
    '''
    brain_index = np.asarray(np.nonzero(img_npy))
    start_edge = np.maximum(np.min(brain_index,axis=1)-1,0)
    end_edge = np.minimum(np.max(brain_index,axis=1)+1,img_npy.shape)
    
    return np.vstack((start_edge,end_edge))


def add_data_to_storage(storage_list,
                        subject_data, brain_width, tumor_width, truth_dtype, modality_names):
#     pdb.set_trace()
    modality_storage_list,truth_storage,brain_width_storage,tumor_width_storage = storage_list
    for i in range(len(modality_names)):
        if modality_storage_list[i].name != modality_names[i]:
            print_red('modality_storage.name != modality_name')
            return 1
        modality_storage_list[i].append(np.asarray(subject_data[i])[np.newaxis][np.newaxis])
    if truth_storage.name != 'truth':
        print_red('truth_storage.name != truth')
        return 1
    truth_storage.append(np.asarray(subject_data[-1], dtype=truth_dtype)[np.newaxis][np.newaxis])
    brain_width_storage.append(np.asarray(brain_width, dtype=truth_dtype)[np.newaxis])
    tumor_width_storage.append(np.asarray(tumor_width, dtype=truth_dtype)[np.newaxis])
    return 0

def write_data_to_file(training_data_files, out_file, image_shape, modality_names, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, mean_std_file='../data/mean_std.pkl'):
#     pdb.set_trace()
    n_samples = len(training_data_files)

    hdf5_file, storage_list = create_data_file(out_file,
                                               n_samples=n_samples,
                                               image_shape=image_shape,
                                               modality_names = modality_names)
    modality_storage_list = storage_list[0]
    write_image_data_to_file(training_data_files, 
                             storage_list,
                             image_shape, truth_dtype=truth_dtype, modality_names = modality_names)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(modality_storage_list, save_file=mean_std_file)
    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

