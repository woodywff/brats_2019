# uc: unchanged
import os
import pdb
import numpy as np
import tables
import nibabel as nib
from tqdm import tqdm


import sys
sys.path.append('..')
from dev_tools.my_tools import print_red
from unet3d.normalize import normalize_data_storage_val
from unet3d.data import cal_outline


def create_data_file(out_file, n_samples, image_shape, modality_names):
#     pdb.set_trace()
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    modality_shape = tuple([0, 1] + list(image_shape))
    brain_width_shape = (0,2,3)
    
    
    modality_storage_list = [hdf5_file.create_earray(hdf5_file.root, modality_name, tables.Float32Atom(), shape=modality_shape,
                             filters=filters, expectedrows=n_samples) for modality_name in modality_names]
    
    brain_width_storage = hdf5_file.create_earray(hdf5_file.root, 'brain_width', tables.UInt8Atom(), shape=brain_width_shape,
                                            filters=filters, expectedrows=n_samples)
    
    return hdf5_file, modality_storage_list, brain_width_storage



def write_image_data_to_file(image_files, data_storage,brain_width_storage, 
                             image_shape, modality_names, trivial_check = True):
    '''
    trivial_check: to see if all images share the same affine info and pad_width, the incompliance file names 
                   would be printed in red lines.
                   Also to check the order of modalities when added to the .h5
    '''
#     pdb.set_trace()
    affine_0 = np.load('affine.npy')
    
#     temp = 0
    print('write_image_data_to_file...')
    for set_of_files in tqdm(image_files):
        if trivial_check:
            if not [os.path.basename(img_file).split('.')[0] for img_file in set_of_files] == modality_names:
                print('wrong order of modalities')
                print_red(image_nii_path)
        subject_data = []
        brain_widths = []
        for i,image_nii_path in enumerate(set_of_files):
            img = nib.load(image_nii_path)
            affine = img.affine
            if trivial_check:
                if np.sum(affine_0 - affine):
                    print('affine incompliance:')
                    print_red(image_nii_path)
            img_npy = img.get_data()
            subject_data.append(img_npy)
            
            brain_widths.append(cal_outline(img_npy))
                
        start_edge = np.min(brain_widths,axis=0)[0]
        end_edge = np.max(brain_widths,axis=0)[1]
        brain_width = np.vstack((start_edge,end_edge))
        
        if add_data_to_storage(data_storage, brain_width_storage, 
                               subject_data, brain_width, modality_names = modality_names):
            print_red('modality_storage.name != modality_name')
            print_red(set_of_files)
    print('write_image_data_to_file...FINISHED')
    return data_storage


def add_data_to_storage(data_storage, brain_width_storage, 
                        subject_data, brain_width, modality_names):
#     pdb.set_trace()
    for i in range(len(modality_names)):
        if data_storage[i].name != modality_names[i]:
            print_red('modality_storage.name != modality_name')
            return 1
        data_storage[i].append(np.asarray(subject_data[i])[np.newaxis][np.newaxis])
    
    brain_width_storage.append(np.asarray(brain_width, dtype=np.uint8)[np.newaxis])
    return 0

def write_data_to_file(training_data_files, out_file, image_shape, modality_names, subject_ids=None,
                       normalize=True, mean_std_file='../data/mean_std.pkl'):

#     pdb.set_trace()
    n_samples = len(training_data_files)

    hdf5_file, data_storage, brain_width_storage = create_data_file(out_file,
                                                                      n_samples=n_samples,
                                                                      image_shape=image_shape,
                                                                      modality_names = modality_names)

    write_image_data_to_file(training_data_files, 
                                data_storage, brain_width_storage, 
                                image_shape, modality_names = modality_names)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage_val(data_storage, save_file = mean_std_file)
    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

