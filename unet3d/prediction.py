import os

import nibabel as nib
import numpy as np
import tables

from .training import load_old_model
from .utils import pickle_load
from .patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices, compute_patch_indices_for_prediction
import pdb
from tqdm import tqdm
import time
from dev_tools.my_tools import print_red
from progressbar import *


def patch_wise_prediction(model, data, brain_width, overlap=0, batch_size=1, permute=False, center_patch=True):
#     pdb.set_trace()
    pdb_set = False
    
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()
    
    brain_wise_image_shape = brain_width[1] - brain_width[0] + 1
    brain_wise_data = data[0,:,brain_width[0,0]:brain_width[1,0]+1,
                               brain_width[0,1]:brain_width[1,1]+1,
                               brain_width[0,2]:brain_width[1,2]+1]
    
    indices = compute_patch_indices_for_prediction(brain_wise_image_shape, patch_size=patch_shape,center_patch=center_patch)
    batch = list()
    i = 0
    if pdb_set:
        pbar = ProgressBar().start()
        print('Predicting patches of single subject...')
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(brain_wise_data, patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)
        if pdb_set:
            pbar.update(int((i-1)*100/(len(indices)-1)))
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(brain_wise_image_shape)
    brain_wise_output = reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)
    
    origin_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    final_output = np.zeros(origin_shape)
    final_output[:,brain_width[0,0]:brain_width[1,0]+1,
                   brain_width[0,1]:brain_width[1,1]+1,
                   brain_width[0,2]:brain_width[1,2]+1] = brain_wise_output

    return final_output



def get_prediction_labels(prediction, threshold=0.5, labels=None):
    label_data = np.argmax(prediction[0], axis=0) + 1
    label_data[np.max(prediction[0], axis=0) < threshold] = 0
    if labels:
        for value in np.unique(label_data).tolist()[1:]:
            label_data[label_data == value] = labels[value - 1]
    label_data = label_data.astype(np.uint8)
    return label_data

def get_prediction_labels_overlap(prediction, threshold=0.5):
#     pdb.set_trace()
    label_data = np.zeros(prediction[0,0].shape)
    label_data[prediction[0,1] >= threshold] = 2
    label_data[prediction[0,0] >= threshold] = 1
    label_data[prediction[0,2] >= threshold] = 4
    label_data = label_data.astype(np.uint8)
    return label_data


def prediction_to_image(prediction, affine, brain_mask, threshold=0.5, labels=None,output_dir='',overlap_label=False):
    '''
    for multi categories classification please refer to Isensee's repository.
    '''
#     pdb.set_trace()
    pdb_set = False
    
    if prediction.shape[1] == 3: 
        if overlap_label:
            data = get_prediction_labels_overlap(prediction, threshold=threshold)
        else:
            data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    masked_output = data * brain_mask
    if np.sum(masked_output - data):
        if pdb_set:
            print_red('changed after mask')
            print_red(output_dir)
            print_red(np.array(np.where(masked_output != data)).shape[1])
        nib.Nifti1Image(data, affine).to_filename(os.path.join(output_dir, "prediction_before_mask.nii.gz"))
    return nib.Nifti1Image(masked_output, affine)


def run_validation_case(data_index, output_dir, model, data_file, training_modalities,
                        threshold=0.5, labels=None, overlap=16, 
                        permute=False, center_patch=True, overlap_label=True, 
                        final_val=False):
#     pdb.set_trace()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = np.load('./affine.npy')
    test_data = np.array([modality_img[data_index,0] 
                      for modality_img in [data_file.root.t1,
                                           data_file.root.t1ce,
                                           data_file.root.flair,
                                           data_file.root.t2]])[np.newaxis]
    for i in range(test_data.shape[1]):
        if i == 0:
            brain_mask = np.copy(test_data[0,i])
            brain_mask[np.nonzero(brain_mask)] = True
        else:
            temp_mask = np.copy(test_data[0,i])
            temp_mask[np.nonzero(temp_mask)] = True
            brain_mask = np.logical_or(brain_mask,temp_mask)
    
    
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))
    
    if not final_val:
        test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
        test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))
    
    brain_width = data_file.root.brain_width[data_index]

    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, brain_width=brain_width,
                                           overlap=overlap, permute=permute, center_patch=center_patch)[np.newaxis]
#     pdb.set_trace()
    prediction_image = prediction_to_image(prediction, affine, brain_mask,
                                           threshold=threshold, labels=labels,output_dir=output_dir,overlap_label=overlap_label)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(output_dir, data_file.root.subject_ids[data_index].decode()+'.nii.gz'))


def run_validation_cases(validation_keys_file, model_file, training_modalities, labels, hdf5_file,
                         output_dir=".", threshold=0.5, overlap=16, 
                         permute=False,center_patch=True, overlap_label=True, final_val = False):
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    data_file = tables.open_file(hdf5_file, "r")
    
    for index in tqdm(validation_indices):
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=training_modalities, labels=labels,
                            threshold=threshold, overlap=overlap, permute=permute, center_patch=center_patch,
                            overlap_label=overlap_label,
                            final_val=final_val)
    data_file.close()
#     pdb.set_trace()


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)
