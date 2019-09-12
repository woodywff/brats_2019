import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

import pdb
from train_model import config
from tqdm import tqdm
from dev_tools.my_tools import print_red


def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + ".nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))
    return

def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def normalize_image(in_file, out_file, bias_correction=True):
    if not os.path.exists(out_file):
        if bias_correction:
            correct_bias(in_file, out_file)
        else:
            shutil.copy(in_file, out_file)
    return out_file

def check_origin(in_file, in_file2):
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_file)

def convert_brats_folder(in_folder, out_folder, truth_name='seg', no_bias_correction_modalities=None, bias_correct=True):
#     pdb.set_trace()
    for name in config["all_modalities"]:
        try:
            image_file = get_image(in_folder, name)
        except RuntimeError as error:
            if name == 't1ce':
                print_red(in_fold)
                image_file = get_image(in_folder, 't1Gd')
                truth_name = "GlistrBoost_ManuallyCorrected"
            else:
                raise error

        out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
        
        if bias_correct:
            perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
            normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
        else:
            if not os.path.exists(out_file):
                shutil.copy(image_file, out_file)
    
    # copy the truth file only for training dataset
    if in_folder.split('/')[-2] == 'val':
        return
    try:
        truth_file = get_image(in_folder, truth_name)
    except RuntimeError:
        truth_file = get_image(in_folder, truth_name.split("_")[0])

    out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
    if not os.path.exists(out_file):
        shutil.copy(truth_file, out_file)
    check_origin(out_file, get_image(in_folder, config["all_modalities"][0]))
    
    return

def convert_brats_data(brats_folder, out_folder, bias_correct=True, overwrite=True, no_bias_correction_modalities=("flair",)):
    """
    Preprocesses the BRATS data and writes it to a given output folder. 
    :param brats_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param bias_correct: if False, just copy the original images to preprocessed folders.
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
#     pdb.set_trace()
    
    for subject_folder in tqdm(glob.glob(os.path.join(brats_folder, "*", "*"))):
#         continue
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_brats_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities,bias_correct=bias_correct)
        else:
            print(subject_folder)

    return