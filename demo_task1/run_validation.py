import os
from train_model import config
import pdb
import glob
from data_for_val import write_data_to_file
from unet3d.prediction import run_validation_cases
import pickle
from dev_tools.my_tools import my_mkdir, my_makedirs
from tqdm import tqdm
import shutil

def fetch_val_data_files(return_subject_ids=True):
#     pdb.set_trace()
    val_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join("../data", "preprocessed_val_data", "*", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config['all_modalities']:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        val_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return val_data_files, subject_ids
    else:
        return val_data_files


def gen_val_h5():
    if os.path.exists(config['val_data_file']):
        print(config['val_data_file'],'exists already!')
        return

    val_files, subject_ids = fetch_val_data_files()

    write_data_to_file(val_files, 
                        config['val_data_file'], 
                        image_shape=config["image_shape"], 
                        modality_names = config['all_modalities'],
                        subject_ids=subject_ids,
                       mean_std_file = config['mean_std_file'])
    return
    
def mv_results(source_dir,target_dir):
#     print('moving for upload...')
    my_makedirs(target_dir)
    for sub_id in tqdm(os.listdir(source_dir)):
        source_name = os.path.join(source_dir,sub_id,sub_id+'.nii.gz')
        target_name = os.path.join(target_dir,sub_id+'.nii.gz')
        if not os.path.exists(target_name):
            shutil.move(source_name,target_name)
    
def main_run():
    gen_val_h5()
    
    if not os.path.exists(config['val_index_list']):
        with open(config['val_index_list'],'wb') as f:
            pickle.dump(list(range(config['num_val_subjects'])),f)
    print('Validation dataset prediction starts...')        
    run_validation_cases(validation_keys_file=config['val_index_list'],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["val_data_file"],
                         output_dir=config['val_predict_dir'],
                         center_patch=config['center_patch'],
                         overlap_label=config['overlap_label_predict'],
                         final_val = True)
    mv_results(config['val_predict_dir'],config['val_to_upload'])
    print('Validation dataset prediction finished.')
    return

def predict_training_dataset():
    if not os.path.exists(config['training_index_list']):
        with open(config['training_index_list'],'wb') as f:
            pickle.dump(list(range(config['num_training_subjects'])),f)
    print('Training dataset prediction starts...')        
    run_validation_cases(validation_keys_file=config['training_index_list'],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_dir=config['training_predict_dir'],
                         center_patch=config['center_patch'],
                         overlap_label=config['overlap_label_predict'],
                         final_val = True)
    mv_results(config['training_predict_dir'],config['training_to_upload'])
    print('Training dataset prediction finished.')
    return
    
if __name__ == '__main__':
    main_run()