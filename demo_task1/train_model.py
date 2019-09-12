import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model
import pdb
import time
from dev_tools.my_tools import sec2hms

config = dict()
config["overwrite"] = False              # To overwrite data.h5.
# config["pool_size"] = (2, 2, 2)          # pool size for the max pooling operations

config["image_shape"] = (240,240,155)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (128, 128, 128)     # switch to None to train on the whole image
config["training_patch_start_offset"] = (4, 4, 4)  # randomly offset the first patch index by up to this offset
config["validation_patch_overlap"] = 32                # if > 0, during training, validation patches will be overlapping

# config['pred_specific'] = True           # To train with patching strategy specificly for prediction. 
config['pred_specific'] = False

config['center_patch'] = True            # To include the center patch in the patching strategy.

config["batch_size"] = 1
config["validation_batch_size"] = 2
config["n_epochs"] = 300
config["data_file"] = os.path.abspath("../data/data.h5")
# config["data_file"] = os.path.abspath("../data/test_trashcan/data.h5")
config["model_file"] = os.path.abspath("seg_model.h5")
config['mean_std_file'] = os.path.abspath('../data/mean_std.pkl')
config['val_data_file'] = os.path.abspath("../data/val_data.h5")

config['val_predict_dir'] = os.path.abspath("val_prediction")
config['val_index_list'] = os.path.abspath('../data/val_index_list.pkl')

config['num_val_subjects'] = 125
# config['num_val_subjects'] = 166  # manually set the number of validation subjects! 
                                    # You need to refresh config['val_index_list'] once you changed config['num_val_subjects']
config['val_to_upload'] = os.path.abspath('saves/val_to_upload')

config['training_predict_dir'] = os.path.abspath("training_prediction")
config['training_index_list'] = os.path.abspath('../data/training_index_list.pkl')
config['num_training_subjects'] = 335  # manually set the number of training subjects (for final uploading)
                                         # You need to refresh config['training_index_list'] once you changed 
                                         # config['num_training_subjects']
config['training_to_upload'] = os.path.abspath('saves/training_to_upload')


#------------------- 5-fold cross validation -----------------------------------
# config["training_file"] = os.path.abspath("../data/list_training_ids.pkl")
# config["validation_file"] = os.path.abspath("../data/list_validation_ids.pkl")

config["training_file"] = os.path.abspath("../data/list_cv1_train.pkl")
config["validation_file"] = os.path.abspath("../data/list_cv1_val.pkl")

# config["training_file"] = os.path.abspath("../data/list_cv2_train.pkl")
# config["validation_file"] = os.path.abspath("../data/list_cv2_val.pkl")

#config["training_file"] = os.path.abspath("../data/list_cv3_train.pkl")
#config["validation_file"] = os.path.abspath("../data/list_cv3_val.pkl")

# config["training_file"] = os.path.abspath("../data/list_cv4_train.pkl")
# config["validation_file"] = os.path.abspath("../data/list_cv4_val.pkl")

config['for_final_val'] = True
#--------------------------------------------------------------------------------
config['logging_file'] = os.path.abspath('training.log')

# truth.shape = (240,240,155) with value in [1,2,4], if 4 is on top of others or surrounded by others 
# config['overlap_label_generator'] = False
# config['overlap_label_predict'] = False
config['overlap_label_generator'] = True
config['overlap_label_predict'] = True


config["labels"] = (1, 2, 4)             # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["n_base_filters"] = 16

if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True           # if False, will use upsampling instead of deconvolution


config["patience"] = 10    # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8    # portion of the data that will be used for training

# config["flip"] = False              # augments the data by randomly flipping an axis during
config["flip"] = True
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
# config["distort"] = None  # switch to None if you want no distortion
config["distort"] = 0.25
config["augment"] = config["flip"] or config["distort"]

config["skip_blank"] = True                           # if True, then patches without any target will be skipped




def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join("../data", "preprocessed", "*", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    pdb.set_trace()
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, 
                            config["data_file"], 
                            image_shape=config["image_shape"], 
                            modality_names = config['all_modalities'],
                            subject_ids=subject_ids,
                           mean_std_file = config['mean_std_file'])
#     return
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])

    # get training and testing generators
#     pdb.set_trace()
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"],
        pred_specific=config['pred_specific'],
        overlap_label=config['overlap_label_generator'],
        for_final_val=config['for_final_val'])

    # run training
#     pdb.set_trace()
    time_0 = time.time()
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                logging_file = config['logging_file'])
    print('Training time:', sec2hms(time.time() - time_0))
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
