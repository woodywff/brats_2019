import os

# from train import config
from train_model import config
from unet3d.prediction import run_validation_cases
import pdb

def main():
#     pdb.set_trace()
    prediction_dir = os.path.abspath("prediction")
    print('Start predicting...')
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
#                          overlap=32,
                         overlap=0,  # this param doesn't work anymore when using prediction specific patching strategy
                         hdf5_file=config["data_file"],
                         output_dir=prediction_dir,
                         center_patch=config['center_patch'],
                         overlap_label=config['overlap_label_predict'])


if __name__ == "__main__":
    main()
