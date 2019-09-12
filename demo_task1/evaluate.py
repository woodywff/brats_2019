import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
from dev_tools.my_tools import print_red
from tqdm import tqdm



def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
#     pdb.set_trace()
    if np.sum(truth) + np.sum(prediction) == 0:
        return 1
    else:
        return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def main(masked=True):
#     pdb.set_trace()
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()
    for case_folder in tqdm(glob.glob("prediction/*")):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        if masked:
            prediction_file = os.path.join(case_folder, case_folder.split('/')[-1]+'.nii.gz')
            prediction_image = nib.load(prediction_file)
        else:
            try:
                prediction_file = os.path.join(case_folder, "prediction_before_mask.nii.gz")
                prediction_image = nib.load(prediction_file)
            except FileNotFoundError:
                prediction_file = os.path.join(case_folder, case_folder.split('/')[-1]+'.nii.gz')
                prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
    mean_values = np.mean(np.asarray(rows),axis=0)
#     pdb.set_trace()
    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("./prediction/brats_scores.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]
    plt.title('Mean value: WT:{}, TC:{}, ET:{}'.format(round(mean_values[1],4),\
                                              round(mean_values[2],4),round(mean_values[0],4)))
    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')

def evaluate_single_sub(sub_id):
    truth = nib.load(os.path.joint('../data/preprocessed/HGG',sub_id,'truth.nii.gz')).get_data()
    prediction = nib.load(os.path.joint('prediction',sub_id,sub_id+'.nii.gz')).get_data()

    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    return [dice_coefficient(func(truth), func(prediction))for func in masking_functions]
        

if __name__ == "__main__":
    main()
