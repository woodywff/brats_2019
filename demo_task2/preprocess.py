from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np
import os
import sys
sys.path.append('..')
from dev_tools.my_tools import print_red
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import pdb
from train_model import config
from dev_tools.my_tools import my_mkdir

def get_truth(sub_id):
    '''
    for training
    '''
    try:
        img = nib.load(os.path.join('../data/preprocessed/HGG',sub_id,'t1.nii.gz')).get_data()
        truth = nib.load(os.path.join('../data/preprocessed/HGG',sub_id,'truth.nii.gz')).get_data()
    except FileNotFoundError:
        img = nib.load(os.path.join('../data/preprocessed/LGG',sub_id,'t1.nii.gz')).get_data()
        truth = nib.load(os.path.join('../data/preprocessed/LGG',sub_id,'truth.nii.gz')).get_data()
    return img,truth

def get_truth_val(sub_id):
    img = nib.load(os.path.join('../data/preprocessed_val_data/val',sub_id,'t1.nii.gz')).get_data()
    truth = nib.load(os.path.join('../demo_task1/saves/val_to_upload',sub_id+'.nii.gz')).get_data()
    return img,truth

def get_length(b):
    return np.sum(np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]))


def create_csv(source_csv,target_csv):
    if not os.path.exists(target_csv):
        df = pd.read_csv(source_csv)
        df = df[df['ResectionStatus'] == 'GTR']
        df = df[df['Survival'] != 'ALIVE (361 days later)']
        df = df.fillna(-1)
        df = df[df['Survival'] != -1]
        df.to_csv(target_csv, index=False, sep=',')
    return

def create_csv_val(source_csv,target_csv):
    if not os.path.exists(target_csv):
        df = pd.read_csv(source_csv)
        df = df[df['ResectionStatus'] == 'GTR']
        df.to_csv(target_csv, index=False, sep=',')
    return

        
def get_xy(csv_file, save_name, gtr_only=True, for_val=True):
#     pdb.set_trace()
    if not for_val:
        if os.path.exists(save_name):
            print(save_name,'exists already.')
            return
    
#     df = pd.read_csv('os_data/gtr_only.csv' if gtr_only else 'os_data/all.csv')
    df = pd.read_csv(csv_file)

    X = []
    Y = []
    for i in tqdm(range(len(df))):
        sub_id = df.iloc[i]['BraTS19ID']
#         print(sub_id)
        if for_val:
            img,truth = get_truth_val(sub_id)
        else:
            img,truth = get_truth(sub_id)
    
        size_brain = np.sum(np.nonzero(img))

        size_et = np.sum(truth==4)
        size_tc = size_et + np.sum(truth==1)
        size_wt = size_tc + np.sum(truth==2)

        prop_et = size_et/size_brain
        prop_tc = size_tc/size_brain
        prop_wt = size_wt/size_brain

        prop_et_tc = size_et/size_tc
        prop_et_wt = size_et/size_wt
        prop_tc_wt = size_tc/size_wt

        bound_et = np.gradient((truth==4).astype(np.float32))
        length_et = get_length(bound_et)
        bound_tc = np.gradient(np.logical_or(truth==1,truth==4).astype(np.float32))
        length_tc = get_length(bound_tc)
        bound_wt = np.gradient(np.logical_or(np.logical_or(truth==1,truth==4),truth==2).astype(np.float32))
        length_wt = get_length(bound_wt)

        age = df.iloc[i]['Age']

        features = [size_brain,size_et,size_tc,size_wt,
                    prop_et,prop_tc,prop_wt,
                    prop_et_tc,prop_et_wt,prop_tc_wt,
                    length_et,length_tc,length_wt,age]

        if not gtr_only:
            status = df.iloc[i]['ResectionStatus']
            if status == '-1':
                features.append(0)
                features.append(0)
            elif status == 'GTR':
                features.append(1)
                features.append(0)
            elif status == 'STR':
                features.append(0)
                features.append(1)
            else:
                pdb.set_trace()
                print_red('wrong ResectionStatus'+sub_id)


        X.append(features)
        if not for_val:
            Y.append(int(df.iloc[i]['Survival']))

    np.savez(save_name,X,Y)
    return

def main():
    my_mkdir('../data/os_data')
    create_csv(config['training_source_csv'],config['training_target_csv'])
    create_csv_val(config['val_source_csv'],config['val_target_csv'])
    
    get_xy(config['training_target_csv'], config['training_npz'],for_val=False)
    get_xy(config['val_target_csv'], config['val_npz'])

    
if __name__ == '__main__':
    main()