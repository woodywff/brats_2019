from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np
import os
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import pdb
from keras.layers import Flatten,Dense
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.models import load_model
import keras.backend as K
from dev_tools.my_tools import my_makedirs

config = dict()
config['initilizer'] = True
config['l2'] = True
config['model_file'] = 'os_model.h5'
config['feature_indices'] = [4,5,6,10,11,12,13]

config['training_target_csv'] = os.path.abspath('../data/os_data/training_gtr_only.csv')
config['training_source_csv'] = os.path.abspath('../data/survival_data.csv')
config['training_npz'] = os.path.abspath('../data/os_data/training_mine_gtr_only.npz')

config['val_target_csv'] = os.path.abspath('../data/os_data/val_gtr_only.csv')
config['val_source_csv'] = os.path.abspath('../data/val_data/val/survival_evaluation.csv')
config['val_npz'] = os.path.abspath('../data/os_data/val_mine_gtr_only.npz')


def lms_acc(y_true, y_pred):
    '''
    long-mid-short survivor accuracy
    '''
#     pdb.set_trace()
    y_pred = K.reshape(y_pred,(-1,))
    y_true = K.reshape(y_true,(-1,))
    long_survivors = K.sum(K.cast(y_pred > 15*30,'int32') * K.cast(y_true > 15*30, 'int32'))
    short_survivors = K.sum(K.cast(y_pred < 10*30,'int32') * K.cast(y_true < 10*30, 'int32'))
    mid_survivors = K.sum(K.cast(y_pred < 15*30,'int32') * K.cast(y_pred > 10*30,'int32')
                        * K.cast(y_true < 15*30,'int32') * K.cast(y_true > 10*30,'int32')) 
    return (long_survivors + short_survivors + mid_survivors)/K.shape(y_true)[0]

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae','mse','acc'])
#     model.compile(optimizer=RMSprop(lr=0.001), loss='mse', metrics=['mae', 'acc'])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mae', 'mse','acc',lms_acc])
    return model
    


def main_run(npz_file='os_data/mine_gtr_only.npz',feature_index=[4,5,6,10,11,12,13]):
    xy_zip = np.load(npz_file)
    train_data = xy_zip['arr_0']
    train_targets = xy_zip['arr_1']
    
    if feature_index:
        train_data = train_data[:,feature_index]
    
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    k = 5
    num_val_samples = len(train_data) // k
    num_epochs = 800
    all_mae_histories = []
    all_train_mae = []
    all_val_loss = []
    all_train_loss = []
    all_train_lms = []
    all_val_lms = []
    
    for i in range(k):
        print('processing fold #', i)
        # Prepare the validation data: data from partition # k
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        # Prepare the training data: data from all other partitions
        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                            train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                                train_targets[(i + 1) * num_val_samples:]],
                                                axis=0)

        model = build_model((train_data.shape[1],))
    
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=1e-8)
        history = model.fit(partial_train_data, 
                            partial_train_targets,
                            validation_data=(val_data,val_targets),
                            epochs=num_epochs, 
                            batch_size=5, 
                            verbose=0,
                            callbacks=[reduce_lr]
                            )
        print(model.evaluate(val_data,val_targets))
#         pdb.set_trace()
        mae_history = history.history['val_mean_absolute_error']
        val_loss = history.history['val_loss']
        train_loss = history.history['loss']
        train_mae = history.history['mean_absolute_error']
        val_lms = history.history['val_lms_acc']
        train_lms = history.history['lms_acc']

        all_mae_histories.append(mae_history)
        all_train_mae.append(train_mae)
        all_val_loss.append(val_loss)
        all_train_loss.append(train_loss)
        all_train_lms.append(train_lms)
        all_val_lms.append(val_lms)
        
        
        plt.figure()
        plt.plot(range(1, len(mae_history) + 1), mae_history,label='val_mae')
        plt.plot(range(1, len(mae_history) + 1), train_mae,label='train_mae')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.show()

        plt.figure()
        plt.plot(range(1, len(mae_history) + 1), val_loss,label='val_loss')
        plt.plot(range(1, len(mae_history) + 1), train_loss,label='train_loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mse')
        plt.show()
        
        plt.figure()
        plt.plot(range(1, len(mae_history) + 1), val_lms,label='val_lms')
        plt.plot(range(1, len(mae_history) + 1), train_lms,label='train_lms')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('lms acc')
        plt.show()
        
    average_mae_history = np.mean(all_mae_histories,axis=0)
    average_train_mae = np.mean(all_train_mae,axis=0)
    average_val_loss = np.mean(all_val_loss,axis=0)
    average_train_loss = np.mean(all_train_loss,axis=0)
    average_train_lms = np.mean(all_train_lms,axis=0)
    average_val_lms = np.mean(all_val_lms,axis=0)
    
    plt.figure()
    plt.plot(range(1, len(mae_history) + 1), average_mae_history,label='val_mae')
    plt.plot(range(1, len(mae_history) + 1), average_train_mae,label='train_mae')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.show()


    plt.figure()
    plt.plot(range(1, len(mae_history) + 1), average_val_loss,label='val_loss')
    plt.plot(range(1, len(mae_history) + 1), average_train_loss,label='train_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mse')
    plt.show()
    
    plt.figure()
    plt.plot(range(1, len(mae_history) + 1), average_val_lms,label='val_lms')
    plt.plot(range(1, len(mae_history) + 1), average_train_lms,label='train_lms')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('lms acc')
    plt.show()
    return model



def final_train(epochs=800, npz_file='../data/os_data/training_mine_gtr_only.npz',feature_index=[4,5,6,10,11,12,13]):
    xy_zip = np.load(npz_file)
    train_data = xy_zip['arr_0']
    train_targets = xy_zip['arr_1']
    
    if feature_index:
        train_data = train_data[:,feature_index]
        
    val_data = train_data[25: 50]
    val_targets = train_targets[25:50]
    
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    
    model = build_model((train_data.shape[1],))
    
    callbacks_list=[]
    callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=1e-8))
    callbacks_list.append(ModelCheckpoint(config['model_file'], save_best_only=True))
    callbacks_list.append(EarlyStopping(patience=50))
    history = model.fit(train_data, 
                        train_targets,
                        validation_data=(val_data,val_targets),
                        epochs=epochs, 
                        batch_size=5, 
                        verbose=0,
                        callbacks=callbacks_list
                        )

    print(model.evaluate(val_data,val_targets))
    print('predicted values:', model.predict(val_data).reshape(-1,))
    print('targeted values:', val_targets)
#     pdb.set_trace()
    
    val_lms = history.history['val_lms_acc']
    train_lms = history.history['lms_acc']
    mae_history = history.history['val_mean_absolute_error']
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    train_mae = history.history['mean_absolute_error']

    plt.figure()
    plt.plot(range(1, len(mae_history) + 1), mae_history,label='val_mae')
    plt.plot(range(1, len(mae_history) + 1), train_mae,label='train_mae')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.show()

    plt.figure()
    plt.plot(range(1, len(mae_history) + 1), val_loss,label='val_loss')
    plt.plot(range(1, len(mae_history) + 1), train_loss,label='train_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('mse')
    plt.show()
    
    plt.figure()
    plt.plot(range(1, len(mae_history) + 1), val_lms,label='val_lms')
    plt.plot(range(1, len(mae_history) + 1), train_lms,label='train_lms')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('lms acc')
    plt.show()
    return model

def evaluate(npz_file='../data/os_data/training_mine_gtr_only.npz',feature_index=[4,5,6,10,11,12,13]):
#     pdb.set_trace()
    xy_zip = np.load(npz_file)
    train_data = xy_zip['arr_0']
    train_targets = xy_zip['arr_1']
    if feature_index:
        train_data = train_data[:,feature_index]
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    val_data = train_data[25: 50]
    val_targets = train_targets[25:50]
    
    model = load_model(config['model_file'],custom_objects={'lms_acc':lms_acc})
    predicted = np.reshape(model.predict(val_data),(-1,))
    
    print('predicted survival days:',predicted)
    print('val_targets:',val_targets)
    print(model.evaluate(val_data,val_targets))
    return predicted, val_targets

def predict():
#     pdb.set_trace()
    xy_zip = np.load(config['training_npz'])
    train_data = xy_zip['arr_0']
    train_data = train_data[:,config['feature_indices']]
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    
    xy_zip = np.load(config['val_npz'])
    val_data = xy_zip['arr_0']
    val_data = val_data[:,config['feature_indices']]
    val_data -= mean
    val_data /= std
    
    model = load_model(config['model_file'],custom_objects={'lms_acc':lms_acc})
    
    train_predicted = model.predict(train_data).reshape(-1,)
    val_predicted = model.predict(val_data).reshape(-1,)
    
#     print(train_predicted)
#     print(val_predicted)
    train_predicted = np.round(train_predicted).astype(int)
    val_predicted = np.round(val_predicted).astype(int)
    
    my_makedirs('saves')
    
    df = pd.read_csv(config['training_target_csv'])
    id_list = list(df['BraTS19ID'])
    os_list = list(train_predicted)
    data_to_save = pd.DataFrame({'id':id_list,'os':os_list})
    data_to_save.to_csv(os.path.join('saves','training_to_upload.csv'), index=False, sep=',',header=False)
    
    df = pd.read_csv(config['val_target_csv'])
    id_list = list(df['BraTS19ID'])
    os_list = list(val_predicted)
    data_to_save = pd.DataFrame({'id':id_list,'os':os_list})
    data_to_save.to_csv(os.path.join('saves','val_to_upload.csv'), index=False, sep=',',header=False)
    
    
    return

if __name__ == '__main__':
    main_run()
    final_train()
    evaluate()
