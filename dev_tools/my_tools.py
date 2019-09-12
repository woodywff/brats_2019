import xlrd
import numpy as np
import pandas as pd
import os
import re
from progressbar import *
import nibabel as nib
import pdb
import scipy.ndimage
import matplotlib.pyplot as plt
import time
from tensorflow.python import pywrap_tensorflow 
from tensorflow.python.tools import inspect_checkpoint as chkp

def show_ckpt(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename) 
    var_to_shape_map = reader.get_variable_to_shape_map() 
#     print(var_to_shape_map)
    print_sep()
    for key in var_to_shape_map: 
        print("tensor_name: ", key)
        
#     print_sep()
#     chkp.print_tensors_in_checkpoint_file(filename, tensor_name='', all_tensors=True)
    return


def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))

def print2d(npy_img,trivial=False,img_name='',save=False,save_name='./test.jpg'):
    '''
    !!!dataset specific
    plot 2d mri images in Sagittal, Coronal and Axial dimension.
    img: 3d ndarray
    '''
    dim = npy_img.shape
#     print('Dimension: ',npy_img.shape)
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15,5))

    img = npy_img[round(dim[0]/2),:,:]
#     img = npy_img[87,:,:]
    ax1.imshow(np.rot90(img), cmap=plt.cm.gray)
#     ax1.imshow(np.rot90(img))
    if trivial:
        print('value of point(0,0) = ', img[0,0])
    ax1.set_title('Sagittal '+(img_name if trivial else ''),fontsize=15)
#     ax1.imshow(img, cmap=plt.cm.gray)
    ax1.axis('off')
    img = npy_img[:,round(dim[1]/2),:]
#     img = npy_img[:,123,:]
    ax2.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax2.set_title('Coronal '+(str(npy_img.shape) if trivial else ''),fontsize=15)
    ax2.axis('off')
    img = npy_img[:,:,round(dim[2]/2)]
#     img = npy_img[:,:,154]
    ax3.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax3.set_title('Axial',fontsize=15)
#     ax3.imshow(img, cmap=plt.cm.gray)
    ax3.axis('off')
    # plt.subplot(131); plt.imshow(np.rot90(img), cmap=plt.cm.gray)
    # img = npy_img[:,65,:]
    # plt.subplot(132); plt.imshow(img, cmap=plt.cm.gray)
    # img = npy_img[65,:,:]
    # plt.subplot(133); plt.imshow(np.rot90(img,2), cmap=plt.cm.gray)
    if trivial:
        print(np.max(npy_img))
        print(np.min(npy_img))
    
    if save:
        plt.savefig(save_name)
    return

def print2d_origin(npy_img,img_name='',save=False,save_name='./test.jpg'):
    '''
    !!!dataset specific
    plot 2d mri images in Sagittal, Coronal and Axial dimension.
    img: 3d ndarray
    '''
    dim = npy_img.shape
#     print('Dimension: ',npy_img.shape)
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15,5))

    img = npy_img[round(dim[0]/2),:,:]
    ax1.imshow(img, cmap=plt.cm.gray)
    print(img[0,0])
    ax1.axis('off')
    img = npy_img[:,round(dim[1]/2),:]
    ax2.imshow(img, cmap=plt.cm.gray)
    ax2.axis('off')
    img = npy_img[:,:,round(dim[2]/2)]
    ax3.imshow(img, cmap=plt.cm.gray)
    ax3.axis('off')
    
    if save:
        plt.savefig(save_name)
    return

def printimg(filename,size=10):
    f, (ax1) = plt.subplots(1, 1, figsize=(size,size))
    npy = plt.imread(filename)
    ax1.imshow(npy)
    ax1.axis('off')
    return

def print_sep(something='-'):
    print('----------------------------------------',something,'----------------------------------------')
    return

# 3D rotatation
def rot_clockwise(arr,n=1):
    return np.rot90(arr,n,(0,2))
def rot_anticlockwise(arr,n=1):
    return np.rot90(arr,n,(2,0))

def rot_ixi2abide(img_ixi):
    '''
    to rot IXI to the same direction as ABIDE
    '''
    temp = np.rot90(img_ixi,axes=(1,2))
    temp = np.rot90(temp,axes=(1,0))
    return temp

def rot_oasis2abide(img_ixi):
    '''
    to rot OASIS to the same direction as ABIDE
    '''
    temp = np.rot90(img_ixi,axes=(1,2))
    temp = np.rot90(temp,axes=(0,1))
    return temp


def time_now():
    return time.strftime('%Y.%m.%d.%H:%M:%S',time.localtime(time.time()))

def sec2hms(seconds):    
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return str(int(d))+' days, '+str(int(h))+' hours, '+str(int(m))+' mins, '+str(round(s,3))+' secs.'
#     print("%d:%02d:%02d" % (h, m, s))

def my_mkdir(path_name):
    try:
        os.mkdir(path_name)
    except FileExistsError:
        print(path_name,' exists already!')
    return

def my_makedirs(path_name):
    try:
        os.makedirs(path_name)
    except FileExistsError:
        print(path_name,' exists already!')
    return


def get_shuffled(imgs, labels):
    temp = np.array([imgs,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    return image_list,label_list

def minmax_normalize(img_npy):
    '''
    img_npy: ndarray
    '''
    min_value = np.min(img_npy)
    max_value = np.max(img_npy)
    return (img_npy - min_value)/(max_value - min_value)

def z_score_norm(img_npy):
    '''
    img_npy: ndarray
    '''
    return (img_npy - np.mean(img_npy))/np.std(img_npy)


def dist_check(img_npy):
    '''
    have a look at the distribution of the img
    '''
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    ax1.hist(img_npy.reshape(-1))
    ax1.set_title('origin')
    ax2.hist(minmax_normalize(img_npy).reshape(-1))
    ax2.set_title('minmax')
    ax3.hist(z_score_norm(img_npy).reshape(-1))
    ax3.set_title('z score')


def pad_image(img_npy, target_image_shape):
    '''
    image: ndarray
    target_image_shape: tuple or list
    '''
    source_shape = np.asarray(img_npy.shape)
    target_image_shape = np.asarray(target_image_shape)
    edge = (target_image_shape - source_shape)/2
    pad_width = tuple((i,j) for i,j in zip(np.floor(edge).astype(int),np.ceil(edge).astype(int)))
    padded_img = np.pad(img_npy,pad_width,'constant',constant_values=0)
    return padded_img, pad_width

