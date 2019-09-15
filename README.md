# A 3D U-Net Based Solution to BraTS 2019
## Introduction
This is my solution used for the participation of MICCAI BraTS 2019 competation. For more information about tasks in BraTS 2019 please refer to their [homepage](https://www.med.upenn.edu/cbica/brats2019.html). 

I've only touched the segmentation task(task1) and the survival task(task2).


## Development Environment
Both my desktop and laptop had contributed a lot during the project.

Desktop: 
- gtx1080ti 
- ubuntu16.04 + virtualenv + python==3.5.2 + tensorflow-gpu==1.11.0 + keras==2.2.4

Laptop:
- gtx1060
- ubuntu16.04 + virtualenv + python==3.5.2 + tensorflow-gpu==1.14.0 + keras==2.2.4

### How to setting up the GPU development environment
You could refer to my setting up history [here](https://github.com/woodywff/history-of-setting-up-deep-learning-environment)

Other packages you may want to know:
- h5py==2.9.0
- niblabel==2.4.1
- numpy==1.16.4
- pandas==0.24.2
- tables==3.5.2

## How to run it








There are three tasks in the challenge this year. I've only submitted the first two of them. 


The provided dataset includes four structural MRI modalities. Each MRI image is a 240\*240\*155 sized 3D image saved in a .nii.gz file. The first task is to mark out the brain tumor area in the MRI pictures, which is a segmentation problem. In the training dataset you could find the label image  
