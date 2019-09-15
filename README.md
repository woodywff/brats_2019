# A 3D U-Net Based Solution to BraTS 2019
## Introduction
This is my solution used for the participation of MICCAI BraTS 2019 competation. For more information about tasks in BraTS 2019 please refer to their [homepage](https://www.med.upenn.edu/cbica/brats2019.html). 

The details of this project have been written in the preceeding article ["Brain-wise Tumor Segmentation and Patient
Overall Survival Prediction"]()

I've only touched the segmentation task(task1) and the survival task(task2).

The 3D U-Net model is borrowed from [Isensee et.al's paper](https://doi.org/10.1007/978-3-030-11726-9 21) and [ellisdg's repository](https://github.com/ellisdg/3DUnetCNN.git). You could also see this implementation as an extention of ellisdg's work. 

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
original_tree.txt shows the original orgnization of this whole project before you start training process. 

pay attention that 






