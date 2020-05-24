# A 3D U-Net Based Solution to BraTS 2019
## Introduction
This is our solution used for the participation of MICCAI BraTS 2019 competition. We were lucky and won a third place award in the second task. For more information about tasks in BraTS 2019 please refer to their [homepage](https://www.med.upenn.edu/cbica/brats2019.html). 

The details of this project have been written in the pre-proceedings article ["3D U-Net Based Brain Tumor Segmentation and
Survival Days Prediction"](https://arxiv.org/abs/1909.12901)

We've only touched the segmentation task(task1) and the survival task(task2).

The 3D U-Net model is borrowed from [Isensee et.al's paper](https://doi.org/10.1007/978-3-030-11726-9_21) and [ellisdg's repository](https://github.com/ellisdg/3DUnetCNN.git). You could also see this implementation as an extension to ellisdg's work. The solution to task 2 is a development based on [xf4j's work](https://github.com/xf4j/brats18) which won the title in BraTS 2018.

If you wanna download our trained models, there is [an image we uploaded to dockerhub](https://hub.docker.com/r/woodywff/brats2019) inside which you could find the saved .h5 files.

## Development Environment
Both my desktop and laptop had contributed a lot to the project.

Desktop: 
- gtx1080ti 
- ubuntu16.04 + virtualenv + python==3.5.2 + tensorflow-gpu==1.11.0 + keras==2.2.4

Laptop:
- gtx1060
- ubuntu16.04 + virtualenv + python==3.5.2 + tensorflow-gpu==1.14.0 + keras==2.2.4

### How to setting up the GPU development environment
You could refer to my setting up history [here](https://github.com/woodywff/history-of-setting-up-deep-learning-environment)

### Packages
Here are some packages you may need to install.

- For [n4itk bias correction](https://doi.org/10.1109/TMI.2010.2046908) preprocessing, you may need to install `ants`. 
  - Just follow the installation guide on their homepage [here](http://neuro.debian.net/install_pkg.html?p=ants)
  - Add `ants` to your environment variable `PATH`, for instance like `$ export PATH=${PATH}:/usr/lib/ants/`
- 
    ```
    $ sudo apt-get install libhdf5-serial-dev 
    $ pip install tables
    ```
- 
  ```
  $ pip install numpy, nibable, SimpleITK, tqdm, xlrd, pandas, progressbar, matplotlib, nilearn, sklearn
  ```
- For Instance Normalization, you may need to download and install `keras-contrib`.
  ```
  $ git clone https://www.github.com/farizrahman4u/keras-contrib.git
  $ pip install <where you saved it>
  ```
For other missed packages you may come across, just install them as required according to the `ImportError`.



## How to Run It
`original_tree.txt` shows the original organization of this whole project before you start the training process. 

`data/original` saves training dataset. `data/val_data` saves validation or test dataset. `data/survival_data.csv` is the phenotypic information for training subjects. `data/val_data/val/survival_evaluation.csv` is the phenotypic information for validation or test subjects. `data/preprocessed` and `data/preprocessed_val_data` saves the dataset after preprocessing procedure.   

Folder `dev_tools` provides some of my own functions in common use.

Folder `unet3d` encapsulates 3D U-Net related functions that you could invoke in different demos.

`demo_task1` includes brain tumor segmentation task specific codes.

`demo_task2` includes survival days prediction task specific codes.

The self-explained `demo_run.ipynb` in `demo_task1` and `demo_task2` illustrate the basic flow of the program. Since there are two phases of training process with different patching strategies, we need to switch the bool value of `config['pred_specific']` in `demo_task1/train_model.py` to decide which strategy do we need to use for training. 
You may also need to manually delete two temporarily generated file `demo_task1/num_patches_training.npy` and `demo_task1/num_patches_val.npy` once you changed the patching strategy.

This program is valid for both validation dataset and test dataset. Once you changed the validation dataset (to test dataset), please delete the old `data/val_data.h5` and `data/val_index_list.pkl`.

The [n4itk bias correction](https://doi.org/10.1109/TMI.2010.2046908) is time consuming and optional during this project. You could manually turn it off by means of setting the certain argument during the preprocessing process. If you don't use the bias correction then you may not need to install `ants` at the beginning of `demo_task1/demo_run.ipynb`.


Last but not the least, pay attention that we shield something in `.gitignore`.



## Acknowledgement
Again, this work refers to [Isensee et.al's paper](https://doi.org/10.1007/978-3-030-11726-9_21), [ellisdg's repository](https://github.com/ellisdg/3DUnetCNN.git) and [xf4j's repository](https://github.com/xf4j/brats18). Deeply appreciate their contributions to the community.

Many thanks to the host of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019.html).










