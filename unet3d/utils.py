import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

