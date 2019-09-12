import numpy as np
import pdb
from dev_tools.my_tools import print_red

def compute_patch_indices(image_shape, patch_size, overlap, start=None):
#     pdb.set_trace()
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None: # this method gets an even distribution of cubics as I wished
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    patches = get_set_of_patch_indices(start, stop, step)
    # add the center cubic:
    patches = np.vstack((patches, (image_shape - patch_size)//2))
    return patches

def compute_patch_indices_for_prediction(image_shape, patch_size, center_patch=True):
#     pdb.set_trace()
    pdb_set = False
    if pdb_set:
        if np.any(np.array(2*np.array(patch_size) - np.array(image_shape))<=0):
            print_red('error patch: too large')
        if  np.any(np.array(image_shape-patch_size)<=0):
            print_red('error patch: too small')
    start_2 = np.asarray(image_shape - patch_size)
    start_2[start_2 < 0] = 0
    patches = np.array([[0,         0,         0         ],
                        [start_2[0],0,         0         ],
                        [0,         start_2[1],0         ],
                        [0,         0,         start_2[2]],
                        [start_2[0],start_2[1],0         ],
                        [start_2[0],start_2[1],start_2[2]],
                        [start_2[0],0,         start_2[2]],
                        [0,         start_2[1],start_2[2]]])
    if center_patch:
        patches = np.vstack((patches, (image_shape - patch_size)//2))
    return patches


def get_set_of_patch_indices(start, stop, step):
#     pdb.set_trace()
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)


def get_random_nd_index(index_max):
    return tuple([np.random.choice(index_max[index] + 1) for index in range(len(index_max))])


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
#     data = np.pad(data, pad_args, mode="edge")
    data = np.pad(data, pad_args, 'constant',constant_values=0)
    patch_index += pad_before
    return data, patch_index


def reconstruct_from_patches(patches, patch_indices, data_shape, default_value=0):
    """
    Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
    patches are averaged.
    :param patches: List of numpy array patches.
    :param patch_indices: List of indices that corresponds to the list of patches.
    :param data_shape: Shape of the array from which the patches were extracted.
    :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
    be overwritten.
    :return: numpy array containing the data reconstructed by the patches.
    """
#     pdb.set_trace()
    data = np.ones(data_shape) * default_value
    image_shape = data_shape[-3:]
    count = np.zeros(data_shape, dtype=np.int)
    image_patch_shape = patches[0].shape[-3:]
    for patch, index in zip(patches, patch_indices):
        if np.any(index < 0):
            fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
            patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
            index[index < 0] = 0
        if np.any((index + image_patch_shape) >= image_shape):
            fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                        * ((index + image_patch_shape) - image_shape)), dtype=np.int)
            patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
        patch_index = np.zeros(data_shape, dtype=np.bool)
        patch_index[...,
                    index[0]:index[0]+patch.shape[-3],
                    index[1]:index[1]+patch.shape[-2],
                    index[2]:index[2]+patch.shape[-1]] = True
        patch_data = np.zeros(data_shape)
        patch_data[patch_index] = patch.flatten()

        new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
        data[new_data_index] = patch_data[new_data_index]

        averaged_data_index = np.logical_and(patch_index, count > 0)
        if np.any(averaged_data_index):
            data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
        count[patch_index] += 1
    return data

