#!/usr/bin/env python3

from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from tools.data.preprocess.augmentation import augment_data_train_test_split
from tools.data.preprocess.noise import real_noise

# For reproducibility, set the seed for the RNG.
# Used by numpy and scikit-learn, and probably other libraries.
# Will produce the same dataset (including shuffle order) for every run.
np.random.seed(1234567)

beg = timer()

##########################################################################
# Specify noise generation method and parameters.

# Multi-site event parameters
# Same fraction of single- and multi-site events.
n_mse_dict = {1: 1,
              2: 0.25,
              3: 0.25,
              4: 0.25,
              5: 0.25}

mse_args_dict = {'n_mse_dict': n_mse_dict,
                 'shuffle': True,
                 'amp_low': 0.2,
                 'amp_high': 1.0,
                 'hshift_low': -100,
                 'hshift_high': 0}

# Parameters of the noise addition.
hshift_params = {'low': -300, 'high': -50,
                 'abs_shift': True, 'use_ends': True}
vshift_params = {'low': -0.1, 'high': 0.1}
vscale_params = {'low': 0.9, 'high': 1.1}

# Use noise files that are even.
noise_file_dir = './data/noise'
noise_file_list = [Path(noise_file_dir, "noise.npy")]

np_arrays = [np.load(f, mmap_mode='r') for f in noise_file_list]
mask = np.ones(sum(n.shape[0] for n in np_arrays), dtype=bool)

noise_function = real_noise
noise_params = {'noise_arr_list': np_arrays,
                'mask': mask,
                'sigma_low': 0.00,
                'sigma_high': 0.20}

##########################################################################
# Collect all of the arguments.
# Pass to the function to augment the data.
# Processes the original noise arrays.

n_dup = 1
augment_data_params = {'n_dup': n_dup,
                       'noise_function': noise_function,
                       'noise_params': noise_params,
                       'horizontal_shift_params': hshift_params,
                       'vertical_shift_params': vshift_params,
                       'vertical_scale_params': vscale_params,
                       'mse_args_dict': mse_args_dict}

# Directory from where to load the numpy files.
numpy_file_dir = "./data/simulations"
numpy_path = "./data/simulations/library.npy"

# Augment the training data.
augment_data_train_test_split(numpy_filepath_list=(numpy_path,),
                              numpy_file_dir=numpy_file_dir,
                              **augment_data_params)

##########################################################################
# Display the time elapsed and a summary of the outputs.

time_seconds = timer() - beg
m, s = divmod(time_seconds, 60)
h, m = divmod(m, 60)
print(f"Time elapsed: {h:02.0f}:{m:02.0f}:{s:02.0f}")
