from pathlib import Path
import warnings

import numpy as np

################################################################################

def load_pulses_by_index(np_arrays, indexes, nsamples=None):
    '''
    Load and process certain entries of a list of numpy arrays into one array.
    '''
    # Counters to keep track of last index of the previous arrays.
    s = 0
    n = 0

    indexes_list = []
    for arr in np_arrays:
        # Select only the indexes for the given file.
        mask = np.logical_and((indexes >= s), (indexes < s + arr.shape[0]))
        indexes_arr = indexes[mask] - s
        indexes_arr.sort()
        indexes_list.append(indexes_arr)

        # Increment the counters.
        s += arr.shape[0]
        n += indexes_arr.shape[0]

    arr_new_tuple = tuple(arr[iarr, :nsamples] for arr, iarr in zip(np_arrays, indexes_list))
    arr_new = np.concatenate(arr_new_tuple, axis=0)

    return arr_new

################################################################################

def gaussian_noise(pulse_arr, sigma_low, sigma_high=None):
    '''
    Add normally distributed noise with random variance to a pulse.
    Variance is based off of input parameters.
    '''
    # Generate the standard deviation from a random uniform distribution.
    if sigma_high is None:
        sigma = sigma_low
    else:
        sigma = np.random.uniform(low=sigma_low, high=sigma_high,
                                  size=(pulse_arr.shape[0], 1))

    # Generate the array of noise pulses and add them to the original array.
    noise_arr = np.random.normal(0, sigma, size=pulse_arr.shape)
    pulse_arr += noise_arr

def real_noise(pulse_arr, noise_arr_list, mask=None, **kwargs):
    '''
    Add real noise to an array sampled from a list of numpy array(s).
    Should be memory-mapped if noise arrays are large.
    '''
    # Check if list of files or list of numpy arrays already.
    if all(isinstance(f, (str, Path)) for f in noise_arr_list):
        noise_arr_list = [np.load(f, mmap_mode='r') for f in noise_arr_list]
    elif all(isinstance(arr, np.ndarray) for arr in noise_arr_list):
        # Flush the memory-mapped arrays.
        for arr in noise_arr_list:
            arr.flush()
    else:
        raise ValueError("noise_arr_list must be either a "
                         "list of numpy arrays or a "
                         "list of file paths to numpy arrays.")

    # Get the number of entries in the noise files and in the array.
    noise_entries = sum(arr.shape[0] for arr in noise_arr_list)
    pulse_entries = pulse_arr.shape[0]

    indexes_pulse = draw_random_indexes(samples=pulse_entries,
                                        n=noise_entries,
                                        mask=mask)

    # Load in the selected pulses by entry.
    noise_arr = load_pulses_by_index(noise_arr_list,
                                     indexes=indexes_pulse,
                                     nsamples=pulse_arr.shape[-1])

    # Rescale and add the noise.
    rescale_noise(noise_arr, **kwargs)
    pulse_arr += noise_arr

################################################################################

def rescale_noise(noise_arr, sigma_low, sigma_high=None):
    '''
    Given an array, rescale it to a random standard deviation.
    Can either generate sigma from a uniform or normal distribution.
    Should modify the array in place!
    '''
    # Normalize to mean zero and unit variance.
    noise_arr -= noise_arr.mean(axis=1, keepdims=True)
    noise_arr /= noise_arr.std(axis=1, keepdims=True)

    # Generate the standard deviation from a random uniform distribution.
    if sigma_high is None:
        sigma = sigma_low
    else:
        sigma = np.random.uniform(low=sigma_low, high=sigma_high,
                                  size=(noise_arr.shape[0], 1))

    noise_arr *= sigma

def draw_random_indexes(samples, n, mask=None):
    '''
    Function to draw random indexes between zero and some number, n.
    A mask can be supplied to prevent resampling the same entries.
    This function will update the mask based on the indexes selected.
    '''
    # Check if mask is the same length as expected.
    # Raise error otherwise.
    if mask is not None and mask.shape[0] != n:
        raise ValueError("Shape of mask is not the same "
                         "as the number of noise entries.")

    # Check if the noise files have been exhausted.
    # Reset all entries to True to reuse noise files.
    if mask is not None and samples > np.count_nonzero(mask):
        mask[:] = True
        warnings.warn("Mask has been reset. Resusing pulses. "
                      "This is may not be your intention!")

    # Choose random indexes for the array of pulses in question.
    # Set the entries drawn to False so that they are not drawn again.
    indexes_all = np.arange(0, n)
    if mask is not None:
        indexes = np.random.choice(indexes_all[mask], replace=False, size=samples)
        mask[indexes] = False
    else:
        indexes = np.random.choice(indexes_all, replace=False, size=samples)

    return indexes
