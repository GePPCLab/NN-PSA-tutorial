import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

def get_data_numpy(filename_X, filename_y):
    '''
    Get inputs and targets for a given numpy array.
    '''
    X = np.load(filename_X)
    y = np.load(filename_y)

    return X, y

def get_data_wrapper_numpy(filename_X, filename_y, type_X, type_y):
    '''
    Loads in numpy arrays via numpy_function to represent as tensors.
    Returns tf.data.Dataset
    '''
    X, y = tf.numpy_function(get_data_numpy,
                             [filename_X, filename_y],
                             (type_X, type_y))

    # Since it is known that the rank is 2, this is good enough...
    X.set_shape(tf.TensorShape([None, None]))
    y.set_shape(tf.TensorShape([None, None]))

    return tf.data.Dataset.from_tensor_slices((X, y))

def one_hot_encode(X, y):
    '''
    Convert the data to single-site and multi-site binary values.
    One hot encode the data.
    '''
    # Convert to single-site/multi-site (0/1 respectively).
    y = tf.cast(y > 1, dtype=y.dtype)

    # One hot encode the values.
    # It adds an extra dimension, so remove it.
    y = tf.one_hot(y, depth=2, axis=-1)
    y = tf.squeeze(y, axis=-2)

    return X, y

def create_dataset(list_X, list_y,
                   type_X, type_y,
                   batch_size=128,
                   shuffle_files=True,
                   shuffle_buffer=1000,
                   prefetch=150,
                   binarize_sse_mse=True):
    '''
    Create the TensorFlow Dataset, sometimes using the above functions.
    '''
    ds = Dataset.from_tensor_slices((list_X, list_y))

    # Shuffles the files that are being loaded in.
    if shuffle_files:
        ds = ds.shuffle(len(list_X), reshuffle_each_iteration=True)

    # Load in the data from numpy arrays.
    ds = ds.flat_map(lambda X, y: get_data_wrapper_numpy(X, y, type_X=type_X, type_y=type_y))

    # Convert target to binary classification (single-site/multi-site)
    # and one hot encode the data.
    if binarize_sse_mse:
        ds = ds.map(one_hot_encode)

    # Create a buffer to load individual events into for shuffling.
    if shuffle_buffer is not None:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    # Return the data in batches if the given batch size.
    ds = ds.batch(batch_size)

    # Prefetch the given number of batches ahead of time
    # to ensure GPU is always being used.
    ds = ds.prefetch(prefetch)

    return ds
