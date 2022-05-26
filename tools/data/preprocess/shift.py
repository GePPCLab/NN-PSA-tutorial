import numpy as np

def hshift(pulse_arr, shift, use_ends=True, copy=False):
    '''
    Shift an array horizontally.
    Negative shift is to the right.
    Positive shift is to the left.

    If use_ends is true, it will set the values that are
    "cut off" to the last value in the array.
    '''
    if copy:
        pulse_arr = pulse_arr.copy()

    if shift > 0:
        # Shifts the array to the left.
        pulse_arr[..., 0:-shift] = pulse_arr[..., shift:]
        # Ensures that the beginning/end are all the same value.
        if use_ends:
            pulse_arr[..., -shift:] = pulse_arr[..., -1:]

    elif shift < 0:
        # Shifts the array to the right.
        pulse_arr[..., -shift:] = pulse_arr[..., 0:shift]
        # Ensures that the beginning/end are all the same value.
        if use_ends:
            pulse_arr[..., 0:-shift] = pulse_arr[..., 0:1]

    if copy:
        return pulse_arr

def add_hshift(pulse_arr, low, high, abs_shift, use_ends=True, copy=False):
    '''
    Shift an array of pulses horizontally.
    Each shift is drawn from a random uniform distribution.

    The abs_shift argument reshifts pulses to zero before applying the shift,
    ensuring that shifts are absolute (not relative) to the existing shift.
    '''
    # If desired, can have a constant shift.
    if high is None:
        high = low + 1

    # Check that maximum value (first instance) is the same as
    # last value in array (i.e., flat top).
    ind_rows = np.arange(pulse_arr.shape[0])
    ind_cols = np.argmax(pulse_arr, axis=1)
    if not np.allclose(pulse_arr[ind_rows, ind_cols], pulse_arr[:, -1]):
        raise ValueError("Attempting to horizontally shift array "
                         "that may have noise or exponential decay. "
                         "This is not supported.")

    # Get the current shift (if any),
    # which is needed to perform an absolute shift.
    if abs_shift:
        rise_start = (pulse_arr.shape[1] - np.argmin(pulse_arr[:, ::-1], axis=1) - 1)
    else:
        rise_start = 0

    # Generate an array of random horizontal shifts.
    horizontal_shift = np.random.randint(low=low, high=high,
                                         size=pulse_arr.shape[0])

    # Compute the total shift, first to the left (positive) to undo the shift,
    # then to the right (negative) to get to the new shift point.
    total_shift = rise_start + horizontal_shift

    # Apply random horizontal shift to each pulse.
    for i, hshift_i in enumerate(total_shift):
        hshift(pulse_arr[i], hshift_i, use_ends=use_ends, copy=copy)

def add_vshift(pulse_arr, low, high=None):
    '''
    Shift an array of pulses vertically.
    Each shift is drawn from a random uniform distribution.
    '''
    # If desired, can have a constant shift.
    if high is None:
        vertical_shift = low
    else:
        vertical_shift = np.random.uniform(low=low, high=high, size=(pulse_arr.shape[0], 1))

    pulse_arr += vertical_shift

def add_vscale(pulse_arr, low, high=None):
    '''
    Scale an array of pulses by amplitude.
    Each scale is drawn from a random uniform distribution.
    '''
    # If desired, can have a constant scale.
    if high is None:
        vertical_scale = low
    else:
        vertical_scale = np.random.uniform(low=low, high=high, size=(pulse_arr.shape[0], 1))

    pulse_arr *= vertical_scale
