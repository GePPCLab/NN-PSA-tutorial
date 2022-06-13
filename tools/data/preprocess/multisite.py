import numpy as np
from sklearn.utils import shuffle as shuffle_arrays

from .shift import add_hshift

def generate_single_multisite_event(pulse_arr_sse, n,
                                    hshift_low=-100,
                                    hshift_high=0,
                                    amp_low=0.0,
                                    amp_high=1.0):
    '''
    Function to generate a single multi-site event
    given a set of single-site event pulses as a numpy array.
    Will grab n random pulses, shift them randomly, and add them together.
    Also scales them by a random factor (such that the sum is unity).
    '''
    if n < 1:
        raise ValueError("Number of sites must be a positive integer, "
                         f"not {n}.")

    # Number of events in the array of pulses.
    n_ev = pulse_arr_sse.shape[0]

    # Generate multiple indexes based on the number of sites.
    # Probably no need to check if any of the indexes are the same, since it is fairly unlikely.
    indexes = np.random.randint(n_ev, size=n)

    # This makes a copy of the data since it is not a slice.
    # See numpy advanced indexing.
    pulses = pulse_arr_sse[indexes, :]

    if n == 1:
        # No need for shifting or scaling if a single-site event is requested.
        return pulses.squeeze(axis=0)

    elif n > 1:
        # Generate random scaling factors (must sum to unity).
        scale = np.random.uniform(amp_low, amp_high, size=(n, 1))
        scale *= 1 / scale.sum()
        pulses *= scale

    # Randomly shift all of the pulses.
    add_hshift(pulses, low=hshift_low, high=hshift_high, abs_shift=True, use_ends=True)

    # Finally, sum all of the pulses together.
    pulse = np.sum(pulses, axis=0)

    return pulse

def generate_multisite_events(pulse_arr_sse, n_mse_dict,
                              n_ev_sse=None, shuffle=True, **kwargs):
    '''
    Function to generate a given number of multi-site events from a set of pulses.
    n_mse_dict should be a dictionary where the key(s) are the number of sites
    and the value(s) are the number of events to generate for that site,
    or the fraction of events relative to the number of events in the given pulse array.

    For example:
      {1: 200, 2: 1000, 3: 500, 4: 1200}
      {1: 0.5, 2: 0.5, 3: 0.75, 4: 0.25}
    '''
    # Initialize the array.
    if n_ev_sse is None:
        n_ev_sse = pulse_arr_sse.shape[0]

    samples = pulse_arr_sse.shape[1]

    n_ev_all = sum(int(n_ev * (n_ev_sse if n_ev <= 1 else 1)) for n_ev in n_mse_dict.values())

    # Initialize arrays for storing pulses and corresponding number of sites.
    pulse_arr_mse = np.zeros((n_ev_all, samples), dtype=pulse_arr_sse.dtype)
    nsite_arr = np.zeros((n_ev_all, 1), dtype=int)

    # Fill the arrays with multi-site events.
    n_start = 0
    for n_sites, n_ev in n_mse_dict.items():
        # Number of events for this particular number of sites.
        n_ev_abs = int(n_ev * (n_ev_sse if n_ev <= 1 else 1))

        # Generate random multi-site events for particular number of sites.
        for ev in range(n_ev_abs):
            pulse = generate_single_multisite_event(pulse_arr_sse, n=n_sites, **kwargs)
            pulse_arr_mse[n_start + ev] = pulse

        nsite_arr[n_start:n_start + n_ev_abs] = n_sites

        n_start += n_ev_abs

    # Shuffle the arrays at the end (along axis 0, the rows/entries).
    if shuffle:
        pulse_arr_mse, nsite_arr = shuffle_arrays(pulse_arr_mse, nsite_arr)

    return pulse_arr_mse, nsite_arr
