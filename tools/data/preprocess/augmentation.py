from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .multisite import generate_multisite_events
from .shift import add_hshift, add_vshift, add_vscale

def augment_data_train_test_split(numpy_filepath_list,
                                  numpy_file_dir,
                                  n_dup=1,
                                  val_size=0.1,
                                  test_size=0.1,
                                  **kwargs):
    '''
    Function to artificially augment dataset given in a list of numpy files.
    Should be an array of single-site events.
    '''
    # Loop through all files in the file list.
    for npy_file in numpy_filepath_list:
        # Load the numpy file in.
        pulse_arr = np.load(npy_file)
        print(f"Augmenting data in file: {npy_file}")

        # Get the basename without the extension for naming.
        npy_file_stem = str(Path(npy_file).stem)

        # Split the dataset before data augmentation (if desired).
        if test_size is not None:
            pulse_arr_train, pulse_arr_test = train_test_split(pulse_arr, test_size=test_size)
        if val_size is not None:
            pulse_arr_train, pulse_arr_val = train_test_split(pulse_arr_train, test_size=val_size)

        # If both are None, all data becomes the "training" set.
        if test_size is None and val_size is None:
            pulse_arr_train = pulse_arr

        # Create data with artificial augmentation.
        for i in range(n_dup):
            # Construct names for augmented files.
            if n_dup == 1:
                f_base = f"{{{{}}}}/{{}}_{npy_file_stem}.npy"
            else:
                f_base = f"{{{{}}}}/{{}}_{npy_file_stem}_r{i}.npy"

            output_dict = {}
            output_dict["noise"] = str(Path(numpy_file_dir, f_base.format("X_noisy")))
            output_dict["clean"] = str(Path(numpy_file_dir, f_base.format("y_clean")))
            output_dict["nsite"] = str(Path(numpy_file_dir, f_base.format("y_nsite")))

            output_path_dict_train = {key: val.format("train")
                                      for key, val in output_dict.items()}

            augment_data(pulse_arr_train,
                         output_path_dict=output_path_dict_train,
                         **kwargs)

            if val_size is not None:
                output_path_dict_val = {key: val.format("val")
                                        for key, val in output_dict.items()}

                augment_data(pulse_arr_val,
                             output_path_dict=output_path_dict_val,
                             **kwargs)

            if test_size is not None:
                output_path_dict_test = {key: val.format("test")
                                         for key, val in output_dict.items()}

                augment_data(pulse_arr_test,
                             output_path_dict=output_path_dict_test,
                             **kwargs)

def augment_data(pulse_arr, noise_function, noise_params,
                 mse_args_dict=None,
                 output_path_dict=None,
                 horizontal_shift_params=None,
                 vertical_shift_params=None,
                 vertical_scale_params=None):
    '''
    Augment a given set of data by generating
    multi-site events and adding noise to them.

    Randomly adds noise, a vertical shift,
    and horizontal shift to each pulse.

    Returns clean version (only shifted and moved)
    and noisy version (same shifts).
    If output_path_dict is given, it will save
    the generated arrays (must contain the correct keys).
    '''
    # Copy clean array to add noise to.
    pulse_arr_clean = pulse_arr.copy()

    # Multi-site events.
    if mse_args_dict is not None:
        pulse_arr_clean, n_sites = generate_multisite_events(pulse_arr_clean,
                                                             **mse_args_dict)

    # Add a random vertical scale to each event.
    # The order (vscale, then hshift and vshift) is important here!
    if vertical_scale_params is not None:
        add_vscale(pulse_arr_clean, **vertical_scale_params)

    # Add a random horizontal shift to each event.
    if horizontal_shift_params is not None:
        add_hshift(pulse_arr_clean, **horizontal_shift_params)

    # Add a random vertical shift to each event.
    if vertical_shift_params is not None:
        add_vshift(pulse_arr_clean, **vertical_shift_params)

    # Add noise to the duplicated array as defined by the noise generation method.
    # Noise should only be added to the noise array!
    # Make a copy for the noisy targets.
    pulse_arr_noisy = np.copy(pulse_arr_clean)
    noise_function(pulse_arr_noisy, **noise_params)

    # Save (if desired) and return the files.
    if output_path_dict is not None:
        if any(Path(p_npy).is_file() for p_npy in output_path_dict.values()):
            file_str = "\n".join(p_npy for p_npy in output_path_dict.values())
            raise FileExistsError("At least one of these files already exists:\n"
                                  f"{file_str}")

        Path(output_path_dict["noise"]).parent.mkdir(exist_ok=True, parents=True)
        Path(output_path_dict["clean"]).parent.mkdir(exist_ok=True, parents=True)
        if "nsite" in output_path_dict:
            Path(output_path_dict["nsite"]).parent.mkdir(exist_ok=True, parents=True)

        np.save(output_path_dict["noise"], pulse_arr_noisy, allow_pickle=False)
        np.save(output_path_dict["clean"], pulse_arr_clean, allow_pickle=False)
        if "nsite" in output_path_dict:
            np.save(output_path_dict["nsite"], n_sites, allow_pickle=False)

    results_dict = {"clean": pulse_arr_clean,
                    "noisy": pulse_arr_noisy,
                    "nsite": n_sites}

    return results_dict
