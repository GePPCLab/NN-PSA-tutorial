[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GePPCLab/NN-PSA-tutorial/HEAD)

# Pulse shape analysis and discrimination tutorial

This a repository to illustrate how to train a basic multi-site event classifier
using simulated pulses from a high purity p-point contact germanium detector.
The tutorial uses TensorFlow with the Keras API.

## Setup

### Configure the environment

If using Binder, the container will already contain all
of the required dependencies, and you can skip to the next section.

If using your own computer, you will need to ensure that all
relevant dependencies are installed.
It is recommended to use a virtual environment.
To construct a virtual environment with the name `env`
and install all of the required dependencies, run the following:

```
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel
python3 -m pip install -r requirements.txt
```

This only needs to be done once for the initial setup.
Afterwards, the environment will continue to exist in the
`env` folder with the required dependencies.
Note that you may call the environment something other than `env`.

When coming back to use the repository,
run `source env/bin/activate` to activate the environment.
To exit from the virtual environment, simply call `deactivate`.

### Download the data

The data are not stored in the repository.
Instead, call the `get_data.sh` script (`./get_data.sh`)
to download the data from an external source.
This will download a ZIP archive in the folder `data/`
containing two files:

- `simulations/library.npy`
- `noise/noise.npy`

The first file is a set of unique single-site event library pulses.
The second file is a large noise set used in the data augmentation.

## Generate the training, validation, and test data

To generate the training data, run `./generate_data.py`.
This will take the single-site library pulse file and generate
multi-site events with noise. Options for the data augmentation and noise
addition can be changed by modifying the relevant values in the script.
Augmentation options include shifts and scales,
the level of noise to be added,
the type of noise to be added (Gaussian noise or real detector noise),
and the proportion of multi-site events to generate.
