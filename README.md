[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GePPCLab/NN-PSA-tutorial/main)

# Pulse shape analysis and discrimination tutorial

This is a repository to illustrate how to train a basic multi-site event classifier
using simulated pulses from a high purity p-point contact germanium detector.
The tutorial uses [TensorFlow](https://www.tensorflow.org/)
with the [Keras](https://keras.io/) API.

## Setup

### Configure the environment

**If using Binder**, the container will already contain all
of the required dependencies, and you can skip to the next section.
However, please read the
[usage guidelines](https://mybinder.readthedocs.io/en/latest/about/user-guidelines.html)
for Binder. In particular:

> Binder is meant for interactive and ephemeral interactive coding,
> meaning that it is ideally suited for relatively short sessions.
> Binder will automatically shut down user sessions that have more than
> 10 minutes of inactivity (if you leave a jupyterlab window open in the foreground,
> this will generally be counted as “activity”).

The takeaway is that these sessions are not persistent,
so if you make any changes you would like to keep,
make sure to download the relevant files before exiting.

Also, the following files in this repository are used
when creating the Binder image:

* `apt.txt`
* `requirements.txt`
* `runtime.txt`

**If using your own computer**, you will need to ensure that all
relevant dependencies are installed.
It is recommended to use a virtual environment.
To construct a virtual environment with the name `env`
and install all of the required dependencies
(including Jupyter for the notebooks), run the following:

```sh
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel
python3 -m pip install jupyter jupyterlab
python3 -m pip install -r requirements.txt
```

This only needs to be done once for the initial setup.
Afterwards, the environment will continue to exist in the
`env` folder with the required dependencies.
Note that you may call the environment something other than `env`.

When coming back to use the repository,
run `source env/bin/activate` to activate the environment.
To exit from the virtual environment, simply call `deactivate`.

For other dependencies, see the `apt.txt` file.
In particular, you will need [Graphviz](https://graphviz.org/)
in order to plot the TensorFlow model diagrams.
However, the notebook will still work without it.

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

## Build and train the classifier

A full tutorial for building and training a TensorFlow model from scratch
is provided in the `notebooks/tutorial_classifier_training.ipynb`
Jupyter notebook.
This file contains detailed explanations of each step
in the model building and training process,
as well as numerous helpful links to documentation and other resources.

For more advanced training, take a look at the `train_classifiers.py` script.
This effectively accomplishes the same thing as the notebook, but the details
of the model and parameters are more abstracted, and there is no tutorial.
It also contains some other features not covered in the notebook,
including the TensorFlow Dataset API.

Run `./train_classifiers.py --help` for a list of all arguments (required and optional).
Definitions of the neural networks are contained in the `networks/classifiers.py` file.
An example command to train a convolutional neural network is:

```
./train_classifier.py data/simulations --model-name cnn --model-dir models/classifiers --model-type cnn --X-pattern "X_noisy*.npy" --y-pattern "y_nsite*.npy" --nepochs 10
```

To create your own model using this script,
simply add another function to the `networks/classifiers.py` file.
You will also need to ensure that you give it a unique name
and check for it when contstructing the model in the `train_classifiers.py` script
(it should be a choice for the required `--model-type` argument).

## Analyze the classifier

A full tutorial for analyzing a trained TensorFlow model
is provided in the `notebooks/tutorial_classifier_analysis.ipynb`
Jupyter notebook.
At the end, a series of questions are provided.
You are encouraged to think about and answer as many of these as you can.
Do note that some questions are open-ended
while others have a more direct answer.
