#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

from tools.train.arguments import process_arguments

######################################################################################
# Process arguments.
######################################################################################

args = process_arguments()

######################################################################################
# Finish configuration and imports.
######################################################################################

from tools.configure import configure_tensorflow
configure_tensorflow(gpu_index=args.gpu, allow_growth_gpu=True)

import tensorflow as tf
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

from tools.data.tfdataset import create_dataset

from networks.classifiers import nn_dense, nn_conv

######################################################################################
# Callback definitions for saving the model.
######################################################################################

# Construct model path and create callback for saving best model.
model_path = Path(args.model_dir, args.model_name)
model_path.parent.mkdir(exist_ok=True, parents=True)
model_checkpoint = ModelCheckpoint(model_path, monitor="val_loss",
                                   save_best_only=True, mode="min")

# Save the exact command used, based on the model name above.
info_path = Path(args.model_dir, model_path.stem + ".txt")
info = ' '.join(sys.argv)
with open(info_path, 'w', encoding="utf-8") as f:
    f.write(info)
    f.write('\n')

######################################################################################
# Get lists of all the data files and create a TensorFlow Dataset.
######################################################################################

list_X_train = sorted(str(f) for f in Path(args.data_path, 'train').glob(args.X_pattern))
list_y_train = sorted(str(f) for f in Path(args.data_path, 'train').glob(args.y_pattern))

list_X_val = sorted(str(f) for f in Path(args.data_path, 'val').glob(args.X_pattern_val))
list_y_val = sorted(str(f) for f in Path(args.data_path, 'val').glob(args.y_pattern_val))

if len(list_X_train) != len(list_y_train):
    raise ValueError("Number of files for training input and target lists must match.")

if len(list_X_val) != len(list_y_val):
    raise ValueError("Number of files for validation input and target lists must match.")

# Train dataset.
dataset_train = create_dataset(list_X_train, list_y_train,
                               type_X=tf.float32, type_y=tf.int64,
                               batch_size=args.batch_size,
                               shuffle_buffer=args.shuffle_buffer,
                               prefetch=args.prefetch,
                               binarize_sse_mse=True)

# Validation dataset.
dataset_val = create_dataset(list_X_val, list_y_val,
                             type_X=tf.float32, type_y=tf.int64,
                             batch_size=args.batch_size_val,
                             shuffle_buffer=None,
                             prefetch=args.prefetch_val,
                             binarize_sse_mse=True)

# Specify validation steps, which is annoyingly required.
# The +1 is to get the remainder (which is less than the batch size).
val_rows = sum(np.load(filepath, mmap_mode='r').shape[0] for filepath in list_X_val)
validation_steps = int(val_rows / args.batch_size_val) + 1

######################################################################################
# Build the Model
######################################################################################

# Optimizer
opt = optimizers.Adam(learning_rate=args.learning_rate)

# Metrics
eval_metrics = [metrics.categorical_crossentropy,
                metrics.categorical_accuracy]

input_tuple = (512,)

# Load in the model.
if args.model_type.lower() == "dense":
    model = nn_dense(input_tuple=input_tuple, n_outputs=2)
elif args.model_type.lower() == "cnn":
    model = nn_conv(input_tuple=input_tuple, n_outputs=2)
else:
    raise ValueError(f"Invalid model type: {args.model_type}.")

# Compile the model, specifying the loss function and optimizer.
model.compile(loss=metrics.categorical_crossentropy,
              optimizer=opt,
              metrics=eval_metrics)

model.summary()

history = model.fit(dataset_train,
                    epochs=args.nepochs,
                    verbose=args.verbose,
                    validation_data=dataset_val,
                    validation_steps=validation_steps,
                    callbacks=[model_checkpoint])
