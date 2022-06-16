import argparse

def process_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str,
                        help=('Path to the folder of input files. '
                              'Expects the folder to contain train and val folders.'))

    # Required named arguments

    required = parser.add_argument_group('required arguments')

    required.add_argument('--model-name', type=str, required=True,
                        help='Name of the model to save.')

    required.add_argument('--model-dir', type=str, required=True,
                        help='Subfolder to save the model into.')

    required.add_argument('--model-type', type=str, required=True,
                          help='Type of model (e.g., dense, CNN).')

    # Optional named arguments

    parser.add_argument('--X-pattern', type=str, default='X_noisy_*.npy',
                        help=('Pattern for the training input files. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--y-pattern', type=str, default='y_nsite_*.npy',
                        help=('Pattern for the training output files. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--X-pattern-val', type=str, default=None,
                        help=('Pattern for the validation input files. '
                              'Defaults to the value of X_pattern'))

    parser.add_argument('--y-pattern-val', type=str, default=None,
                        help=('Pattern for the validation output files. '
                              'Defaults to the value of y_pattern.'))

    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help=('Learning rate to use in training. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--nepochs', type=int, default=100,
                        help=('Number of epochs to use in training. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--batch-size', type=int, default=128,
                        help=('Batch size to use in training. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--batch-size-val', type=int, default=None,
                        help=('Batch size to use in validation. '
                              'Defaults to the value of batch_size.'))

    parser.add_argument('--prefetch', type=int, default=50,
                        help=('Number of batches to prefetch '
                              'in Dataset for training. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--prefetch-val', type=int, default=None,
                        help=('Number of batches to prefetch '
                              'in Dataset for validation. '
                              'Defaults to the value of prefetch.'))

    parser.add_argument('--shuffle-buffer', type=int, default=20000,
                        help=('Size of shuffle buffer for Dataset. '
                              'Defaults to: %(default)s.'))

    parser.add_argument('--autoencoder-path', type=str, default=None,
                        help=('Path to autoencoder for classifier. '
                              'Only needed when model_type == "autoencoder", '
                              'otherwise unused.'))

    # TensorFlow configuration parameters

    parser.add_argument('--gpu', type=int, default=None,
                        help=('GPU to use in training and validation. '
                              'If set to None, will select default GPU '
                              'if available. Disable usage of GPU with -1. '
                              'Defaults to: %(default)s.'))

    # Display

    parser.add_argument('--verbose', type=int, default=1,
                        help=('Verbosity parameter passed to the '
                              'tf.keras.Model.fit method. '
                              'Defaults to: %(default)s.'))

    args = parser.parse_args()

    # Handle certain default arguments.

    if args.X_pattern_val is None:
        args.X_pattern_val = args.X_pattern

    if args.y_pattern_val is None:
        args.y_pattern_val = args.y_pattern

    if args.batch_size_val is None:
        args.batch_size_val = args.batch_size

    if args.prefetch_val is None:
        args.prefetch_val = args.prefetch

    return args
