from tensorflow import expand_dims
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Activation, Lambda,
                                     Conv1D, AveragePooling1D, Flatten)

def nn_dense(input_tuple, n_outputs, activation='relu', output_activation='softmax'):
    # Define the input and specify the input shape.
    input_signal = Input(shape=input_tuple)

    layer = Dense(units=256, activation=activation)(input_signal)
    layer = Dense(units=128, activation=activation)(layer)
    layer = Dense(units=64, activation=activation)(layer)
    layer = Dense(units=32, activation=activation)(layer)
    layer = Dense(units=16, activation=activation)(layer)

    output = Dense(units=n_outputs, activation=output_activation)(layer)

    # Create the model.
    model = Model(input_signal, output)

    return model

def nn_conv(input_tuple, n_outputs, activation='relu', output_activation='softmax'):
    # Define the input and specify the input shape.
    input_signal = Input(shape=input_tuple)

    # Add another dimension for the convolutions.
    input_cnn = Lambda(lambda x: expand_dims(x, axis=-1))(input_signal)

    layer = Conv1D(filters=4, kernel_size=15, strides=1, padding='valid')(input_cnn)
    layer = Activation(activation)(layer)
    layer = AveragePooling1D(2)(layer)

    layer = Conv1D(filters=8, kernel_size=11, strides=1, padding='valid')(layer)
    layer = Activation(activation)(layer)
    layer = AveragePooling1D(2)(layer)

    layer = Conv1D(filters=4, kernel_size=7, strides=1, padding='valid')(layer)
    layer = Activation(activation)(layer)
    layer = AveragePooling1D(2)(layer)

    # Flatten for compatibility with the following dense layer.
    layer = Flatten()(layer)

    layer = Dense(units=64, activation=activation)(layer)
    layer = Dense(units=32, activation=activation)(layer)
    layer = Dense(units=16, activation=activation)(layer)

    output = Dense(units=n_outputs, activation=output_activation)(layer)

    # Create the model.
    model = Model(input_signal, output)

    return model
