import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, Input
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Concatenate
from keras.optimizers import Adam

from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Concatenate
from keras.layers import Multiply, Permute, Reshape, Lambda, RepeatVector, LSTM, TimeDistributed, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

def attention_mechanism(inputs):
    # Assuming inputs.shape = (batch_size, time_steps, features)
    time_steps = inputs.shape[1]  # Extracting the time steps dynamically

    # Attention mechanism
    a = Permute((2, 1))(inputs)  # Swap the time and feature axes
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    
    return output_attention_mul

def Conv1DLSTM2InputsAtt_(input_shape, **params):
    """
    Constructs a hybrid neural network model combining Conv1D, LSTM, and attention mechanisms.
    
    This model is tailored for time series analysis, capturing both feature-based patterns using 
    convolutional layers and temporal dynamics through LSTM layers. The attention mechanism 
    further enhances the model's focus on relevant time steps, improving prediction accuracy.

    Parameters:
    - input_shape (tuple of tuples): The shape of the input data, specified for each input stream.
    - params (dict): Additional parameters such as learning rate.

    The architecture is as follows:
    - Two parallel input streams, each passing through Conv1D and MaxPooling1D layers,
      followed by LSTM layers.
    - An attention mechanism applied to each LSTM output.
    - Concatenation of the outputs from each stream.
    - Dense layers following the concatenation for further processing.
    - A final output layer with a single neuron and tanh activation for regression output.

    Returns:
    - model (keras.Model): The compiled Keras model ready for training.
    - custom_layers (dict): Dictionary of custom layers used in the model.
    
    https://chat.openai.com/g/g-GzdUmx53z-market-strategist/c/cc65d36d-7b2d-41c1-a87c-dfdba775e997
    """
    custom_layers = {}
    learning_rate = params.get("learning_rate", 0.001)

    # Main input and its layers
    main_input = Input(shape=input_shape[0], name='main_input')
    x1 = Conv1D(filters=64, kernel_size=5, activation='relu')(main_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = LSTM(50, return_sequences=True)(x1)
    x1 = attention_mechanism(x1)
    x1 = Flatten()(x1)

    # Additional input 1 and its layers
    additional_input1 = Input(shape=input_shape[1], name='additional_input1')
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(additional_input1)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = BatchNormalization()(x2)
    x2 = LSTM(50, return_sequences=True)(x2)
    x2 = attention_mechanism(x2)
    x2 = Flatten()(x2)

    # Concatenate the outputs from each input stream
    merged = Concatenate()([x1, x2])

    # Further processing after concatenation
    merged = Dense(100, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(50, activation='relu')(merged)
    merged = Dropout(0.5)(merged)

    # Output layer
    output = Dense(1, activation='tanh')(merged)

    # Create and compile the model
    model = Model(inputs=[main_input, additional_input1], outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])

    return model, {}