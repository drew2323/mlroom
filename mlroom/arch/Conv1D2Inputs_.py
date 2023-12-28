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
from keras.optimizers import Adam

def Conv1D2Inputs_(input_shape, **params):
    custom_layers = {}
    learning_rate = params.get("learning_rate", 0.001)

    # Main input and its layers
    main_input = Input(shape=input_shape[0], name='main_input')
    x1 = Conv1D(filters=64, kernel_size=3, activation='relu')(main_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Conv1D(filters=128, kernel_size=3, activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)

    # Additional input 1 and its layers
    additional_input1 = Input(shape=input_shape[1], name='additional_input1')
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(additional_input1)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Conv1D(filters=128, kernel_size=3, activation='relu')(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Flatten()(x2)

    # Concatenate the outputs from each input stream
    merged = Concatenate()([x1, x2])

    # Further processing after concatenation
    merged = Dense(100, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(50, activation='relu')(merged)
    merged = Dropout(0.5)(merged)

    # Output layer
    output = Dense(1, activation='tanh')(merged)  # Single output neuron with tanh activation

    # Create and compile the model
    model = Model(inputs=[main_input, additional_input1], outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])

    return model, custom_layers
