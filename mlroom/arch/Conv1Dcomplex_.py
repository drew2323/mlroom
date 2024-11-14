import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, BatchNormalization, AveragePooling1D, Input
from keras.optimizers import Adam

def Conv1Dcomplex_(input_shape, **params):
    custom_layers = {}

    learning_rate = params.get("learning_rate", 0.001)
    model = Sequential()
    
    # Explicitly define the Input layer
    model.add(Input(shape=input_shape[0]))
    # First Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2))

    # Second Convolutional Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2))

    # Third Convolutional Layer
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2))

    model.add(Flatten())

    # First Dense Layer
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.4))  # Adjusted dropout rate

    # Output Layer
    model.add(Dense(1, activation='linear'))  # Retain tanh for output in the range of -1 to 1

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    return model, custom_layers
