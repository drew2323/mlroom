import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM
from keras.optimizers import Adam

def modelLSTM(input_shape):
    # Define the model
    model = Sequential()

    # Add an LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout of 20%
    # Add another LSTM layer with dropout
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # Dropout of 20%
    # Add a Dense layer with 'tanh' activation to output values between -1 and 1
    model.add(Dense(1, activation='tanh'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')   

    return model


def modelConv1D(input_shape):
    model = Sequential()
    # Convolutional layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(1, activation='tanh'))  # Single output neuron with tanh activation
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    return model

def modelConv1DLR(input_shape, initial_learning_rate = 0.001):
    # Create the model
    model = Sequential()
    # Convolutional layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(1, activation='tanh'))  # Single output neuron with tanh activation

    # Compile the model with a custom learning rate
    optimizer = Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    return model