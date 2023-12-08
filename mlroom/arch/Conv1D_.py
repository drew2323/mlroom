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


def Conv1D_(input_shape, **params):
    learning_rate = 0.001
    if "learning_rate" in params:
        learning_rate = float(params["learning_rate"])
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
    optimizer = Adam(learning_rate=learning_rate)

    #print(model.__dict__)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    return model