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
#TODO jak ulozit architecturu a mit jit
# import tensorflow as tf
# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

def LSTM_(input_shape, **params):
    # Define the model
    model = Sequential()

    # Explicitly define the Input layer
    model.add(Input(shape=input_shape[0]))
    # Add an LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))  # Dropout of 20%
    # Add another LSTM layer with dropout
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # Dropout of 20%
    # Add a Dense layer with 'tanh' activation to output values between -1 and 1
    model.add(Dense(1, activation='tanh'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')   

    return model, {}