import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
import keras.layers as layers
from keras.optimizers import Adam

def LSTM3Inputs_(input_shape, **params):

    learning_rate = 0.001
    if "learning_rate" in params:
        learning_rate = float(params["learning_rate"])
        
    # Define the input shape for each resolution
    input_shape_high_res = input_shape[0] # E.g., hourly data
    input_shape_low_res = input_shape[1]   # E.g., daily data
    input_shape_lowest_res = input_shape[2]
    # Define the LSTM layers for each input
    # High resolution input
    input_high_res = layers.Input(shape=input_shape_high_res, name="high_res_input")
    lstm_high_res = layers.LSTM(64)(input_high_res)

    # Low resolution input
    input_low_res = layers.Input(shape=input_shape_low_res, name="low_res_input")
    lstm_low_res = layers.LSTM(32)(input_low_res)

    # Low resolution input
    input_lowest_res = layers.Input(shape=input_shape_lowest_res, name="lowest_res_input")
    lstm_lowest_res = layers.LSTM(32)(input_lowest_res)

    # Combine the features from both resolutions
    combined = layers.concatenate([lstm_high_res, lstm_low_res, lstm_lowest_res])

    # Additional layers for further processing
    # You can add more layers like Dense layers depending on your requirement
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, name="output", activation="tanh")(x)  # Example for regression

    # Create the model
    model = Model(inputs=[input_high_res, input_low_res, input_lowest_res], outputs=x)

    optimizer = Adam(learning_rate=learning_rate)
    #print(model.__dict__)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    return model, {}