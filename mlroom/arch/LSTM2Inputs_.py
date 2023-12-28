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
from keras.regularizers import l2
from keras.optimizers import Adam

def LSTM2Inputs_(input_shape, **params):

    learning_rate = params.get("learning_rate", 0.001)
    l2_reg = params.get("l2_reg", 0.001)  # Added L2 regularization parameter

    # Define the input shape for each resolution
    input_shape_high_res, input_shape_low_res = input_shape

    # High resolution input
    input_high_res = layers.Input(shape=input_shape_high_res, name="high_res_input")
    lstm_high_res = layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(l2_reg))(input_high_res)

    # Low resolution input
    input_low_res = layers.Input(shape=input_shape_low_res, name="low_res_input")
    lstm_low_res = layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(l2_reg))(input_low_res)

    # Combine the features from all resolutions
    combined = layers.concatenate([lstm_high_res, lstm_low_res])

    # Additional layers for further processing
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    x = layers.Dropout(0.3)(x)  # Increased dropout
    x = layers.Dense(1, name="output", activation="tanh")(x)  # Regression output

    # Create the model
    model = Model(inputs=[input_high_res, input_low_res], outputs=x)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    return model, {}