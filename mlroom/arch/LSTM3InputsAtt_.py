import numpy as np
from keras.models import Model
import keras.layers as layers
from keras.optimizers import Adam

def create_model_for_resolution(input_shape, lstm_units, dense_units, name):
    input_layer = layers.Input(shape=input_shape, name=f"{name}_input")
    lstm_layer = layers.LSTM(lstm_units, return_sequences=True)(input_layer)
    attention_layer = layers.Attention()([lstm_layer, lstm_layer])
    flatten_layer = layers.Flatten()(attention_layer)
    dense_layer = layers.Dense(dense_units, activation='relu')(flatten_layer)
    return input_layer, dense_layer

def LSTM3InputsAtt_(input_shapes, **params):
    learning_rate = 0.001
    if "learning_rate" in params:
        learning_rate = float(params["learning_rate"])

    # Create separate models for each resolution
    input_high_res, attention_high_res = create_model_for_resolution(input_shapes[0], 64, 64, "high_res")
    input_low_res, attention_low_res = create_model_for_resolution(input_shapes[1], 32, 64, "low_res")
    input_lowest_res, attention_lowest_res = create_model_for_resolution(input_shapes[2], 32, 64, "lowest_res")

    # Combine the features from all resolutions
    combined = layers.concatenate([attention_high_res, attention_low_res, attention_lowest_res])

    # Additional layers for further processing
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, name="output", activation="tanh")(x)  # Example for regression

    # Create the model
    model = Model(inputs=[input_high_res, input_low_res, input_lowest_res], outputs=x)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])

    return model, {}

# Usage example
# input_shapes = [(75, 3), (30, 6), (25, 3)]
# model, _ = LSTM3Inputs_Attention(input_shapes)
