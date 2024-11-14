import numpy as np
import os
os.environ["KERAS_BACKEND"] = "jax"
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Concatenate, LayerNormalization, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, BatchNormalization, AveragePooling1D, Input
from keras.optimizers import Adam
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder

# Define Transformer block - bud takto explciitne a nebo TransformerEncoder
def transformer_block(inputs, num_heads, ff_dim, rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention_output = Dropout(rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(rate)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

def Transformer1Input_(input_shape, **params):
    """
    - Neural network model for dual time series with varying sequence lengths and features.
    - Incorporates two TransformerEncoder layers, each tailored to respective input features (4 and 12 features).
    - Applies Sine Positional Encoding for sequence order in both time series.
    - Processes each series independently, then merges outputs.
    - Final classification via dense layers with softmax activation.
    - Uses Adam optimizer and sparse categorical cross-entropy loss.
    - Adjustable for specific sequence lengths and data characteristics.

    Notes:
    - ruzna delka sekvence obou rozliseni je zkonzolidovana pomoci GlobalAveragePooliong aby slo zconcatetovanat (lehka ztrata data)
    - bez ztráty dat bud stejná délka sekvencí a nebo využít Masking - viz https://chat.openai.com/c/ad21a774-30c0-4bdc-b25e-8de1bdd256b4

    """
    custom_layers = {}

    learning_rate = params.get("learning_rate", 0.001)

    # Define TransformerEncoder for each input
    # transformer_encoder_1 = Transformer3Encoder(intermediate_dim=128, num_heads=4)  # For input with 4 features
    # transformer_encoder_2 = TransformerEncoder(intermediate_dim=128, num_heads=12)  # For input with 12 features

    # Inputs for each time series with different sequence lengths
    input_ts1 = Input(shape=input_shape[0])  # Adjust sequence_length_1 as needed
    #input_ts2 = Input(shape=input_shape[1])  # Adjust sequence_length_2 as needed

    # Sine Positional Encoding for each time series
    pos_encoding_1 = SinePositionEncoding()(input_ts1)
    #pos_encoding_2 = SinePositionEncoding()(input_ts2)

    #add encoding to the input
    comb_input_ts1 = pos_encoding_1 + input_ts1

    # Transformer block for each time series
    transformer_ts1 = transformer_block(comb_input_ts1, num_heads=4, ff_dim=2048)
    #transformer_ts2 = transformer_block(pos_encoding_2, num_heads=12, ff_dim=2048)

    # Apply Transformer Encoder to each input
    # transformed_ts1 = transformer_encoder_1(pos_encoding_1)
    # transformed_ts2 = transformer_encoder_2(pos_encoding_2)

    # After the TransformerEncoder layers - to allow concatenation
    # pooled_ts1 = GlobalAveragePooling1D()(transformer_ts1)
    # pooled_ts2 = GlobalAveragePooling1D()(transformer_ts2)

    # Then concatenate
    # combined = Concatenate()([pooled_ts1, pooled_ts2])
    #combined = Concatenate()([transformer_ts1, transformer_ts2])

    pooled = GlobalAveragePooling1D()(transformer_ts1)

    # Additional layers
    x = Dense(64, activation='relu')(pooled)
    #output = Dense(3, activation='softmax')(x)  # For trend classification: downtrend, no trend, uptrend
    output = Dense(1, activation='tanh')(x)  # Single output neuron with tanh activation

    # Build the model
    model = Model(inputs=[input_ts1], outputs=output) #, input_ts2

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    #model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])

    return model, custom_layers
