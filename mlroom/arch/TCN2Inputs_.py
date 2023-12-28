from keras.models import Sequential, Model
from tcn import TCN, tcn_full_summary
from keras.layers import Input,Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, concatenate
from keras.optimizers import Adam


"""
Temporal Convolution Network (https://github.com/philipperemy/keras-tcn)

https://chat.openai.com/g/g-GzdUmx53z-market-strategist

zatím jednoduchá TCN, v případě je možno vrstvit s return_sequences=True (jako u LSTM) a nebo Conv1D
"""

def TCN2Inputs_(input_shapes, **params):
    custom_layers = {
        'TCN': TCN
    }
    learning_rate = params.get("learning_rate", 0.001)

    # Define two separate input layers
    input1 = Input(shape=input_shapes[0])
    input2 = Input(shape=input_shapes[1])

    # First TCN layer for the first input
    tcn1 = TCN(nb_filters=128, 
               kernel_size=5,
               dilations=[1, 2, 4, 8], 
               padding='causal', 
               use_skip_connections=True)(input1)

    # Second TCN layer for the second input, parameters can be different
    tcn2 = TCN(nb_filters=64, 
               kernel_size=3, 
               dilations=[1, 2, 4, 8], 
               padding='causal', 
               use_skip_connections=True)(input2)

    # Merging both TCN layers
    merged = concatenate([tcn1, tcn2])

    # Output layer
    output = Dense(1, activation='tanh')(merged)

    # Creating the model
    model = Model(inputs=[input1, input2], outputs=output)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])

    return model, custom_layers