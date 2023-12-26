from keras.models import Sequential
from tcn import TCN, tcn_full_summary
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM
from keras.optimizers import Adam


"""
Temporal Convolution Network (https://github.com/philipperemy/keras-tcn)

https://chat.openai.com/g/g-GzdUmx53z-market-strategist

zatím jednoduchá TCN, v případě je možno vrstvit s return_sequences=True (jako u LSTM) a nebo Conv1D
"""

def TCN_(input_shape, **params):
    custom_layers = {
        'TCN': TCN
    }
    learning_rate = 0.001
    if "learning_rate" in params:
        learning_rate = float(params["learning_rate"])
    model = Sequential()
    # Building the TCN model
    model.add(
        TCN(input_shape=input_shape[0], 
            nb_filters=128, 
            kernel_size=5, #If sequence heavily depends on t-1 and t-2, but less on the rest, then choose a kernel size of 2/3.
            dilations=[1, 2, 4, 8], 
            padding='causal', 
            use_skip_connections=True))
    model.add(Dense(1, activation='tanh'))  # Single output neuron with tanh activation
    # Compile the model with a custom learning rate
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    tcn_full_summary(model, expand_residual_blocks=False) 
    return model, custom_layers