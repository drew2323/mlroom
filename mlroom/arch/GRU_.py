from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Input

def GRU_(input_shape, **params):
    # Define the model
    model = Sequential()

    # Explicitly define the Input layer
    model.add(Input(shape=input_shape))
    # Add a GRU layer with dropout
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))  # Dropout of 20%

    # Add another GRU layer with dropout
    model.add(GRU(units=50))
    model.add(Dropout(0.2))  # Dropout of 20%

    # Add a Dense layer with 'tanh' activation to output values between -1 and 1
    model.add(Dense(1, activation='tanh'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')   

    return model, {}
