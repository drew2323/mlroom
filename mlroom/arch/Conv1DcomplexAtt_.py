import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from collections import defaultdict
# from operator import itemgetter
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, BatchNormalization, AveragePooling1D, MaxPooling1D, Layer
from keras.optimizers import Adam
import keras.backend as K

class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(SelfAttention, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

def Conv1DcomplexAtt_(input_shape, **params):
    learning_rate = params.get("learning_rate", 0.001)
    model = Sequential()
    
    # Convolutional Layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Self-Attention Layer
    model.add(SelfAttention())

    # Flatten Layer
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='tanh'))

    # Compile Model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])

    return model
