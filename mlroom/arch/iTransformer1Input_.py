import os
os.environ["KERAS_BACKEND"] = "jax"
from keras.models import Model
from keras.layers import (
    MultiHeadAttention, LayerNormalization, Dense, Dropout, 
    Input, GlobalAveragePooling1D, Concatenate, Permute, Reshape
)
from keras.optimizers import Adam
from keras_nlp.layers import SinePositionEncoding
from keras.regularizers import l2

"""
Based on iTransformer architecture with feature-wise attention
https://claude.ai/chat/a026cdda-39bc-4f7f-b7f0-57e7f007ca06
https://freedium.cfd/https://towardsdatascience.com/itransformer-the-latest-breakthrough-in-time-series-forecasting-d538ddc6c5d1
also look at
https://github.com/Nixtla/neuralforecast

"""


def feature_projection(x, projection_dim):
    """Project features to a different dimension"""
    return Dense(projection_dim)(x)

def itransformer_block(inputs, num_heads, key_dim, dropout_rate=0.1):
    """
    iTransformer block with feature-wise attention
    """
    # Feature-wise attention (transpose input to make features attend to features)
    x = Permute((2, 1))(inputs)  # Switch time and feature dimensions
    
    # Apply attention across features
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x, x)
    
    attention_output = Dropout(dropout_rate)(attention_output)
    x1 = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # FFN
    ffn = Dense(key_dim * 4, activation="gelu")(x1)
    ffn = Dense(x1.shape[-1])(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    
    # Switch back to original dimension order
    output = LayerNormalization(epsilon=1e-6)(x1 + ffn)
    output = Permute((2, 1))(output)
    
    return output

def iTransformer1Input(input_shape, **params):
    """
    iTransformer architecture for dual time series input
    
    Args:
        input_shape: Tuple of input shapes for both time series
        params: Dictionary containing model parameters
            - learning_rate: Learning rate for Adam optimizer
            - trans_layers: List of number of transformer layers for each input
            - l2_reg: L2 regularization factor
            - projection_dim: Dimension for feature projection
            - num_heads: Number of attention heads
            - dropout_rate: Dropout rate
    
    Returns:
        model: Compiled Keras model
        custom_layers: Dictionary of custom layers used
    """
    custom_layers = {}
    
    # Get parameters with defaults
    learning_rate = params.get("learning_rate", 0.001)
    trans_layers = params.get("trans_layers", [1, 1])
    l2_reg = params.get("l2_reg", 0.001)
    projection_dim = params.get("projection_dim", 64)
    num_heads = params.get("num_heads", 4)
    dropout_rate = params.get("dropout_rate", 0.1)
    
    # Input layers
    input_ts1 = Input(shape=input_shape[0])
    
    # Project features if needed
    x1 = feature_projection(input_ts1, projection_dim)
    
    # Add positional encoding
    pos_encoding_1 = SinePositionEncoding()(x1)
    
    x1 = pos_encoding_1 + x1
    
    # Apply iTransformer blocks
    for _ in range(trans_layers[0]):
        x1 = itransformer_block(
            x1,
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout_rate=dropout_rate
        )
    
    # Global pooling
    x1 = GlobalAveragePooling1D()(x1)

    
    # Final layers
    x = Dense(
        64,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(x1)
    x = Dropout(dropout_rate)(x)
    
    # Output layer for classification
    output = Dense(
        3,
        activation='softmax',
        kernel_regularizer=l2(l2_reg)
    )(x)
    
    # Build and compile model
    model = Model(inputs=[input_ts1], outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, custom_layers

# Example usage:
"""
# Define input shapes for two time series
input_shape_1 = (sequence_length_1, num_features_1)

# Create model with custom parameters
model_params = {
    "learning_rate": 0.001,
    "trans_layers": [2, 2],
    "l2_reg": 0.001,
    "projection_dim": 64,
    "num_heads": 4,
    "dropout_rate": 0.1
}

model, custom_layers = iTransformer2Inputs(
    input_shape=(input_shape_1, input_shape_2),
    **model_params
)
"""