import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Generate dummy time series data
data = np.random.rand(1000, 10, 1)
targets = np.random.rand(1000, 1)

# Build model
model = tf.keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(10, 1)),
    layers.LSTM(50),
    layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(data, targets, epochs=10, batch_size=32)
