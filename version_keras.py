import tensorflow as tf
from tensorflow.keras.datasets import mnist

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("MNIST data loaded successfully.")
