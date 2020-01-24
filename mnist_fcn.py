import tensorflow as tf
import numpy as np

tf.debugging.set_log_device_placement(True)

# 10 digits
NUM_CLASSES = 10

# 28 x 28 image size
# 1 channel: grey
IMAGE_SIZE = 28

mnist = tf.keras.datasets.mnist

(feature_train_flat, label_train), (feature_test_flat, label_test) = mnist.load_data()
feature_train_flat, feature_test_flat = feature_train_flat / 255.0, feature_test_flat / 255.0

fcn = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

fcn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


fcn.fit(feature_train_flat, label_train, epochs=5)

fcn.evaluate(feature_test_flat,  label_test, verbose=2)
