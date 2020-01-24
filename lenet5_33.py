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

feature_train_arr = feature_train_flat.reshape(60000, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32')
feature_test_arr = feature_test_flat.reshape(10000, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32')

feature_train_32 = np.pad(feature_train_arr, ((0,0),(2,2),(2,2),(0,0)), 'constant')
feature_test_32  = np.pad(feature_test_arr, ((0,0),(2,2),(2,2),(0,0)), 'constant')


lenet5_33 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
  tf.keras.layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
  tf.keras.layers.AvgPool2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.AvgPool2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(120, activation='relu'),
  tf.keras.layers.Dense(84, activation='relu'),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

lenet5_33.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


lenet5_33.fit(feature_train_32, label_train, epochs=5)
lenet5_33.evaluate(feature_test_32,  label_test, verbose=2)
