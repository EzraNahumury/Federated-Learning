
############################### SETUP ###############################
import tensorflow as tf


import numpy as np
tf.get_logger().setLevel('ERROR')

import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy



############################### Load and pre-process the dataset ###############################
train, test = tf.keras.datasets.mnist.load_data()
train_data, train_label = train
test_data , test_label = test

train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_label = np.array(train_label, dtype=np.int32)
test_label = np.array(test_label, dtype=np.int32)

train_label = tf.keras.utils.to_categorical(train_label, num_classes=10)
test_label = tf.keras.utils.to_categorical(test_label, num_classes=10)

assert train_data.min() == 0
assert train_data.max() == 1
assert test_data.min() == 0
assert test_data.max() == 1


############################### Define the hyperparameters ###############################
epochs = 3
batch_size = 250

l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 250
learning_rate = 0.25

if batch_size % num_microbatches != 0 :
    raise ValueError ('Batch size should be an integer multiple of the number of microbatches')





############################### Build the model ###############################
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,1),
    tf.keras.layers.Conv2D(32,4, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(2,1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])

optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate
)
loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits = True, reduction=tf.losses.Reduction.NONE
)



############################### Train the model ###############################
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(train_data,
          train_label,
          epochs=epochs,
          validation_data=(test_data, test_label),
          batch_size=batch_size)





############################### Measure the differential privacy guarantee ###############################
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

epsilon, best_alpha = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
    n=train_data.shape[0],
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=epochs,
    delta=1e-5
)

print(f"DP-SGD with ε = {epsilon:.2f}, δ = 1e-5. For α = {best_alpha}.")
