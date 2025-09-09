import tensorflow as tf
import tensorflow_federated as tff

print("TFF version:", tff.__version__)

# Definisi fungsi Hello World pakai tf_computation
@tff.tensorflow.computation
def hello_world():
    return tf.constant("Hello World from TFF")

# Jalankan fungsi
print(hello_world())