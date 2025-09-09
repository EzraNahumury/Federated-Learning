import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# try :
#     constant_10 = tf.constant(10)

#     @tff.tensorflow.computation(np.float32)
#     def add_ten(x):
#         return x + constant_10

# except Exception as err :
#     print(err)

def get_constant_10():
    return tf.constant(10.0)
@tff.tensorflow.computation(np.float32)
def add_ten(x):
    return x + get_constant_10()

print(add_ten(5.0))