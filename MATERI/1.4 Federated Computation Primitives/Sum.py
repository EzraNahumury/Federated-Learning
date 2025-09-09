import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

@tff.tensorflow.computation(np.float32)
def add_half(x):
    return tf.add(x, 0.5)

# print(add_half(10.0))


# print(str(add_half.type_signatu

@tff.federated_computation(tff.FederatedType(np.float32,tff.CLIENTS))
def add_half_on_clients(x):
    return tff.federated_map(add_half, x)

# print(str(add_half_on_clients.type_signature))
result = add_half_on_clients([1.0, 3.0, 2.0])
print(result)