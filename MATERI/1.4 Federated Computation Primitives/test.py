import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

@tff.federated_computation
def hello_world():
  return 'Hello, World!'

print(hello_world())