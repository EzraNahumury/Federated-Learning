import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

@tff.federated_computation(tff.FederatedType(np.float32, tff.CLIENTS))
def get_average_temperature(sensor_readings):
    return tff.federated_mean(sensor_readings)

print(str(get_average_temperature.type_signature))