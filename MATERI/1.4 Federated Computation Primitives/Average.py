import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# @tff.federated_computation(tff.FederatedType(np.float32, tff.CLIENTS))
# def get_average_temperature(sensor_readings):
#     return tff.federated_mean(sensor_readings)

# print(str(get_average_temperature([68.5, 70.3, 69.8])))

@tff.federated_computation(tff.FederatedType(np.float32, tff.CLIENTS))
def get_average_temperature(sensor_readings):
    print('Getting traced, the argument is "{}".'.format(
            type(sensor_readings).__name__
    ))

    return tff.federated_mean(sensor_readings)

print(str(get_average_temperature))