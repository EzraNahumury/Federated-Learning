import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


@tff.tensorflow.computation(tff.SequenceType(np.float32))
def get_local_temperature_average(local_temperatures):
    sum_and_count = local_temperatures.reduce(
        (0.0, 0),
        lambda x, y: (x[0] + y, x[1] + 1)
    )
    return sum_and_count[0] / tf.cast(sum_and_count[1], np.float32)

@tff.federated_computation(
    tff.FederatedType(tff.SequenceType(np.float32), tff.CLIENTS))
def get_global_temperature_average(sensor_readings):
    return tff.federated_mean(
        tff.federated_map(get_local_temperature_average, sensor_readings)
    )

# print(str(get_global_temperature_average.type_signature))

hasil = get_global_temperature_average([[68.0, 70.0], [71.0], [68.0, 72.0, 70.0]])
print(hasil)