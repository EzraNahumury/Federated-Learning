import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# float32_sequence = tff.SequenceType(np.float32)
# print(str(float32_sequence))

# @tff.tensorflow.computation(tff.SequenceType(np.float32))
# def get_local_temperature_average(local_temperatures):
#     sum_and_count = (
#         local_temperatures.reduce((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1))
#     )
#     return sum_and_count[0] / tf.cast(sum_and_count[1], np.float32)

# print(str(get_local_temperature_average.type_signature))

# @tff.tensorflow.computation(tff.SequenceType(np.int32))
# def foo(x) :
#     return x.reduce(np.int32(0), lambda x, y: x + y)
# print(foo([1,2,3]))




# @tff.tensorflow.computation(tff.SequenceType(np.float32))
# def get_local_temperature_average(local_temperatures):
#     sum_and_count = (
#         local_temperatures.reduce((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1))
#     )
#     return sum_and_count[0] / tf.cast(sum_and_count[1], np.float32)

# print(str(get_local_temperature_average([68.5, 70.3, 69.8])))


@tff.tensorflow.computation(tff.SequenceType(collections.OrderedDict([('A', np.int32), ('B', np.int32)])))
def foo(ds):
    print('element_structure = {}'.format(ds.element_spec))
    return ds.reduce(np.int32(0),  lambda total, x:total + x['A'] * x['B'])

print(str(foo.type_signature))

hasil = foo([{'A':2, 'B':3}, {'A':4, 'B' :5}])
print(hasil)