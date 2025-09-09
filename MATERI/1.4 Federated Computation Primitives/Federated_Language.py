import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


# federated_float_on_clients_all_equal = tff.FederatedType(np.float32, tff.CLIENTS, all_equal=True)
# print("All Equal:", federated_float_on_clients_all_equal)

# simple_regression_model_type = (
#     tff.StructType([('a', np.float32), ('b', np.float32)])
# )
# print(str(simple_regression_model_type))

simple_regression_model_type = (
    tff.StructType([('a', np.float32), ('b', np.float32)])
)
print(str(tff.FederatedType(
    simple_regression_model_type, 
    placement=tff.CLIENTS,
    all_equal=True
)))
