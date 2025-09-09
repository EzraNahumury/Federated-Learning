import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
federated_float_on_clients = tff.FederatedType(np.float32, tff.CLIENTS)
# print(str(federated_float_on_clients.member)) #T
# print(str(federated_float_on_clients.placement)) #G
# print(str(federated_float_on_clients))

print(str(federated_float_on_clients.all_equal))



