import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# Preparing federated data sets
mnist_train , mnist_test = tf.keras.datasets.mnist.load_data()
[(x.dtype, x.shape) for x in mnist_train]

NUM_EXAMPLE_PER_USER = 1000
BATCH_SIZE = 100

def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLE_PER_USER), BATCH_SIZE):
        batch_sample = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x' :
                np.array([source[0][i].flatten() / 255.0 for i in batch_sample], dtype=np.float32),
            'y' :
                np.array([source[1][i] for i in batch_sample], dtype=np.int32)
        })
    return output_sequence

federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

# print(federated_train_data[5][-1]['y'])


from matplotlib import pyplot as plt
plt.imshow(federated_train_data[5][-1]['x'][-1].reshape(28,28), cmap='gray')
plt.grid(False)
# plt.savefig('hasil.jpg')


#Defining a loss function
BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None,784], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.int32)
)

BATCH_TYPE = tff.types.StructType([
    ('x', tff.types.TensorType(np.float32, [None, 784])),
    ('y', tff.types.TensorType(np.int32, [None]))
])

# print(str(BATCH_TYPE))


MODEL_SPEC = collections.OrderedDict(
     weights=tf.TensorSpec(shape=[784,10], dtype=tf.float32),
     bias=tf.TensorSpec(shape=[10], dtype=tf.float32)
)

MODEL_TYPE = tff.types.StructType([
    ('weights', tff.types.TensorType(np.float32, [784,10])),
    ('bias', tff.types.TensorType(np.float32, [10]))
])
# print(MODEL_TYPE)



# note: `forward_pass` is defined separately from `batch_loss` so that it can
# be later called from within another tf.function. Necessary because a
# @tf.function  decorated method cannot invoke a @tff.tensorflow.computation.

@tf.function
def forward_pass(model, batch):
    predicted_y = tf.nn.softmax(
        tf.matmul(batch['x'], model['weights']) + model['bias']
    )

    return -tf.reduce_mean(
        tf.reduce_sum(
            tf.one_hot(batch['y'],10) * tf.math.log(predicted_y), axis=[1]
        )
    )

@tff.tensorflow.computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)

# print(str(batch_loss.type_signature))


initial_model = collections.OrderedDict(
    weights=np.zeros([784,10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32)
)
sample_batch = federated_train_data[5][-1]
# print(batch_loss(initial_model,sample_batch))



#Gradien descent on a single batch
from tensorflow_federated.python.common_libs import structure

@tff.tensorflow.computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    model_vars = collections.OrderedDict([
        (name, tf.Variable(name=name, initial_value=value))
        for name, value in structure.to_elements(initial_model)
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars))
        )
        return model_vars

    return _train_on_batch(model_vars, batch)

# print(str(batch_train.type_signature))


model = initial_model
losses = []
for _ in range(5):
    model = batch_train(model, sample_batch, 0.1)
    losses.append(batch_loss(model, sample_batch))

# print(losses)


#Gradient descent on a sequence of local data
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

@tff.federated_computation(MODEL_TYPE, np.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
  
  # Reduction function to apply to each batch.
  @tff.federated_computation((MODEL_TYPE, np.float32), BATCH_TYPE)
  def batch_fn(model_with_lr, batch):
    model, lr = model_with_lr
    return batch_train(model, batch, lr), lr

  trained_model, _ = tff.sequence_reduce(
      all_batches, (initial_model, learning_rate), batch_fn
  )
  return trained_model
# print(str(local_train.type_signature))

locally_trained_model = local_train(initial_model, 0.1, federated_train_data[5])
# print(locally_trained_model)




#Local EValuation
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):

    @tff.tensorflow.computation((MODEL_TYPE, np.float32), BATCH_TYPE)
    def accumulate_evaluation(model_and_accumulator, batch):
        model, accumulator = model_and_accumulator
        return model, accumulator + batch_loss(model, batch)
    
    _, total_loss=tff.sequence_reduce(
        all_batches, (model,0.0), accumulate_evaluation
    )
    return total_loss
# print(str(local_eval.type_signature))\


# print("Initial_model loss = ", local_eval(initial_model,federated_train_data[5]))
# print("Locally_trained_model loss = ", local_eval(locally_trained_model, federated_train_data[5]))

# print("Initial_model loss = ", local_eval(initial_model,federated_train_data[0]))
# print("Locally_trained_model loss = ", local_eval(locally_trained_model, federated_train_data[0]))



#Federated Evaluation
SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)

@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data])
    )


# print("Initial_model loss = ", federated_eval(initial_model,federated_train_data))
# print("Locally_trained_model loss = ", federated_eval(locally_trained_model, federated_train_data))


# Federateed Training
SERVER_FLOAT_TYPE = tff.FederatedType(np.float32, tff.SERVER)

@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    return tff.federated_mean(
        tff.federated_map(local_train, [
            tff.federated_broadcast(model),
            tff.federated_broadcast(learning_rate), data
        ])
    )

model = initial_model
learning_rate = 0.01
for round in range(5):
    model = federated_train(model, learning_rate, federated_train_data)
    learning_rate = learning_rate * 0.9
    loss = federated_eval(model, federated_train_data)
    # print('round {}, loss={}'.format(round, loss))


print(
    'initial_model test loss =',
    federated_eval(initial_model, federated_test_data),
)
print('trained_model test loss =', federated_eval(model, federated_test_data))