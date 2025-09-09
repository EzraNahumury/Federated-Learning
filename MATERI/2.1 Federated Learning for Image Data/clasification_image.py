import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff



#============================================== 1. Preparing The Input Data ==============================================
from matplotlib import pyplot as plt

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# print(len(emnist_train.client_ids))
# print(emnist_train.element_type_structure)

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]
)
example_element = next(iter(example_dataset))
# print("label : ", example_element['label'].numpy())


plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
# plt.savefig("sample.png")






# ============================================== 2. Exploring heterogeneity in federated data ==============================================

#Example MNIST digits for one client
figure = plt.figure(figsize=(20,4))
j = 0

for example in example_dataset.take(40):
    plt.subplot(4, 10, j+1)
    plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
    plt.axis('off')
    j += 1
    # plt.savefig("Example.png")



#Number of examples per layer for a sample of clients
f = plt.figure(figsize=(12,7))
f.suptitle('Label Counts for a sample of clients')
for i in range(6):
    client_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[1]
    )
    plot_data = collections.defaultdict(list)
    for example in client_dataset:
        #Append counts individually per label to make plots
        #more colorful instead of one color per plot
        label = example['label'].numpy()
        plot_data['label'].append(label)
    plt.subplot(2,3, i+1)
    plt.title('Client {}'.format(i))
    for j in range(10):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[0,1,2,3,4,5,6,7,8,9,10]
            )
# plt.savefig("Client.png")


# Each client has different mean images, meaning each client  will be nudging
# the model in their own directions locally

for i in range (5):
    client_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[i]
    )
    plot_data = collections.defaultdict(list)
    for example in client_dataset :
        plot_data[example['label'].numpy()].append(example['pixels'].numpy())
    f = plt.figure(figsize=(12, 5))
    f.suptitle(f"Client #{i}'s Mean Image Per label")
    for j in range(10):
        mean_img = np.mean(plot_data[j],0)
        plt.subplot(2,5, j+1)
        plt.imshow(mean_img.reshape((28,28)))
        plt.axis('off')
    # plt.savefig(f"Client_{i}.png")
    plt.close() 



# ============================================== 3. Preporcessing the input data  ==============================================

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(
            x = tf.reshape(element['pixels'], [-1, 784]),
            y = tf.reshape(element['label'], [-1,1])
        )
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))
# print(sample_batch)


def make_federated_data(client_data, client_ids) :
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
# print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
# print('First dataset: {d}'.format(d=federated_train_data[0]))




# ============================================== 4. Creating a model with Keras ==============================================

def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])



def model_fn():

     # We _must_ create a new model here, and _not_ capture it from an external
     # scope. TFF will call this within different graph contexts.
     keras_model = create_keras_model()
     return tff.learning.models.from_keras_model(
        keras_model,
        input_spec = preprocessed_example_dataset.element_spec,
        loss  = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
     )





# ============================================== 5. Training the model on federated data ==============================================
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0))

# print(training_process.initialize.type_signature)

train_state = training_process.initialize()

result = training_process.next(train_state, federated_train_data)
train_state = result.state
train_metrics = result.metrics
# print('round  1, metrics={}'.format(train_metrics))


NUM_ROUNDS = 11
for round_num in range(2, NUM_ROUNDS):
  result = training_process.next(train_state, federated_train_data)
  train_state = result.state
  train_metrics = result.metrics
#   print('round {:2d}, metrics={}'.format(round_num, train_metrics))




# ============================================== 6. Displaying model metrics in TensorBoard  ==============================================
logdir = "/tmp/logs/scalars/training"
summary_writer = tf.summary.create_file_writer(logdir)
train_state = training_process.initialize()

with summary_writer.as_default():
    for round_num in range(1, NUM_ROUNDS):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics
        for name, value in train_metrics['client_work']['train'].items():
            tf.summary.scalar(name, value, step=round_num)





# ============================================== 7. Customizing the model Implementation  ==============================================

#Defining model Variables, Forward pass, and metrics
MnistVariables = collections.namedtuple(
    "MnistVariables", 'weights bias num_examples loss_sum accuracy_sum'
)


def create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True,
      ),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True,
      ),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False),
  )





def predict_on_batch(variables, x) :
    return tf.nn.softmax(tf.matmul(x, variables.weights) + variables.bias)

def mnist_forward_pass(variables, batch) : 
    y = predict_on_batch(variables, batch['x'])
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1])
    )
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32)
    )

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)
    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions




def get_metric_finalizers():
  return collections.OrderedDict(
      num_examples=tf.function(func=lambda x: x[0]),
      loss=tf.function(func=lambda x: x[0] / x[1]),
      accuracy=tf.function(func=lambda x: x[0] / x[1]),
  )

def get_local_unfinalized_metrics(variables):
  return collections.OrderedDict(
      num_examples=[variables.num_examples],
      loss=[variables.loss_sum, variables.num_examples],
      accuracy=[variables.accuracy_sum, variables.num_examples],
  )




# Constructing an instance of tff.learning.model
import collections
from collections.abc import Callable


class MnistModel(tff.learning.models.VariableModel):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples,
        self._variables.loss_sum,
        self._variables.accuracy_sum,
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32),
    )

  @tf.function
  def predict_on_batch(self, x, training=True):
    del training
    return predict_on_batch(self._variables, x)

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    num_exmaples = tf.shape(batch['x'])[0]
    return tff.learning.models.BatchOutput(
        loss=loss, predictions=predictions, num_examples=num_exmaples
    )

  @tf.function
  def report_local_unfinalized_metrics(
      self,
  ) -> collections.OrderedDict[str, list[tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to unfinalized values."""
    return get_local_unfinalized_metrics(self._variables)

  def metric_finalizers(
      self,
  ) -> collections.OrderedDict[str, Callable[[list[tf.Tensor]], tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to finalizers."""
    return get_metric_finalizers()

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    for var in self.local_variables:
      var.assign(tf.zeros_like(var))



#Simulating federating training with the new model
training_process = tff.learning.algorithms.build_weighted_fed_avg(
   MnistModel,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0))


train_state = training_process.initialize()
result = training_process.next(train_state, federated_train_data)
train_state = result.state
metrics = result.metrics
# print('round  1, metrics={}'.format(metrics))\


for round_num in range(2,11):
   result = training_process.next(train_state, federated_train_data)
   train_state = result.state
   metrics = result.metrics
#    print('round {:2d}, metrics={}'.format(round_num, metrics))


# ============================================== 8. Evaluation ==============================================
evalution_process = tff.learning.algorithms.build_fed_eval(MnistModel)
# print(evalution_process.next.type_signature.formatted_representation())

evalution_state = evalution_process.initialize()
model_weights = training_process.get_model_weights(train_state)
evalution_state = evalution_process.set_model_weights(evalution_state, model_weights)


evaluation_output = evalution_process.next(
   evalution_state, federated_train_data
)

# print(str(evaluation_output.metrics))\

federated_test_data = make_federated_data(emnist_test, sample_clients)
# print(len(federated_test_data), federated_test_data[0])

evaluation_output = evalution_process.next(evalution_state, federated_test_data)
print(str(evaluation_output.metrics))
