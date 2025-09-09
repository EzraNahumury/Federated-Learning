import functools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

#=============== Preparing The Input Data ===============

# This value only applies to EMNIST dataset, consider choosing appropriate
# values if switching to other datasets.
MAX_CLIENT_DATASET_SIZE = 418

CLIENT_EPOCH_PER_ROUND = 1
CLIENT_BATCH_SIZE = 20
TEST_BATCH_SIZE = 500

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
    only_digits=True
)

def reshape_emnist_element(element):
    return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

def preprocess_train_dataset(dataset):
    """Preprocessing function for the EMNIST training dataset."""
    return (dataset
            # Shuffle according to the largest client dataset
            .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)

            # Repeat to do multiple local epochs
            .repeat(CLIENT_EPOCH_PER_ROUND)

            # Batch to a fixed client batch size
            .batch(CLIENT_BATCH_SIZE, drop_remainder=False)

            # Preprocessing step
            .map(reshape_emnist_element)
            )
emnist_train = emnist_train.preprocess(preprocess_train_dataset)



#=============== Defini a model ===============
def create_original_fedavg_cnn_model(only_digits=True) :
     """The CNN model used in https://arxiv.org/abs/1602.05629."""
     data_format = 'channels_last'

     max_pool = functools.partial(
         tf.keras.layers.MaxPooling2D,
         pool_size=(2,2),
         padding='same',
         data_format=data_format)
     
     conv2d = functools.partial(
         tf.keras.layers.Conv2D,
         kernel_size = 5,
         padding='same',
         data_format=data_format,
         activation=tf.nn.relu)
     
     model = tf.keras.models.Sequential([
         tf.keras.layers.InputLayer(input_shape=(28,28,1)),
         conv2d(filters=32),
         max_pool(),
         conv2d(filters=64),
         max_pool(),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(512, activation=tf.nn.relu),
         tf.keras.layers.Dense(10 if only_digits else 62),
         tf.keras.layers.Softmax(),
     ])

     return model

# Gets the type information of the input data. TFF is a strongly typed
# functional programming framework, and needs type information about inputs to 
# the model.
input_spec = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]).element_spec

def tff_model_fn():
    keras_model = create_original_fedavg_cnn_model()
    return tff.learning.models.from_keras_model(
        keras_model = keras_model,
        input_spec = input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )




#=============== Training the model and outputting training metrics ===============
federated_averaging = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=tff_model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
)



def train(federated_averaging_process, num_rounds, num_clients_per_round, summary_wrtiter):
    """Trains the federated averaging process and output metrics."""

    #Initialize the Federated Averaging algorithm to get the initial server state.
    state = federated_averaging_process.initialize()

    with summary_wrtiter.as_default():
        for round_num in range(num_rounds):
            # Sample the clients parcitipated in this round.
            sampled_clients = np.random.choice(
                emnist_train.client_ids,
                size=num_clients_per_round,
                replace=False
            )

            # Create a list of `tf.Dataset` instances from the data of sampled clients.
            sampled_train_data = [
                emnist_train.create_tf_dataset_for_client(client)
                for client in sampled_clients
            ]

            # Round one round of the algorithm based on the server state and client data
            # and output the new state and metrics.
            result = federated_averaging_process.next(state, sampled_train_data)
            state = result.state
            train_metrics = result.metrics['client_work']['train']
            
            # === Print hasil setiap round ===
            # print(f"Round {round_num}, metrics={train_metrics}")

            # Add metrics to Tensorboard.
            for name, value in train_metrics.items():
                tf.summary.scalar(name,value, step=round_num)
            summary_wrtiter.flush()

# Clean the log directory to avoid conflicts.
try :
    tf.io.gfile.rmtree('/tmp/logs/scalar')
except tf.errors.OpError as e :
    pass # Path doesn't exist

# Set up the log directory and writer for Tensorboard.
logdir = "/tmp/logs/scalars/original/"
summary_writter = tf.summary.create_file_writer(logdir)

train(federated_averaging_process=federated_averaging, num_rounds=10,
      num_clients_per_round=10, summary_wrtiter=summary_writter)




#=============== Build a custom aggregation function ===============
compression_aggregator = tff.learning.compression_aggregator()
print(isinstance(compression_aggregator, tff.aggregators.WeightedAggregationFactory))

federated_averaging_with_compression = tff.learning.algorithms.build_weighted_fed_avg(
    tff_model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0),
    model_aggregator=compression_aggregator
)




#=============== Training the model again ===============]
logdir_for_compression = "/tmp/logs/scalars/compression/"
summary_writter_for_compression = tf.summary.create_file_writer(
    logdir_for_compression
)

train(federated_averaging_process=federated_averaging_with_compression,
      num_rounds=10,
      num_clients_per_round=10,
      summary_writter=summary_writter_for_compression)

