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

#tambahan
def create_simple_cnn_model():
    """A smaller CNN as the 'custom sederhana' baseline."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),           # digits: 10 classes
        tf.keras.layers.Softmax(),
    ])
    return model


# Gets the type information of the input data. TFF is a strongly typed
# functional programming framework, and needs type information about inputs to 
# the model.
input_spec = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]).element_spec

# def tff_model_fn():
#     keras_model = create_original_fedavg_cnn_model()
#     return tff.learning.models.from_keras_model(
#         keras_model = keras_model,
#         input_spec = input_spec,
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
#     )

#diganti dengan 
def make_tff_model_fn(keras_model_fn):
    def tff_model_fn():
        keras_model = keras_model_fn()
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    return tff_model_fn



#=============== Training the model and outputting training metrics ===============
# federated_averaging = tff.learning.algorithms.build_weighted_fed_avg(
#     model_fn=tff_model_fn,
#     client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
#     server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
# )

#diiganti dengan
# --------------- Build FedAvg process ----------------------
def build_fedavg_process(keras_model_fn, model_aggregator=None):
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=make_tff_model_fn(keras_model_fn),
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0),
        model_aggregator=model_aggregator
    )



# def train(federated_averaging_process, num_rounds, num_clients_per_round, summary_wrtiter):
#     """Trains the federated averaging process and output metrics."""

#     #Initialize the Federated Averaging algorithm to get the initial server state.
#     state = federated_averaging_process.initialize()

#     with summary_wrtiter.as_default():
#         for round_num in range(num_rounds):
#             # Sample the clients parcitipated in this round.
#             sampled_clients = np.random.choice(
#                 emnist_train.client_ids,
#                 size=num_clients_per_round,
#                 replace=False
#             )

#             # Create a list of `tf.Dataset` instances from the data of sampled clients.
#             sampled_train_data = [
#                 emnist_train.create_tf_dataset_for_client(client)
#                 for client in sampled_clients
#             ]

#             # Round one round of the algorithm based on the server state and client data
#             # and output the new state and metrics.
#             result = federated_averaging_process.next(state, sampled_train_data)
#             state = result.state
#             train_metrics = result.metrics['client_work']['train']
            
#             # === Print hasil setiap round ===
#             # print(f"Round {round_num}, metrics={train_metrics}")

#             # Add metrics to Tensorboard.
#             for name, value in train_metrics.items():
#                 tf.summary.scalar(name,value, step=round_num)
#             summary_wrtiter.flush()


#diganti dengan 
def train(fedavg_process, *, num_rounds, num_clients_per_round, writer, label):
    state = fedavg_process.initialize()
    client_ids = np.array(emnist_train.client_ids)

    with writer.as_default():
        for round_num in range(1, num_rounds + 1):
            sampled_clients = np.random.choice(
                client_ids, size=num_clients_per_round, replace=False
            )
            sampled_train_data = [
                emnist_train.create_tf_dataset_for_client(cid) for cid in sampled_clients
            ]

            result = fedavg_process.next(state, sampled_train_data)
            state = result.state
            mt = result.metrics['client_work']['train']

            acc = float(mt.get('sparse_categorical_accuracy', 0.0))
            loss = float(mt.get('loss', 0.0))

            #Tambahan 1: print ringkas setiap round
            print(f"[{label}] Round {round_num:02d} | acc={acc:.4f} | loss={loss:.4f}")

            #Tambahan 2: logging scalar dengan prefix label
            tf.summary.scalar(f"{label}/train/accuracy", acc, step=round_num)
            tf.summary.scalar(f"{label}/train/loss", loss, step=round_num)

            writer.flush()
    return state


# Clean the log directory to avoid conflicts.
try :
    tf.io.gfile.rmtree('/tmp/logs/scalar')
except tf.errors.OpError as e :
    pass # Path doesn't exist






#=============== Build a custom aggregation function ===============
compression_aggregator = tff.learning.compression_aggregator()
print(isinstance(compression_aggregator, tff.aggregators.WeightedAggregationFactory))




# =============== Jalankan Training untuk perbandingan ===============

# 1) Original FedAvg CNN
proc_original = build_fedavg_process(create_original_fedavg_cnn_model)
logdir_original = "/tmp/logs/scalars/original/"
writer_original = tf.summary.create_file_writer(logdir_original)

train(fedavg_process=proc_original,
      num_rounds=10,
      num_clients_per_round=10,
      writer=writer_original,
      label="original")

# 2) Custom Simple CNN
proc_simple = build_fedavg_process(create_simple_cnn_model)
logdir_simple = "/tmp/logs/scalars/simple/"
writer_simple = tf.summary.create_file_writer(logdir_simple)

train(fedavg_process=proc_simple,
      num_rounds=10,
      num_clients_per_round=10,
      writer=writer_simple,
      label="simple")

#===============  Compression Aggregator ===============
try:
    compression_aggregator = tff.learning.compression_aggregator()
    print("[info] compression_aggregator is WeightedAggregationFactory:",
          isinstance(compression_aggregator, tff.aggregators.WeightedAggregationFactory))

  
    proc_simple_comp = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=make_tff_model_fn(create_simple_cnn_model),   # FIXED
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0),
        model_aggregator=compression_aggregator
    )

    logdir_comp = "/tmp/logs/scalars/compression/"
    writer_comp = tf.summary.create_file_writer(logdir_comp)

    train(fedavg_process=proc_simple_comp,
          num_rounds=10,
          num_clients_per_round=10,
          writer=writer_comp,
          label="simple_compression")
except Exception as e:
    print("[warn] Compression aggregator unavailable, skipping. Reason:", e)