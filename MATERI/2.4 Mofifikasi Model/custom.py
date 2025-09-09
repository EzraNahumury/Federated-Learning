
import os
import functools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# --------------- Hyperparameters & constants ---------------
ONLY_DIGITS = True               # EMNIST digits => 10 classes
MAX_CLIENT_DATASET_SIZE = 418    # EMNIST-specific shuffle size
CLIENT_EPOCHS_PER_ROUND = 1
CLIENT_BATCH_SIZE = 20
NUM_ROUNDS = 10                  # adjust as needed
NUM_CLIENTS_PER_ROUND = 10
BASE_LOGDIR = "/tmp/logs/scalars"

# --------------- Load EMNIST (federated) -------------------
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
    only_digits=ONLY_DIGITS
)

def reshape_emnist_element(element):
    # (28,28) -> (28,28,1), return (x, y)
    return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

def preprocess_train_dataset(dataset):
    return (dataset
            .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)
            .repeat(CLIENT_EPOCHS_PER_ROUND)
            .batch(CLIENT_BATCH_SIZE, drop_remainder=False)
            .map(reshape_emnist_element))

# Apply preprocessing for training set
emnist_train = emnist_train.preprocess(preprocess_train_dataset)

# Input spec (AFTER preprocessing) for TFF model wrapping
input_spec = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]
).element_spec

# --------------- Keras model definitions -------------------
def create_original_fedavg_cnn_model(only_digits: bool = True):
    """Model from McMahan et al. (2016) used widely in TFF examples."""
    data_format = 'channels_last'
    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D, pool_size=(2, 2), padding='same', data_format=data_format
    )
    conv2d = functools.partial(
        tf.keras.layers.Conv2D, kernel_size=5, padding='same', data_format=data_format, activation=tf.nn.relu
    )
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
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

# --------------- Wrap Keras -> TFF model -------------------
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

# --------------- Build FedAvg process ----------------------
def build_fedavg_process(keras_model_fn, model_aggregator=None):
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=make_tff_model_fn(keras_model_fn),
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.02),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0),
        model_aggregator=model_aggregator
    )

# --------------- Training loop with TB logging -------------
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

            # Console output
            print(f"[{label}] Round {round_num:02d} | acc={acc:.4f} | loss={loss:.4f}")

            # TensorBoard scalars
            tf.summary.scalar(f"{label}/train/accuracy", acc, step=round_num)
            tf.summary.scalar(f"{label}/train/loss", loss, step=round_num)
            writer.flush()
    return state

# --------------- Utilities -------------------------------
def safe_rmtree(path):
    try:
        tf.io.gfile.rmtree(path)
    except tf.errors.OpError:
        pass

def print_param_counts():
    m1 = create_original_fedavg_cnn_model(ONLY_DIGITS)
    m2 = create_simple_cnn_model()
    print(f"[params] original_fedavg_cnn : {m1.count_params():,}")
    print(f"[params] simple_cnn         : {m2.count_params():,}")

# --------------- Run Experiments --------------------------
if __name__ == "__main__":
    # Clean base logdir to avoid mixing runs
    safe_rmtree(BASE_LOGDIR)
    os.makedirs(BASE_LOGDIR, exist_ok=True)

    print_param_counts()

    # A) Original FedAvg CNN
    proc_original = build_fedavg_process(create_original_fedavg_cnn_model)
    writer_original = tf.summary.create_file_writer(f"{BASE_LOGDIR}/original")
    _ = train(
        proc_original,
        num_rounds=NUM_ROUNDS,
        num_clients_per_round=NUM_CLIENTS_PER_ROUND,
        writer=writer_original,
        label="original"
    )

    # B) Simple CNN (custom sederhana)
    proc_simple = build_fedavg_process(create_simple_cnn_model)
    writer_simple = tf.summary.create_file_writer(f"{BASE_LOGDIR}/simple")
    _ = train(
        proc_simple,
        num_rounds=NUM_ROUNDS,
        num_clients_per_round=NUM_CLIENTS_PER_ROUND,
        writer=writer_simple,
        label="simple"
    )

    # C) (Optional) Simple CNN + Compression Aggregator (skip if unavailable)
    try:
        compression_aggregator = tff.learning.compression_aggregator()
        # Quick sanity check:
        print("[info] compression_aggregator built:",
              isinstance(compression_aggregator, tff.aggregators.WeightedAggregationFactory))
        proc_simple_comp = build_fedavg_process(
            create_simple_cnn_model,
            model_aggregator=compression_aggregator
        )
        writer_simple_comp = tf.summary.create_file_writer(f"{BASE_LOGDIR}/simple_compression")
        _ = train(
            proc_simple_comp,
            num_rounds=NUM_ROUNDS,
            num_clients_per_round=NUM_CLIENTS_PER_ROUND,
            writer=writer_simple_comp,
            label="simple_compression"
        )
    except Exception as e:
        print(f"[warn] Compression aggregator unavailable, skipping. Reason: {e}")

    print("\nDone.")
    print("Launch TensorBoard, e.g.:")
    print(f"  tensorboard --logdir {BASE_LOGDIR} --port 6006")
