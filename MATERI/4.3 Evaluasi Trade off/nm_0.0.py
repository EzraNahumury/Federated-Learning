#================================= IMPORT ================================= 
import collections
import dp_accounting
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import seaborn as sns


#================================= Download and preprocess the federated EMNIST dataset. ================================= 
def get_emnist_dataset():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=True
    )

    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label']
        )

    def preprocess_train_dataset(dataset):
        # buffer_size = maksimum client dataset size (418 utk Federated EMNIST)
        return (dataset.map(element_fn)
                .shuffle(buffer_size=418)
                .repeat(1)
                .batch(32, drop_remainder=False))

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(128, drop_remainder=False)

    emnist_train = emnist_train.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        emnist_test.create_tf_dataset_from_all_clients()
    )
    return emnist_train, emnist_test


train_data, test_data = get_emnist_dataset()


#================================= Define our model. ================================= 
def my_model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=test_data.element_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


#================================= Helper: convert dict -> (x,y) =================================
def to_xy(dataset):
    return dataset.map(lambda elem: (elem['x'], elem['y']))


#================================= Manual evaluation function =================================
def evaluate(state, eval_data, learning_process):
    # Ambil bobot model federated
    model_weights = learning_process.get_model_weights(state)

    # Ambil model Keras dari wrapper TFF
    keras_model = my_model_fn()._keras_model
    model_weights.assign_weights_to(keras_model)

    # Compile ulang untuk evaluasi
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Konversi dataset ke (x,y) sebelum evaluasi
    eval_xy = to_xy(eval_data)

    # Evaluasi
    result = keras_model.evaluate(eval_xy, verbose=0, return_dict=True)
    return result


#================================= Training function =================================
total_clients = len(train_data.client_ids)

def train(rounds, noise_multiplier, clients_per_round, data_frame):
    aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
        noise_multiplier, clients_per_round
    )

    sampling_prob = clients_per_round / total_clients

    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        my_model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(0.01),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(
            1.0, momentum=0.9),
        model_aggregator=aggregation_factory
    )

    state = learning_process.initialize()
    for round in range(1, rounds + 1):
        # Sampling clients
        x = np.random.uniform(size=total_clients)
        sampled_clients = [
            train_data.client_ids[i] for i in range(total_clients)
            if x[i] < sampling_prob
        ]
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        result = learning_process.next(state, sampled_train_data)
        state = result.state

        # Evaluasi hanya setiap kelipatan 5
        if round % 5 == 0:
            metrics = evaluate(state, test_data, learning_process)
            print(f'Round {round:3d}: {metrics}')
            data_frame = pd.concat([
                data_frame,
                pd.DataFrame([{'Round': round,
                               'NoiseMultiplier': noise_multiplier,
                               **metrics}])
            ], ignore_index=True)

    return data_frame


#================================= Logging Akurasi ================================= 
data_frame = pd.DataFrame()
rounds = 40
clients_per_round = 40

# Hanya 1 noise multiplier: 0.0
noise_multiplier = 0.0
print(f'Starting training with noise multiplier: {noise_multiplier}')
data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)
print("Training selesai.")


#================================= Plot Accuracy & Loss dalam 1 Grafik ================================= 
plt.figure(figsize=(8, 5))

# Accuracy (sumbu kiri)
ax1 = sns.lineplot(
    data=data_frame,
    x="Round",
    y="sparse_categorical_accuracy",
    marker="o",
    color="blue",
    label="Accuracy"
)
ax1.set_ylabel("Accuracy", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

# Loss (sumbu kanan)
ax2 = ax1.twinx()
sns.lineplot(
    data=data_frame,
    x="Round",
    y="loss",
    marker="s",
    color="red",
    label="Loss",
    ax=ax2
)
ax2.set_ylabel("Loss", color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title("Federated Learning: Accuracy & Loss per Round")
plt.xlabel("Round")
plt.grid(True)

# Simpan grafik ke file
plt.tight_layout()
plt.savefig("nm_0,0.png")
print("Grafik disimpan sebagai fl_metrics.png")
