import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load & Label
df = pd.read_csv("data_dummy_100.csv")
df['layak_subsidi'] = (df["Pendapatan"] > 6000000).astype(int)

# 2. Ambil kolom numerik
df_num = df[["Umur", "Pendapatan", "layak_subsidi"]]

# Normalisasi fitur (Umur & Pendapatan)
df_num["Umur"] = (df_num["Umur"] - df_num["Umur"].mean()) / df_num["Umur"].std()
df_num["Pendapatan"] = (df_num["Pendapatan"] - df_num["Pendapatan"].mean()) / df_num["Pendapatan"].std()

x = df_num.drop("layak_subsidi", axis=1)
y = df_num["layak_subsidi"]

# 3. Dataset builder
def make_tf_dataset(x, y, batch_size=8):
    feats = tf.cast(x.values, tf.float32)
    labels = tf.cast(y.values, tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((feats, labels))
    ds = ds.shuffle(buffer_size=len(x)).batch(batch_size)
    return ds

clients = []
num_clients = 3
split_df = np.array_split(df_num, num_clients)
for client_df in split_df:
    Xc = client_df[["Umur", "Pendapatan"]]
    Yc = client_df["layak_subsidi"]
    clients.append(make_tf_dataset(Xc, Yc))

# 4. Model
def create_keras_model():
    model = keras.Sequential([
        layers.Input(shape=(2,), dtype=tf.float32, name="features"),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 5. Bungkus ke TFF
input_spec = clients[0].element_spec

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# 6. Federated Training
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.05),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
)

state = iterative_process.initialize()

for round_num in range(1, 11):
    state, metrics = iterative_process.next(state, clients)
    print(f"Round {round_num}, metrics={metrics}")



