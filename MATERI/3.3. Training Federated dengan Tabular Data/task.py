
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras
from tensorflow.keras import layers

pd.options.mode.copy_on_write = True  

TARGET = "layak_subsidi"

# =========================
# 1) Load CSV per klien
# =========================
clients_df = {
    "dinsos":   pd.read_csv("data/dinsos_1500.csv"),
    "dukcapil": pd.read_csv("data/dukcapil_1500.csv"),
    "kemenkes": pd.read_csv("data/kemenkes_1500.csv"),
}

# =========================
# 2) Bangun UNION kolom numerik lintas klien
#    (konsistensi dimensi fitur untuk semua klien)
# =========================
def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]

# union kolom numerik
num_union = sorted(set().union(*[set(numeric_cols(df)) for df in clients_df.values()]))



# =========================
# 3) Preprocess: tambahkan kolom yang hilang, urutkan sesuai union, normalisasi numerik
# =========================
def preprocess_with_union(df: pd.DataFrame, union_cols: list[str]):
    df2 = df.copy()

    # pastikan semua kolom union ada (yg tidak ada -> diisi 0)
    for col in union_cols:
        if col not in df2.columns:
            df2[col] = 0.0

    # ambil fitur numerik sesuai union + label
    df_num = df2[union_cols + [TARGET]].copy()

    # normalisasi per kolom; hindari std=0
    for c in union_cols:
        col = df_num[c].astype("float32")
        mean = float(col.mean())
        std = float(col.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        df_num[c] = (col - mean) / std

    x = df_num[union_cols].astype("float32")
    y = df_num[TARGET].astype("float32")
    return x, y

def make_tf_dataset(x, y, batch_size=16):
    feats = tf.convert_to_tensor(x.values, dtype=tf.float32)       # shape: (N, n_features)
    labels = tf.convert_to_tensor(y.values, dtype=tf.float32)      # shape: (N,)
    ds = tf.data.Dataset.from_tensor_slices((feats, labels))
    return ds.shuffle(buffer_size=len(x)).batch(batch_size)

# buat federated dataset (1 klien = 1 CSV)
federated_train_data = []
for name, df in clients_df.items():
    Xc, Yc = preprocess_with_union(df, num_union)
    federated_train_data.append(make_tf_dataset(Xc, Yc))

# Debug kecil: pastikan element_spec sama
# print([ds.element_spec for ds in federated_train_data])

# =========================
# 4) Model Keras: input shape = len(num_union)
# =========================
n_features = len(num_union)

def create_keras_model():
    model = keras.Sequential([
        layers.Input(shape=(n_features,), dtype=tf.float32, name="features"),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Ambil element_spec dari salah satu klien (semua sama sekarang)
input_spec = federated_train_data[0].element_spec

# =========================
# 5) Bungkus ke TFF
# =========================
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# =========================
# 6) Federated Training (FedAvg) + logging akurasi
# =========================
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.01),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
)

state = trainer.initialize()

for round_num in range(1, 11):  
    result = trainer.next(state, federated_train_data)
    state = result.state
    metrics = result.metrics
    acc = float(metrics["client_work"]["train"]["binary_accuracy"])
    loss = float(metrics["client_work"]["train"]["loss"])
    print(f"Round {round_num} -> acc={acc:.4f}, loss={loss:.4f}")
