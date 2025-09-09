# task.py — Federated Learning (FedAvg) 3 klien dengan numerik + kategorikal (one-hot)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import keras
from tensorflow.keras import layers
import re

pd.options.mode.copy_on_write = True
TARGET = "layak_subsidi"

# =========================
# 1. Load CSV per klien
# =========================
clients_df = {
    "dinsos": pd.read_csv("data/dinsos_1500.csv"),
    "dukcapil": pd.read_csv("data/dukcapil_1500.csv"),
    "kemenkes": pd.read_csv("data/kemenkes_1500.csv"),
}

# =========================
# 2. Helper: sanitasi nama kolom
# =========================
def safe_col(name: str) -> str:
    s = str(name)
    s = s.replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    if not re.match(r"^[A-Za-z0-9]", s):
        s = "f_" + s
    return s

# =========================
# 3. Bangun UNION fitur one-hot lintas klien
# =========================
def build_union_columns(clients_df: dict) -> list:
    feats = []
    for df in clients_df.values():
        X = df.drop(columns=[TARGET])
        X = pd.get_dummies(X, drop_first=False)
        X.columns = [safe_col(c) for c in X.columns]
        feats.append(X)
    combo = pd.concat(feats, axis=0, ignore_index=True)
    return sorted(combo.columns)

UNION_COLS = build_union_columns(clients_df)

# =========================
# 4. Preprocess per klien
# =========================
def preprocess_with_union(df: pd.DataFrame, union_cols: list[str]):
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=False)
    X.columns = [safe_col(c) for c in X.columns]

    # reindex ke UNION, kolom yang tidak ada diisi 0
    X = X.reindex(columns=union_cols, fill_value=0.0)
    X = X.astype("float32")

    y = df[TARGET].astype("float32")
    return X, y

def make_tf_dataset(x, y, batch_size=32):
    feats = tf.convert_to_tensor(x.values, dtype=tf.float32)
    labels = tf.convert_to_tensor(y.values, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((feats, labels))
    return ds.shuffle(buffer_size=len(x)).batch(batch_size)

# Federated dataset (1 CSV = 1 client)
federated_train_data = []
for name, df in clients_df.items():
    Xc, Yc = preprocess_with_union(df, UNION_COLS)
    federated_train_data.append(make_tf_dataset(Xc, Yc))

# =========================
# 5. Model Keras
# =========================
n_features = len(UNION_COLS)

def create_keras_model():
    model = keras.Sequential([
        layers.Input(shape=(n_features,), dtype=tf.float32, name="features"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

input_spec = federated_train_data[0].element_spec

# =========================
# 6. Bungkus ke TFF
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
# 7. Federated Training (FedAvg) + Logging
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
