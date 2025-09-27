# gabungan.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

# ================== CONFIG ==================
DATA_IN   = Path("NEW_DATASET") / "gabungan_labeled.csv"
SAVE_DIR  = Path("saved_model_gabungan")
PKL_PATH  = Path("preprocess_gabungan.pkl")

BATCH_SIZE = 32
ROUNDS     = 5
N_CLIENTS  = 5

print("TensorFlow:", tf.__version__)
print("TFF:", tff.__version__)

# ================== LOAD DATA ==================
if not DATA_IN.exists():
    raise FileNotFoundError(f"{DATA_IN} tidak ditemukan, jalankan labeled_gabungan.py dulu")

df = pd.read_csv(DATA_IN)

# Pastikan kolom label ada
if "label_gabungan" not in df.columns:
    raise ValueError("Kolom 'label_gabungan' tidak ditemukan di gabungan_labeled.csv")

y_all = df["label_gabungan"].astype(np.int32).values

# ================== FEATURE ENGINEERING ==================
# Tambahkan BMI sebagai fitur
df_feat = df.copy()
bmi = (df_feat["berat_kg"] / ((df_feat["tinggi_cm"]/100.0) ** 2)).replace([np.inf, -np.inf], np.nan)
df_feat["bmi"] = bmi.fillna(0.0)

num_cols = ["jumlah_tanggungan","penghasilan","umur","tinggi_cm","berat_kg","bmi"]
cat_cols = ["kondisi_rumah","status_pekerjaan","status_pernikahan","riwayat_penyakit","status_gizi"]

oh_df = pd.get_dummies(
    df_feat[cat_cols].fillna("unknown").astype(str).apply(lambda s: s.str.lower()),
    drop_first=False
).astype(float)

X_raw = pd.concat([df_feat[num_cols], oh_df], axis=1)

mins = X_raw.min()
rng  = (X_raw.max() - mins).replace(0, 1.0)
X_scaled = ((X_raw - mins) / rng).fillna(0.0).astype("float32")
FEATURE_COLS = list(X_scaled.columns)

# ================== DATASET TFF ==================
def to_tf_dataset(X: pd.DataFrame, y: np.ndarray, batch=BATCH_SIZE):
    feats = X.values.astype("float32")
    labels = y.astype("float32").reshape(-1,1)
    return tf.data.Dataset.from_tensor_slices((feats, labels)).shuffle(len(X)).batch(batch)

def split_clients(X: pd.DataFrame, y: np.ndarray, n_clients=N_CLIENTS):
    size = max(1, len(X) // n_clients)
    clients = []
    for i in range(n_clients):
        s = i*size
        e = (i+1)*size if i < n_clients-1 else len(X)
        if s >= len(X): break
        clients.append(to_tf_dataset(X.iloc[s:e], y[s:e]))
    return clients

clients = split_clients(X_scaled, y_all, N_CLIENTS)
if len(clients) == 0:
    raise RuntimeError("Tidak ada client data. Pastikan CSV berisi data.")

# ================== MODEL FN ==================
def model_fn():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(1,   activation="sigmoid"),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=m,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
    )

# ================== FEDERATED TRAINING ==================
process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(0.05, 0.9),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(1.0, 0.9),
)
state = process.initialize()

for r in range(1, ROUNDS+1):
    state, metrics = process.next(state, clients)
    acc  = metrics["client_work"]["train"]["binary_accuracy"]
    loss = metrics["client_work"]["train"]["loss"]
    print(f"[Gabungan] Round {r:02d} acc={acc:.4f} loss={loss:.4f}")

# ================== SAVE MODEL ==================
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64,  activation="relu"),
    tf.keras.layers.Dense(1,   activation="sigmoid"),
])
process.get_model_weights(state).assign_weights_to(keras_model)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
keras_model.save(SAVE_DIR)

joblib.dump({"FEATURE_COLS": FEATURE_COLS, "mins": mins, "rng": rng}, PKL_PATH)

# ================== QUICK EVAL ==================
preds = (keras_model.predict(X_scaled.values, verbose=0) > 0.5).astype(int).flatten()
print("Akurasi terhadap label_gabungan:", float(np.mean(preds == y_all)))
