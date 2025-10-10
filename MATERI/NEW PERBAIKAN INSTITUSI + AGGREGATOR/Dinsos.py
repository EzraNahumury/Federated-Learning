import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

# =======================================================
# KONFIGURASI
# =======================================================
INSTANSI = "dinsos"  # ubah ke: "dinsos" / "dukcapil" / "kemenkes"
DATA_PATH = f"BALANCED_DATASET_100K_REALISTIC/{INSTANSI}_balanced.csv"
SAVE_DIR  = Path(f"Models/saved_{INSTANSI}_tff")
BATCH_SIZE = 32
N_CLIENTS  = 10
ROUNDS     = 15

# =======================================================
# LOAD DATA
# =======================================================
print(f"📂 Membaca dataset {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"✅ {len(df):,} baris, {len(df.columns)} kolom.")

if "layak_subsidi" not in df.columns:
    raise ValueError("Kolom 'layak_subsidi' tidak ditemukan!")

y_all = df["layak_subsidi"].astype(np.int32).values
X_raw = df.drop(columns=["layak_subsidi"])

# =======================================================
# MUAT FITUR GLOBAL & ALIGN ONE-HOT
# =======================================================
GLOBAL_FEATS_PATH = Path("Models/fitur_global.pkl")
if not GLOBAL_FEATS_PATH.exists():
    raise FileNotFoundError(
        "Models/fitur_global.pkl tidak ditemukan. "
        "Jalankan dulu pembuatnya:\n"
        "  python fitur_global.py\n"
        "agar semua instansi memakai ruang fitur yang IDENTIK."
    )

GLOBAL_FEATURES = joblib.load(GLOBAL_FEATS_PATH)

# One-hot di data lokal
X_oh = pd.get_dummies(X_raw, drop_first=False).astype(float)

# Tambah kolom yang belum ada (isi 0), dan buang kolom ekstra
for col in GLOBAL_FEATURES:
    if col not in X_oh.columns:
        X_oh[col] = 0.0
X_oh = X_oh[GLOBAL_FEATURES]  # pastikan URUTAN identik

FEATURE_COLS = GLOBAL_FEATURES  # <- kunci agar input shape sama di semua model

# =======================================================
# NORMALISASI MIN-MAX (per instansi, mengikuti kolom global)
# =======================================================
mins = X_oh.min()
rng  = (X_oh.max() - mins).replace(0, 1.0)
X_scaled = ((X_oh - mins) / rng).fillna(0.0).astype("float32")

# =======================================================
# BUAT CLIENT DATASET
# =======================================================
def to_tf_dataset(X_df, y, batch=BATCH_SIZE):
    feats  = X_df.values.astype("float32")
    labels = y.astype("float32").reshape(-1, 1)
    return tf.data.Dataset.from_tensor_slices((feats, labels)).shuffle(len(X_df)).batch(batch)

def split_clients(X_df, y, n_clients=N_CLIENTS):
    idx = np.arange(len(X_df)); np.random.shuffle(idx)
    size = len(X_df) // n_clients
    clients = []
    for i in range(n_clients):
        s, e = i*size, (i+1)*size if i < n_clients-1 else len(X_df)
        clients.append(to_tf_dataset(X_df.iloc[idx[s:e]], y[idx[s:e]]))
    return clients

clients = split_clients(X_scaled, y_all, N_CLIENTS)
print(f"👥 {len(clients)} klien federated data siap digunakan.")

# =======================================================
# DEFINISI MODEL (input mengikuti FEATURE_COLS global)
# =======================================================
def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
    )

# =======================================================
# FEDERATED TRAINING
# =======================================================
process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.005),
    server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.01),
)

state = process.initialize()
print("\n🚀 Mulai Federated Training ===========================")
for r in range(1, ROUNDS + 1):
    state, metrics = process.next(state, clients)
    acc  = metrics['client_work']['train']['binary_accuracy']
    loss = metrics['client_work']['train']['loss']
    print(f"[{INSTANSI.upper()}] Round {r:02d} | acc={acc:.4f} | loss={loss:.4f}")

# =======================================================
# SIMPAN MODEL + PREPROCES
# =======================================================
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
process.get_model_weights(state).assign_weights_to(keras_model)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
keras_model.save(SAVE_DIR)
joblib.dump(
    {"FEATURE_COLS": FEATURE_COLS, "mins": mins, "rng": rng},
    SAVE_DIR / f"preprocess_{INSTANSI}.pkl"
)

# =======================================================
# EVALUASI RINGAN (TRAIN SET)
# =======================================================
preds = (keras_model.predict(X_scaled.values, verbose=0) > 0.5).astype(int).flatten()
acc_eval = np.mean(preds == y_all)
print(f"\n📊 Akurasi terhadap data training: {acc_eval:.4f}")

print("\n========== INFO BOBOT ==========")
print("Total parameter (weight + bias):", keras_model.count_params())
for layer in keras_model.layers:
    w = layer.get_weights()
    if w:
        shapes = [p.shape for p in w]
        counts = [int(np.prod(s)) for s in shapes]
        print(f"Layer: {layer.name}")
        print("  Shapes :", shapes)
        print("  Params :", counts, " -> total:", sum(counts))
print(f"✅ Model {INSTANSI.upper()} tersimpan di: {SAVE_DIR}")
