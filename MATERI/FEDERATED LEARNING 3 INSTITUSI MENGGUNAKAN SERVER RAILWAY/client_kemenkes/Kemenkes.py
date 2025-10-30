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
INSTANSI = "kemenkes"  
DATA_PATH = f"data/{INSTANSI}_balanced.csv"
SAVE_DIR  = Path(f"Models/saved_{INSTANSI}_tff")
BATCH_SIZE = 32
N_CLIENTS  = 10
ROUNDS     = 10  # cukup 10 karena 15 cenderung overfit

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
GLOBAL_FEATS_PATH = Path("Models/fitur_global_dict_baru.pkl")
if not GLOBAL_FEATS_PATH.exists():
    raise FileNotFoundError("❌ Models/fitur_global_dict.pkl tidak ditemukan!")

GLOBAL_FEATURES = joblib.load(GLOBAL_FEATS_PATH)
if isinstance(GLOBAL_FEATURES, dict):
    GLOBAL_FEATURES = GLOBAL_FEATURES.get("FEATURE_COLS", list(GLOBAL_FEATURES.keys()))

# One-hot
X_oh = pd.get_dummies(X_raw, drop_first=False).astype(float)
for col in GLOBAL_FEATURES:
    if col not in X_oh.columns:
        X_oh[col] = 0.0
X_oh = X_oh[GLOBAL_FEATURES]  # urutan kolom identik
FEATURE_COLS = GLOBAL_FEATURES

# =======================================================
# NORMALISASI MIN-MAX
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
    return tf.data.Dataset.from_tensor_slices((feats, labels))\
             .shuffle(len(X_df))\
             .batch(batch)

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
# DEFINISI MODEL (lebih ringan dan regularisasi lebih kuat)
# =======================================================
def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
    )

# =======================================================
# FEDERATED TRAINING (LR diturunkan agar lebih halus)
# =======================================================
process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.0001),
    server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.01),
)

state = process.initialize()
prev_acc = 0.0
print("\n🚀 Mulai Federated Training ===========================")

for r in range(1, ROUNDS + 1):
    state, metrics = process.next(state, clients)
    acc  = metrics['client_work']['train']['binary_accuracy']
    loss = metrics['client_work']['train']['loss']
    print(f"[{INSTANSI.upper()}] Round {r:02d} | acc={acc:.4f} | loss={loss:.4f}")

    # Early stop manual jika akurasi stagnan
    if abs(acc - prev_acc) < 1e-4 and r > 3:
        print(f"🛑 Early stop di round {r}, akurasi sudah stabil.")
        break
    prev_acc = acc

# =======================================================
# SIMPAN MODEL + PREPROCES
# =======================================================
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
    tf.keras.layers.Dense(64, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
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
print("Total parameter:", keras_model.count_params())
for layer in keras_model.layers:
    w = layer.get_weights()
    if w:
        print(f"Layer: {layer.name}, Params: {[p.shape for p in w]}")
print(f"✅ Model {INSTANSI.upper()} tersimpan di: {SAVE_DIR}")
