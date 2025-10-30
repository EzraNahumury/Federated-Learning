import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

# =======================================================
# ⚙️ KONFIGURASI
# =======================================================
INSTANSI = "dukcapil"
DATA_PATH = f"data/{INSTANSI}_round2.csv"
GLOBAL_MODEL_PATH = "Models/saved_global_tff"     # SavedModel (.pb) hasil iterasi 1
GLOBAL_FEATS_PATH = Path("Models/fitur_global_test.pkl") 
SAVE_DIR  = Path(f"Models/saved_{INSTANSI}_tff_round2")

BATCH_SIZE = 32
N_CLIENTS  = 10
ROUNDS     = 10

# =======================================================
# 📊 LOAD DATA
# =======================================================
print(f"📂 Membaca dataset {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"✅ {len(df)} baris, {len(df.columns)} kolom.")

if "layak_subsidi" not in df.columns:
    raise ValueError("Kolom 'layak_subsidi' tidak ditemukan!")

y_all = df["layak_subsidi"].astype(np.int32).values
X_raw = df.drop(columns=["layak_subsidi"])

# =======================================================
# 🌐 MUAT FITUR GLOBAL (DICT) & ONE-HOT ALIGNMENT
# =======================================================
if not GLOBAL_FEATS_PATH.exists():
    raise FileNotFoundError("❌ Tidak ditemukan: Models/fitur_global.pkl. Jalankan `fitur_global.py` dulu!")

preproc_global = joblib.load(GLOBAL_FEATS_PATH)
if not (isinstance(preproc_global, dict) and "FEATURE_COLS" in preproc_global):
    raise TypeError("❌ Models/fitur_global.pkl harus DICTIONARY dengan kunci: FEATURE_COLS, mins, rng")

FEATURE_COLS = preproc_global["FEATURE_COLS"]
mins_global  = pd.Series(preproc_global["mins"]).reindex(FEATURE_COLS).fillna(0.0)
rng_global   = pd.Series(preproc_global["rng"]).reindex(FEATURE_COLS).replace(0, 1.0)

print(f"🌍 Total fitur global: {len(FEATURE_COLS)} kolom")

# One-hot & align ke ruang fitur global
X_oh = pd.get_dummies(X_raw, drop_first=False).astype("float32")
for col in FEATURE_COLS:
    if col not in X_oh.columns:
        X_oh[col] = 0.0
X_oh = X_oh[FEATURE_COLS]

# Normalisasi mengikuti statistik GLOBAL (bukan lokal!)
X_scaled = ((X_oh - mins_global) / rng_global).fillna(0.0).astype("float32")

# =======================================================
# 👥 BUAT CLIENT DATASET
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
# 🧩 DEFINISI MODEL & INISIALISASI DARI GLOBAL
#   → Inisialisasi bobot global dilakukan DI DALAM model_fn()
#     agar state awal TFF = bobot global iterasi-1.
# =======================================================
def build_keras_model(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
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

def model_fn():
    # Bangun model
    model = build_keras_model(len(FEATURE_COLS))
    # Coba muat bobot global iterasi-1
    try:
        if os.path.isdir(GLOBAL_MODEL_PATH):
            pretrained = tf.keras.models.load_model(GLOBAL_MODEL_PATH)
            model.set_weights(pretrained.get_weights())
            print("🔗 Inisialisasi model klien dari bobot GLOBAL (iterasi-1).")
    except Exception as e:
        print("⚠️ Gagal memuat bobot global untuk inisialisasi:", e)

    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
    )

# =======================================================
# 🚀 FEDERATED TRAINING (Round-2 DUKCAPIL)
# =======================================================
process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.001),  # lebih halus
    server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.003),
)

state = process.initialize()
print("\n🚀 Mulai Federated Training — Iterasi ke-2 (DUKCAPIL) ===========================")
for r in range(1, ROUNDS + 1):
    state, metrics = process.next(state, clients)
    acc  = metrics['client_work']['train']['binary_accuracy']
    loss = metrics['client_work']['train']['loss']
    print(f"[{INSTANSI.upper()}] Round {r:02d} | acc={acc:.4f} | loss={loss:.4f}")

# =======================================================
# 💾 SIMPAN MODEL + PREPROSES (format kompatibel tester)
# =======================================================
keras_model_final = build_keras_model(len(FEATURE_COLS))
process.get_model_weights(state).assign_weights_to(keras_model_final)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
keras_model_final.save(SAVE_DIR)

# Simpan preprocess DICTIONARY (sesuai konvensi global)
joblib.dump(
    {
        "FEATURE_COLS": FEATURE_COLS,
        "mins": mins_global.to_dict(),
        "rng": rng_global.to_dict()
    },
    SAVE_DIR / f"preprocess_{INSTANSI}.pkl"
)

# =======================================================
# 📊 EVALUASI SINGKAT (TRAIN SET CLIENT)
# =======================================================
preds = (keras_model_final.predict(X_scaled.values, verbose=0) > 0.5).astype(int).flatten()
acc_eval = np.mean(preds == y_all)
print(f"\n📊 Akurasi terhadap data training {INSTANSI.upper()} (round-2): {acc_eval:.4f}")
print(f"✅ Model iterasi ke-2 ({INSTANSI.upper()}) tersimpan di: {SAVE_DIR}")


# =======================================================
# 🔍 CEK JUMLAH DAN URUTAN BOBOT MODEL
# =======================================================
weights = keras_model_final.get_weights()
print(f"\n🧠 DETAIL BOBOT MODEL ({INSTANSI.upper()})")
print(f"Total tensor: {len(weights)}\n")

for i, w in enumerate(weights):
    print(f"Layer {i+1:02d} → shape: {w.shape}")

print("\n✅ Pemeriksaan struktur layer selesai.\n")
