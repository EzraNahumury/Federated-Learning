import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np, pandas as pd, joblib
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

# ================== CONFIG ==================
PATH_CSV   = "NEW_DATASET/kemenkes_labeled.csv"
SAVE_DIR   = "saved_kemenkes_tff"
BATCH_SIZE = 32
ROUNDS     = 5
N_CLIENTS  = 5

print("TensorFlow:", tf.__version__)
print("TFF:", tff.__version__)

# =============== RULE ENGINE ===============
KRONIS_SET = {"kronis","jantung","asma","diabetes","disabilitas"}

def label_kemenkes_row(row):
    rp = str(row.get("riwayat_penyakit", "")).lower()
    sg = str(row.get("status_gizi", "")).lower()
    t  = row.get("tinggi_cm", None)
    b  = row.get("berat_kg", None)

    bmi = None
    if pd.notna(t) and pd.notna(b) and t and t > 0:
        bmi = b / ((t/100.0) ** 2)

    # Layak
    if (rp in KRONIS_SET) and (sg in {"kurang","stunting","gizi buruk"}) and (bmi is not None and bmi < 18.5):
        return 1
    if (rp == "hipertensi") and (bmi is not None and bmi >= 25):
        return 1
    if (sg == "baik") and (bmi is not None) and (bmi < 17 or bmi > 35):
        return 1
    if (rp not in KRONIS_SET) and (sg == "baik") and (bmi is not None and bmi < 19):
        return 1
    if (rp in KRONIS_SET) and (sg == "baik") and (bmi is not None and (bmi < 18.5 or bmi > 30)):
        return 1

    # Tidak Layak
    if (rp == "sehat") and (sg == "baik") and (bmi is not None and 18.5 <= bmi <= 25):
        return 0

    return 0

# =============== LOAD & PREP ===============
df = pd.read_csv(PATH_CSV)

# label via aturan
y_all = df.apply(label_kemenkes_row, axis=1).astype(np.int32).values

# fitur: numerik + BMI + kategori
df_feat = df.copy()
# hitung BMI sebagai fitur tambahan
bmi = df_feat["berat_kg"] / ((df_feat["tinggi_cm"]/100.0) ** 2)
df_feat["bmi"] = bmi.replace([np.inf, -np.inf], np.nan).fillna(0.0)

num_cols = ["tinggi_cm","berat_kg","bmi"]
cat_cols = ["riwayat_penyakit","status_gizi"]
oh_df    = pd.get_dummies(df_feat[cat_cols].fillna("unknown").astype(str), drop_first=False).astype(float)
X_raw    = pd.concat([df_feat[num_cols], oh_df], axis=1)

mins = X_raw.min()
rng  = (X_raw.max() - mins).replace(0, 1.0)
X_scaled = ((X_raw - mins) / rng).fillna(0.0).astype("float32")
FEATURE_COLS = list(X_scaled.columns)

def to_tf_dataset(X: pd.DataFrame, y: np.ndarray, batch=BATCH_SIZE):
    feats = X.values.astype("float32")
    labels = y.astype("float32").reshape(-1,1)
    return tf.data.Dataset.from_tensor_slices((feats, labels)).shuffle(len(X)).batch(batch)

def split_clients(X: pd.DataFrame, y: np.ndarray, n_clients=N_CLIENTS):
    size = len(X) // n_clients
    clients = []
    for i in range(n_clients):
        s, e = i*size, (i+1)*size if i < n_clients-1 else len(X)
        clients.append(to_tf_dataset(X.iloc[s:e], y[s:e]))
    return clients

clients = split_clients(X_scaled, y_all, N_CLIENTS)

# =============== MODEL FN ==================
def model_fn():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1,  activation="sigmoid"),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=m,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
    )

# ============== FEDERATED AVG ==============
process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(0.05, 0.9),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(1.0, 0.9),
)
state = process.initialize()

for r in range(1, ROUNDS+1):
    state, metrics = process.next(state, clients)
    acc  = metrics['client_work']['train']['binary_accuracy']
    loss = metrics['client_work']['train']['loss']
    print(f"[Kemenkes] Round {r:02d}  acc={acc:.4f}  loss={loss:.4f}")

# ========== EXTRACT & SAVE KERAS ===========
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
])
process.get_model_weights(state).assign_weights_to(keras_model)

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
keras_model.save(SAVE_DIR)

joblib.dump(
    {"FEATURE_COLS": FEATURE_COLS, "mins": mins, "rng": rng},
    "preprocess_kemenkes_rule.pkl"
)

preds = (keras_model.predict(X_scaled.values, verbose=0) > 0.5).astype(int).flatten()
print("Akurasi thd aturan:", np.mean(preds == y_all))
