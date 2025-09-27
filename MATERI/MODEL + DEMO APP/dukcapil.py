import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np, pandas as pd, joblib
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

# ================== CONFIG ==================
PATH_CSV   = "NEW_DATASET/dukcapil_labeled.csv"
SAVE_DIR   = "saved_dukcapil_tff"
BATCH_SIZE = 32
ROUNDS     = 5
N_CLIENTS  = 5

print("TensorFlow:", tf.__version__)
print("TFF:", tff.__version__)

# =============== RULE ENGINE ===============
def label_dukcapil_row(row):
    u  = int(row.get("umur", 0))
    sp = str(row.get("status_pekerjaan", "")).lower()
    st = str(row.get("status_pernikahan", "")).lower()

    # Layak
    if (u > 65) and (sp in {"pengangguran","buruh","pekerja informal"}) and (st in {"janda","duda","cerai"}):
        return 1
    if (55 <= u <= 65) and (sp == "wirausaha") and (st != "menikah"):
        return 1
    if (40 <= u <= 60) and (sp == "buruh") and (st == "menikah"):
        return 1

    # Tidak Layak
    if (25 <= u <= 55) and (sp in {"pegawai tetap","pns","karyawan tetap"}) and (st == "menikah"):
        return 0
    if (u < 25) and (sp == "wirausaha") and (st in {"lajang","menikah"}):
        return 0

    return 0

# =============== LOAD & PREP ===============
df = pd.read_csv(PATH_CSV)
y_all = df.apply(label_dukcapil_row, axis=1).astype(np.int32).values

num_cols = ["umur"]
cat_cols = ["status_pekerjaan","status_pernikahan"]
oh_df    = pd.get_dummies(df[cat_cols], drop_first=False).astype(float)
X_raw    = pd.concat([df[num_cols], oh_df], axis=1)

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
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
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
    print(f"[Dukcapil] Round {r:02d}  acc={acc:.4f}  loss={loss:.4f}")

# ========== EXTRACT & SAVE KERAS ===========
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
])
process.get_model_weights(state).assign_weights_to(keras_model)

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
keras_model.save(SAVE_DIR)

joblib.dump(
    {"FEATURE_COLS": FEATURE_COLS, "mins": mins, "rng": rng},
    "preprocess_dukcapil_rule.pkl"
)

preds = (keras_model.predict(X_scaled.values, verbose=0) > 0.5).astype(int).flatten()
print("Akurasi thd aturan:", np.mean(preds == y_all))
