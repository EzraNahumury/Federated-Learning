import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np, pandas as pd, joblib
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

# ================== CONFIG ==================
PATH_CSV   = "NEW_DATASET/dinsos_labeled.csv"
SAVE_DIR   = "saved_dinsos_tff"
BATCH_SIZE = 32
ROUNDS     = 5
N_CLIENTS  = 5

print("TensorFlow:", tf.__version__)
print("TFF:", tff.__version__)

# =============== RULE ENGINE ===============
def build_rule_table(df: pd.DataFrame) -> pd.DataFrame:
    penghasilan = df["penghasilan"].astype(float)
    tanggungan  = df["jumlah_tanggungan"].astype(int)
    rumah       = df["kondisi_rumah"].astype(str).str.lower()

    rule = pd.DataFrame(index=df.index)
    rule["layak1"] = ((penghasilan < 2_000_000) & (tanggungan >= 1) &
                      (rumah.isin(["tidak layak","semi permanen","sangat sederhana"]))).astype(int)
    rule["layak2"] = ((2_000_000 <= penghasilan) & (penghasilan < 3_500_000) &
                      (tanggungan >= 4)).astype(int)
    rule["layak3"] = ((penghasilan < 5_000_000) &
                      (rumah.isin(["tidak layak","semi permanen","sangat sederhana"]))).astype(int)
    rule["layak4"] = ((tanggungan >= 5) & (penghasilan < 6_000_000)).astype(int)

    rule["tidak1"] = ((penghasilan >= 6_000_000) & (tanggungan <= 2) &
                      (rumah == "layak")).astype(int)
    rule["tidak2"] = (penghasilan >= 8_000_000).astype(int)

    layak = (rule[["layak1","layak2","layak3","layak4"]].sum(axis=1) > 0).astype(int)
    tidak = (rule[["tidak1","tidak2"]].sum(axis=1) > 0).astype(int)
    hasil = layak.copy()
    hasil[(layak==1) & (tidak==1)] = 0  # tie-break: tidak layak
    rule["hasil_aturan"] = hasil
    return rule

# =============== LOAD & PREP ===============
df = pd.read_csv(PATH_CSV)

# label dari aturan
y_all = build_rule_table(df)["hasil_aturan"].astype(np.int32).values

# fitur
num_cols = ["penghasilan","jumlah_tanggungan"]
cat_cols = ["kondisi_rumah"]
oh_df    = pd.get_dummies(df[cat_cols], drop_first=False).astype(float)
X_raw    = pd.concat([df[num_cols], oh_df], axis=1)

# scaling min-max (aman buat inference)
mins = X_raw.min()
rng  = (X_raw.max() - mins).replace(0, 1.0)
X_scaled = ((X_raw - mins) / rng).fillna(0.0).astype("float32")
FEATURE_COLS = list(X_scaled.columns)

def to_tf_dataset(X: pd.DataFrame, y: np.ndarray, batch=BATCH_SIZE):
    feats = X.values.astype("float32")
    labels = y.astype("float32").reshape(-1,1)
    return tf.data.Dataset.from_tensor_slices((feats, labels)).shuffle(len(X)).batch(batch)

# federated split
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
    print(f"[Dinsos] Round {r:02d}  acc={acc:.4f}  loss={loss:.4f}")

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
    "preprocess_dinsos_rule.pkl"
)

# optional quick eval on all data
preds = (keras_model.predict(X_scaled.values, verbose=0) > 0.5).astype(int).flatten()
print("Akurasi thd aturan:", np.mean(preds == y_all))
