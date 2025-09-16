# hasil.py — Federated Learning dengan TFF 0.87.0 + Server-level DP Aggregator
# Pakai dataset Dinsos, Dukcapil, Kemenkes + aturan labeling

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("TensorFlow:", tf.__version__)
print("TFF:", tff.__version__)

# ============================================================
# 0) Load CSV
# ============================================================
def load_or_fail(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(p)

dinsos   = load_or_fail("dinsos.csv")
dukcapil = load_or_fail("dukcapil.csv")
kemenkes = load_or_fail("kemenkes.csv")

# ============================================================
# 1) Aturan Labeling
# ============================================================
def label_dinsos(row):
    jt = row.get("jumlah_tanggungan", np.nan)
    ph = row.get("penghasilan", np.nan)
    kr = row.get("kondisi_rumah", "")
    if pd.isna(jt): jt = 0
    if pd.isna(ph): ph = 0
    if ph < 2_000_000: return 1
    if (2_000_000 <= ph < 3_500_000) and jt >= 4: return 1
    if (kr in {"tidak layak","semi permanen","sangat sederhana"}) and (ph < 5_000_000): return 1
    if (jt >= 5) and (ph < 6_000_000): return 1
    if ph >= 8_000_000: return 0
    return 0

def label_dukcapil(row):
    u  = row.get("umur", np.nan)
    sp = row.get("status_pekerjaan", "")
    st = row.get("status_pernikahan", "")
    if pd.isna(u): u = 0
    if u > 65: return 1
    if sp in {"pengangguran","buruh","pekerja informal"}: return 1
    if (sp=="wirausaha") and (u>55): return 1
    if (st in {"cerai","janda","duda"}) and (sp!="pegawai tetap"): return 1
    if (sp=="pegawai tetap") and (25<=u<=55) and (st=="menikah"): return 0
    if (u<25) and (sp=="wirausaha"): return 0
    return 0

def label_kemenkes(row):
    rp = row.get("riwayat_penyakit", "")
    sg = row.get("status_gizi", "")
    t  = row.get("tinggi_cm", np.nan)
    b  = row.get("berat_kg", np.nan)
    bmi = None
    if pd.notna(t) and pd.notna(b) and t > 0:
        bmi = b / ((t/100.0) ** 2)
    if rp in {"kronis","jantung","asma","diabetes","disabilitas"}: return 1
    if sg in {"kurang","stunting","gizi buruk"}: return 1
    if (bmi is not None) and (bmi < 18.5): return 1
    if (bmi is not None) and ((bmi < 17) or (bmi > 35)): return 1
    return 0

dinsos["layak_subsidi"]   = dinsos.apply(label_dinsos, axis=1)
dukcapil["layak_subsidi"] = dukcapil.apply(label_dukcapil, axis=1)
kemenkes["layak_subsidi"] = kemenkes.apply(label_kemenkes, axis=1)

# ============================================================
# 2) Preprocessing (one-hot + min-max global)
# ============================================================
cat_cols_all = ["kondisi_rumah","status_pekerjaan","status_pernikahan","riwayat_penyakit","status_gizi"]
num_cols_all = ["jumlah_tanggungan","penghasilan","umur","tinggi_cm","berat_kg"]

for df in (dinsos, dukcapil, kemenkes):
    if {"tinggi_cm","berat_kg"}.issubset(df.columns):
        df["BMI"] = df["berat_kg"] / ((df["tinggi_cm"]/100.0) ** 2)
    else:
        df["BMI"] = np.nan
if "BMI" not in num_cols_all:
    num_cols_all.append("BMI")

def union_categories(series_list):
    cats = set()
    for s in series_list:
        if s is not None:
            for v in s.dropna().unique().tolist():
                if isinstance(v, str): cats.add(v)
    return sorted(list(cats))

global_vocabs = {
    "kondisi_rumah":     union_categories([dinsos.get("kondisi_rumah")]),
    "status_pekerjaan":  union_categories([dukcapil.get("status_pekerjaan")]),
    "status_pernikahan": union_categories([dukcapil.get("status_pernikahan")]),
    "riwayat_penyakit":  union_categories([kemenkes.get("riwayat_penyakit")]),
    "status_gizi":       union_categories([kemenkes.get("status_gizi")]),
}

def encode_and_scale(df: pd.DataFrame):
    for c in num_cols_all:
        if c not in df: df[c] = np.nan
    for c in cat_cols_all:
        if c not in df: df[c] = np.nan

    oh_parts = []
    for col, vocab in global_vocabs.items():
        for v in vocab:
            oh_parts.append((f"{col}__{v}", (df[col] == v).astype(float)))
    oh_df = pd.DataFrame({n:a.values for n,a in oh_parts}, index=df.index)

    num_df = df[num_cols_all].copy()
    for c in num_cols_all:
        if not np.isfinite(num_df[c]).any(): num_df[c] = 0.0
        else: num_df[c] = num_df[c].fillna(num_df[c].median())

    X_raw = pd.concat([num_df, oh_df], axis=1)
    y = df["layak_subsidi"].astype(int).values
    return X_raw, y

X_dinsos_raw,   y_dinsos   = encode_and_scale(dinsos)
X_dukcapil_raw, y_dukcapil = encode_and_scale(dukcapil)
X_kemenkes_raw, y_kemenkes = encode_and_scale(kemenkes)

all_X = pd.concat([X_dinsos_raw, X_dukcapil_raw, X_kemenkes_raw], axis=0)
mins, maxs = all_X.min(axis=0), all_X.max(axis=0)
rng = (maxs - mins).replace(0, 1.0)
def scale_like_global(X): return (X - mins) / rng

X_dinsos   = scale_like_global(X_dinsos_raw).fillna(0.0)
X_dukcapil = scale_like_global(X_dukcapil_raw).fillna(0.0)
X_kemenkes = scale_like_global(X_kemenkes_raw).fillna(0.0)

FEATURE_COLS = list(X_dinsos.columns)

# ============================================================
# 3) Dataset federated untuk TFF
# ============================================================
def to_tf_dataset(X, y, batch_size=64):
    Xf = X.astype("float32").values
    yf = y.astype("float32").reshape(-1,1)
    return tf.data.Dataset.from_tensor_slices((Xf, yf)).batch(batch_size)

clients = [
    ("dinsos",   to_tf_dataset(X_dinsos,   y_dinsos),   len(y_dinsos)),
    ("dukcapil", to_tf_dataset(X_dukcapil, y_dukcapil), len(y_dukcapil)),
    ("kemenkes", to_tf_dataset(X_kemenkes, y_kemenkes), len(y_kemenkes)),
]
total_clients = len(clients)

# ============================================================
# 4) Model TFF
# ============================================================
def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=clients[0][1].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )

# ============================================================
# 5) Server-level DP Aggregator 
# ============================================================
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_federated.python.aggregators import differential_privacy

def build_process_with_server_dp(noise_multiplier, clients_per_round):
    # Buat GaussianSumQuery (dari TF-Privacy 0.9.0)
    dp_query = gaussian_query.GaussianSumQuery(
        l2_norm_clip=1.0,
        stddev=noise_multiplier
    )

    # Bungkus query jadi factory (TFF 0.87.0 API)
    dp_factory = differential_privacy.DifferentiallyPrivateFactory(dp_query)

    return tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(
            learning_rate=0.05, momentum=0.9),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(
            learning_rate=1.0, momentum=0.9),
        model_aggregator=dp_factory
    )


# ============================================================
# 6) Training loop (perbaikan sampling client)
# ============================================================
def run_experiment(noise_multiplier, rounds=10, clients_per_round=None):
    if clients_per_round is None:
        clients_per_round = total_clients

    learning_process = build_process_with_server_dp(noise_multiplier, clients_per_round)
    state = learning_process.initialize()

    logs = []
    for r in range(1, rounds+1):
        # ambil random sample fix sejumlah clients_per_round
        idx = np.random.choice(range(total_clients), size=clients_per_round, replace=False)
        sampled = [clients[i][1] for i in idx]

        state, metrics = learning_process.next(state, sampled)

        train_acc  = float(metrics['client_work']['train']['binary_accuracy'])
        train_loss = float(metrics['client_work']['train']['loss'])

        print(f"[nm={noise_multiplier:.3f}] Round {r:02d} "
              f"| train_acc={train_acc:.4f} | train_loss={train_loss:.4f}")

        logs.append({
            "Round": r,
            "NoiseMultiplier": noise_multiplier,
            "train_binary_accuracy": train_acc,
            "train_loss": train_loss,
        })
    return pd.DataFrame(logs)

# ============================================================
# 7) Jalankan semua noise
# ============================================================
NOISE_LIST = [0.01, 0.05, 0.1, 0.5, 1.0]
ROUNDS = 5
all_logs = []

for nm in NOISE_LIST:
    df_log = run_experiment(
        noise_multiplier=nm,
        rounds=ROUNDS,
        clients_per_round=total_clients  # semua client ikut
    )
    all_logs.append(df_log)

df_log_all = pd.concat(all_logs, ignore_index=True)

print("\n=== Rata-rata Akurasi per NoiseMultiplier ===")
avg_acc = df_log_all.groupby("NoiseMultiplier")["train_binary_accuracy"].mean().reset_index()
print(avg_acc)



# ============================================================
# 8) Plot
# ============================================================
plt.figure(figsize=(9,5))
sns.lineplot(data=df_log_all, x="Round", y="train_binary_accuracy", hue="NoiseMultiplier", marker="o")
plt.ylabel("Train Accuracy")
plt.title("Federated Learning (TFF 0.87.0 + Server DP Aggregator)")
plt.grid(True)
plt.tight_layout()
plt.savefig("tff_serverdp_accuracy.png", dpi=150)
print("Grafik disimpan: tff_serverdp_accuracy.png")

