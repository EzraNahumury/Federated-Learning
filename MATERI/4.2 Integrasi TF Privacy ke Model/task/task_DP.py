# new.py  — Federated Averaging manual + DP-SGD di klien (TensorFlow Privacy)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

print("TensorFlow:", tf.__version__)
try:
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
    TF_PRIVACY_OK = True
    print("TF-Privacy: OK (dp_keras_optimizer tersedia)")
except Exception as e:
    TF_PRIVACY_OK = False
    print("TF-Privacy: TIDAK tersedia -> install 'tensorflow-privacy' untuk DP-SGD klien")
    raise e

# =========================================================
# 0) File input
# =========================================================
DINSOS_CSV   = "dinsos.csv"
DUKCAPIL_CSV = "dukcapil.csv"
KEMENKES_CSV = "kemenkes.csv"

# =========================================================
# 1) Load CSV
# =========================================================
def load_or_fail(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(p)

dinsos   = load_or_fail(DINSOS_CSV)
dukcapil = load_or_fail(DUKCAPIL_CSV)
kemenkes = load_or_fail(KEMENKES_CSV)

# =========================================================
# 2) Labeling rules
# =========================================================
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

# Apply label
dinsos["layak_subsidi"]   = dinsos.apply(label_dinsos, axis=1)
dukcapil["layak_subsidi"] = dukcapil.apply(label_dukcapil, axis=1)
kemenkes["layak_subsidi"] = kemenkes.apply(label_kemenkes, axis=1)

# =========================================================
# 3) Preprocess (one-hot + min-max scaling global)
# =========================================================
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

def encode_and_scale(df):
    for c in num_cols_all:
        if c not in df: df[c] = np.nan
    for c in cat_cols_all:
        if c not in df: df[c] = np.nan
    oh_parts = []
    for col, vocab in global_vocabs.items():
        for v in vocab:
            oh_parts.append((f"{col}__{v}", (df[col] == v).astype(float)))
    oh_df = pd.DataFrame({n:a.values for n,a in oh_parts}, index=df.index) if oh_parts else pd.DataFrame(index=df.index)
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

# =========================================================
# 4) Dataset per klien (PERBAIKAN: drop_remainder=True)
# =========================================================
LOCAL_EPOCHS_WARMUP = 1
LOCAL_EPOCHS_MAIN   = 3
BATCH_SIZE          = 64   # Harus habis dibagi num_microbatches
DP_NUM_MICROBATCHES = BATCH_SIZE
assert BATCH_SIZE % DP_NUM_MICROBATCHES == 0, "BATCH_SIZE harus kelipatan num_microbatches"

def to_ds(X, y, local_epochs, shuffle=True):
    X = X.values.astype("float32")
    y = y.astype("float32").reshape(-1,1)
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(y), 2048), reshuffle_each_iteration=True)
    # >>> KUNCI FIX: drop_remainder=True supaya ukuran batch SELALU = BATCH_SIZE
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).repeat(local_epochs).prefetch(tf.data.AUTOTUNE)
    return ds

client_data = [(X_dinsos, y_dinsos), (X_dukcapil, y_dukcapil), (X_kemenkes, y_kemenkes)]
client_sizes = [len(y_dinsos), len(y_dukcapil), len(y_kemenkes)]

def make_client_datasets(local_epochs, shuffle=True):
    return [to_ds(X, y, local_epochs, shuffle) for (X, y) in client_data]

# =========================================================
# 5) Model
# =========================================================
def build_model(input_dim):
    i = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(i)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    o = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(i, o)

def get_weights(model): return [w.copy() for w in model.get_weights()]
def set_weights(model, weights): model.set_weights([w.copy() for w in weights])

def evaluate_global(model):
    X_all = np.vstack([X_dinsos.values, X_dukcapil.values, X_kemenkes.values]).astype("float32")
    y_all = np.concatenate([y_dinsos, y_dukcapil, y_kemenkes]).astype("float32").reshape(-1,1)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    y_pred = model.predict(X_all, batch_size=256, verbose=0)
    l = float(loss(y_all, y_pred).numpy())
    acc = float(np.mean((y_pred >= 0.5) == y_all))
    return l, acc

# =========================================================
# 6) Federated loop (manual) + DP-SGD klien + opsional server DP
# =========================================================
DP_CLIENT_LR            = 0.05
DP_L2_NORM_CLIP_CLIENT  = 1.5
NOISE_WARMUP            = 0.001   # kecil supaya naik di awal
NOISE_MAIN              = 0.1

SERVER_CLIP             = 5.0
SERVER_NOISE_MULT       = 0.0     # biarkan 0 demi stabilitas; set >0 jika ingin DP di server
SERVER_LR               = 1.0

WARMUP_ROUNDS           = 5
MAIN_ROUNDS             = 10
TOTAL_ROUNDS            = WARMUP_ROUNDS + MAIN_ROUNDS

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

global_model = build_model(len(FEATURE_COLS))
global_model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
global_weights = get_weights(global_model)

def l2_norm_of_list(tensors):
    s = 0.0
    for t in tensors:
        s += np.sum(np.square(t))
    return float(np.sqrt(s))

def clip_by_l2_norm(tensors, clip):
    norm = l2_norm_of_list(tensors) + 1e-12
    if norm <= clip: return tensors, 1.0
    factor = clip / norm
    return [t * factor for t in tensors], factor

def add_gaussian_noise_like(tensors, std):
    if std <= 0.0:
        return [t.copy() for t in tensors]
    return [t + np.random.normal(0.0, std, size=t.shape).astype(t.dtype) for t in tensors]

print("\n=== Federated Training (manual) ===")
ma_acc = None
MA_ALPHA = 0.3

for round_idx in range(1, TOTAL_ROUNDS + 1):
    if round_idx <= WARMUP_ROUNDS:
        local_epochs = LOCAL_EPOCHS_WARMUP
        client_noise = NOISE_WARMUP
        shuffle_flag = False
    else:
        local_epochs = LOCAL_EPOCHS_MAIN
        client_noise = NOISE_MAIN
        shuffle_flag = True

    client_ds = make_client_datasets(local_epochs=local_epochs, shuffle=shuffle_flag)

    clipped_weighted_sums = None
    total_weight = 0.0
    avg_local_loss = []

    for k, ds in enumerate(client_ds):
        # Model lokal start dari bobot global
        local_model = build_model(len(FEATURE_COLS))
        set_weights(local_model, global_weights)

        # DP optimizer
        opt = DPKerasSGDOptimizer(
            l2_norm_clip=DP_L2_NORM_CLIP_CLIENT,
            noise_multiplier=client_noise,
            num_microbatches=DP_NUM_MICROBATCHES,   # = BATCH_SIZE
            learning_rate=DP_CLIENT_LR,
            momentum=0.9
        )
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)
        local_model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

        # Train lokal (dataset sudah repeat(local_epochs); karena drop_remainder=True,
        # semua batch berukuran tepat BATCH_SIZE -> aman utk num_microbatches)
        hist = local_model.fit(ds, epochs=1, verbose=0)
        avg_local_loss.append(hist.history["loss"][-1])

        # Delta
        w_local = local_model.get_weights()
        delta = [wl - wg for wl, wg in zip(w_local, global_weights)]

        # Clip user-level sebelum agregasi
        delta_clipped, _ = clip_by_l2_norm(delta, SERVER_CLIP)

        weight_k = float(client_sizes[k])
        if clipped_weighted_sums is None:
            clipped_weighted_sums = [d * weight_k for d in delta_clipped]
        else:
            for i in range(len(delta_clipped)):
                clipped_weighted_sums[i] += delta_clipped[i] * weight_k
        total_weight += weight_k

    # Server noise (opsional)
    if SERVER_NOISE_MULT > 0.0:
        for i in range(len(clipped_weighted_sums)):
            std = SERVER_CLIP * SERVER_NOISE_MULT
            clipped_weighted_sums[i] = clipped_weighted_sums[i] + \
                np.random.normal(0.0, std, size=clipped_weighted_sums[i].shape).astype(clipped_weighted_sums[i].dtype)

    # Update global
    avg_delta = [cw / (total_weight + 1e-12) for cw in clipped_weighted_sums]
    avg_delta = [SERVER_LR * d for d in avg_delta]
    global_weights = [wg + d for wg, d in zip(global_weights, avg_delta)]
    set_weights(global_model, global_weights)

    # Evaluasi gabungan
    gl_loss, gl_acc = evaluate_global(global_model)
    ma_acc = gl_acc if ma_acc is None else (MA_ALPHA * gl_acc + (1 - MA_ALPHA) * ma_acc)

    print(f"Round {round_idx:02d} | "
          f"client_loss={np.mean(avg_local_loss):.4f} | "
          f"global_loss={gl_loss:.4f} | global_acc={gl_acc:.4f} | ma_acc={ma_acc:.4f} | "
          f"client_noise={client_noise}")

print("\nTraining selesai (FedAvg manual + DP-SGD di klien).")

# =========================================================
# 7)  Estimasi epsilon untuk fase utama
# =========================================================
try:
    n = max(client_sizes)
    eps, best_alpha = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
        n=n,
        batch_size=BATCH_SIZE,
        noise_multiplier=NOISE_MAIN,
        epochs=LOCAL_EPOCHS_MAIN,
        delta=1e-5
    )
    print(f"Perkiraan DP-SGD klien (fase utama): ε ≈ {eps:.2f}, δ = 1e-5 (α = {best_alpha})")
except Exception as e:
    print("Lewati perhitungan epsilon:", e)
