import tensorflow as tf
import numpy as np
import requests, os, joblib
from pathlib import Path

# ============================================================
# 🌍 Unduh model global hasil agregasi FedAvg
# ============================================================
URL = "https://federatedserver-production.up.railway.app/download-global"
print("🌍 Mengunduh model global hasil agregasi FedAvg...")

response = requests.get(URL)
if response.status_code != 200:
    raise Exception(f"❌ Gagal mengunduh model: {response.status_code} - {response.text}")

os.makedirs("models", exist_ok=True)
save_path = "models/global_model_fedavg.npz"

with open(save_path, "wb") as f:
    f.write(response.content)

print(f"✅ Model global disimpan ke: {save_path}")

# ============================================================
# 📊 Muat tensor dari file global
# ============================================================
npz = np.load(save_path, allow_pickle=True)
weights_all = [npz[key] for key in npz]
print(f"📊 Jumlah total tensor di file global: {len(weights_all)}")

# ============================================================
# 🧩 Muat fitur global agar input shape sesuai
# ============================================================
GLOBAL_FEATS_PATH = Path("Models/fitur_global_dict_baru.pkl")
if not GLOBAL_FEATS_PATH.exists():
    raise FileNotFoundError("❌ File 'Models/fitur_global.pkl' tidak ditemukan!")

GLOBAL_FEATURES = joblib.load(GLOBAL_FEATS_PATH)
if not isinstance(GLOBAL_FEATURES, list):
    raise TypeError("❌ File fitur_global.pkl harus berupa list kolom fitur!")

input_dim = len(GLOBAL_FEATURES)
print(f"📏 Total fitur global: {input_dim} kolom")

# ============================================================
# 🧠 Buat model lokal sesuai arsitektur client 
# ============================================================
model = tf.keras.Sequential([
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


expected_shapes = [w.shape for w in model.get_weights()]
print(f"📏 Model lokal mengharapkan {len(expected_shapes)} tensor:")
for i, s in enumerate(expected_shapes):
    print(f"   {i+1}. {s}")

# ============================================================
# 🔎 Cocokkan tensor global dengan model lokal
# ============================================================
matched_weights = []
used_idx = []
for shape in expected_shapes:
    for i, w in enumerate(weights_all):
        if i not in used_idx and w.shape == shape:
            matched_weights.append(w)
            used_idx.append(i)
            break
    else:
        raise ValueError(f"❌ Tidak menemukan tensor cocok untuk shape {shape}")

print(f"✅ {len(matched_weights)} tensor cocok ditemukan dan siap diterapkan")

# ============================================================
# 🔗 Terapkan bobot global ke model lokal
# ============================================================
try:
    model.set_weights(matched_weights)
    print("✅ Bobot global berhasil diterapkan ke model lokal!")
except Exception as e:
    print("❌ Gagal memasang bobot:", e)
    exit(1)

# ============================================================
# 💾 Simpan model global ke format SavedModel (.pb)
# ============================================================
export_dir = "models/saved_global_tff"
model.export(export_dir)

print(f"💾 Model global tersimpan di folder: {export_dir}")
print("📁 Struktur folder:")
print("   ├── saved_model.pb")
print("   ├── keras_metadata.pb")
print("   └── variables/")
