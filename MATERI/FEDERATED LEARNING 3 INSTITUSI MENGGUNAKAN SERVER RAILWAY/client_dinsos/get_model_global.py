import tensorflow as tf
import numpy as np
import requests, os

# ============================================================
# 🌍 Download model global hasil agregasi dari server
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
# 📊 Muat semua tensor dari file NPZ
# ============================================================
npz = np.load(save_path, allow_pickle=True)
weights_all = [npz[key] for key in npz]
print(f"📊 Jumlah tensor bobot di file global: {len(weights_all)}")

# ============================================================
# 🧹 Ambil hanya tensor valid (buang 4 tensor (128,) dummy)
# ============================================================
valid_idx = [0, 1, 6, 7, 8, 9, 10, 11]
weights = [weights_all[i] for i in valid_idx]
print(f"✅ Bobot yang akan digunakan: {len(weights)} tensor ({len(weights)//2} layer)")

# ============================================================
# 🧠 Buat model dengan arsitektur sesuai federated training
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(53,)),   # input sesuai tensor pertama
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ============================================================
# 🔗 Pasang bobot global
# ============================================================
try:
    model.set_weights(weights)
    print("✅ Bobot global berhasil diterapkan ke model lokal!")
except Exception as e:
    print("❌ Gagal memasang bobot:", e)
    exit(1)

# ============================================================
# 💾 Simpan model global dalam format SavedModel (.pb)
# ============================================================
export_dir = "models/saved_global_tff"
model.export(export_dir)
print(f"💾 Model global tersimpan di folder: {export_dir}")
print("📁 Struktur folder:")
print("   ├── saved_model.pb")
print("   ├── keras_metadata.pb")
print("   └── variables/")
