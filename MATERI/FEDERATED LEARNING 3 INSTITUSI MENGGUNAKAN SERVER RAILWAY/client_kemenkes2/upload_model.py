import tensorflow as tf
import numpy as np
import requests
import io
import base64
from tensorflow import keras

# ============================================================
# ⚙️ KONFIGURASI
# ============================================================
SERVER_URL = "https://federatedserver-production.up.railway.app"
CLIENT_NAME = "kemenkes"   # ubah sesuai nama lembaga
MODEL_PATH = "Models/saved_kemenkes_tff_round2"   # folder hasil training (.pb atau .keras)


# ============================================================
# 🧠 1️⃣ Muat model lokal (Keras 3 / SavedModel .pb)
# ============================================================
def load_local_model(model_path):
    """
    Memuat model dari format .keras / .h5 (Keras 3) atau SavedModel (.pb)
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model {CLIENT_NAME} berhasil dimuat dari: {model_path}")
    except Exception as e:
        print(f"⚠️ Gagal memuat model biasa ({e}) → mencoba TFSMLayer...")
        try:
            model = keras.Sequential([
                keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
            ])
            print(f"✅ Model {CLIENT_NAME} berhasil dimuat via TFSMLayer.")
        except Exception as e2:
            raise RuntimeError(f"❌ Gagal memuat model: {e2}")
    return model


# ============================================================
# 💾 2️⃣ Serialisasi bobot model ke Base64 (untuk dikirim via JSON)
# ============================================================
def encode_model_weights(model):
    """
    Mengekstrak bobot model dan mengubahnya ke base64 string.
    """
    print(f"💾 Mengekstrak dan mengkompres bobot model {CLIENT_NAME} ...")
    weights = []

    try:
        weights = [w.numpy() for w in model.weights]
    except Exception as e:
        print("⚠️ Tidak bisa langsung ambil model.weights:", e)
        for layer in model.layers:
            for var in getattr(layer, "weights", []):
                weights.append(var.numpy())

    if not weights:
        raise ValueError("❌ Tidak ada bobot ditemukan pada model!")

    buffer = io.BytesIO()
    np.savez_compressed(buffer, *weights)
    compressed_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(f"✅ Bobot berhasil dikonversi ke base64 ({len(weights)} tensor).")
    return compressed_data


# ============================================================
# 🚀 3️⃣ Upload model ke server (JSON mode)
# ============================================================
def upload_model_to_server(model):
    """
    Mengirim model ke server via JSON base64.
    Server harus mendukung key: 'client' dan 'compressed_weights'.
    """
    compressed_data = encode_model_weights(model)

    payload = {
        "client": CLIENT_NAME,
        "compressed_weights": compressed_data
    }

    print(f"📡 Mengupload model {CLIENT_NAME} ke server...")
    try:
        res = requests.post(f"{SERVER_URL}/upload-model", json=payload, timeout=120)
        print("📨 Respons server:")
        print(res.text)
    except requests.exceptions.RequestException as e:
        print(f"❌ Gagal mengupload ke server: {e}")


# ============================================================
# 🧩 MAIN
# ============================================================
if __name__ == "__main__":
    print(f"🚀 Memulai upload model federated: {CLIENT_NAME.upper()}")
    print("===============================================================")
    model = load_local_model(MODEL_PATH)
    upload_model_to_server(model)
    print("===============================================================")
    print(f"✅ Upload model {CLIENT_NAME.upper()} selesai!\n")
