import tensorflow as tf
import numpy as np
import requests
import gzip
import io
import base64

SERVER_URL = "https://federatedserver-production.up.railway.app"
CLIENT_NAME = "dukcapil"
MODEL_PATH = "models/saved_dukcapil_tff"

def load_local_model(model_path):
    try:
        # ✅ Coba cara normal (untuk .keras atau .h5)
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model lokal berhasil dimuat dari: {model_path}")
    except ValueError:
        # ⚙️ Jika gagal (karena .pb), pakai TFSMLayer
        print("⚠️ Deteksi format SavedModel (.pb), memuat dengan TFSMLayer...")
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        ])
        print(f"✅ Model SavedModel (.pb) berhasil dimuat dari: {model_path}")
    return model

def upload_model_to_server(model):
    print(f"📦 Menyiapkan bobot model {CLIENT_NAME} untuk dikirim...")
    weights = []
    # ⚙️ Jika model hasil TFSMLayer, ambil bobot dengan cara manual
    for var in model.weights:
        weights.append(var.numpy())

    np_weights = np.array(weights, dtype=object)

    # Simpan bobot dalam format biner terkompresi
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *np_weights)
    compressed_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    payload = {
        "client": CLIENT_NAME,
        "compressed_weights": compressed_data
    }

    print(f"📡 Mengirim model {CLIENT_NAME} ke server...")
    res = requests.post(f"{SERVER_URL}/upload-model", json=payload, timeout=120)
    print("📨 Response server:", res.text)

if __name__ == "__main__":
    model = load_local_model(MODEL_PATH)
    upload_model_to_server(model)
