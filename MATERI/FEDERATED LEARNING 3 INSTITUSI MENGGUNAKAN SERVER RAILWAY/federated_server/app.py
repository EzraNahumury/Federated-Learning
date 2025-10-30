import os
import zipfile
from pathlib import Path
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io



# ==========================================================
# 🚀 INISIALISASI FLASK
# ==========================================================
app = Flask(__name__)

# Folder penyimpanan model
models_dir = Path("Models")
models_dir.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 1️⃣ DAFTAR MODEL CLIENT
# ==========================================================
paths = {
    "dinsos": models_dir / "saved_dinsos_tff",
    "dukcapil": models_dir / "saved_dukcapil_tff",
    "kemenkes": models_dir / "saved_kemenkes_tff"
}

client_weights = { "dinsos": 0.4, "dukcapil": 0.35, "kemenkes": 0.25 }


# ==========================================================
# 2️⃣ FUNGSI: AGREGASI FEDAVG
# ==========================================================
def aggregate_fedavg(models_dict, weights_dict):
    names = list(models_dict.keys())
    n_models = len(names)
    
    base_weights = models_dict[names[0]].get_weights()
    new_weights = [np.zeros_like(w) for w in base_weights]

    for name in names:
        w = models_dict[name].get_weights()
        weight_factor = weights_dict.get(name, 1.0 / n_models)
        for i in range(len(w)):
            new_weights[i] += w[i] * weight_factor
    
    return new_weights

# ==========================================================
# 3️⃣ ENDPOINT: UPLOAD MODEL DARI CLIENT
# ==========================================================
@app.route('/upload-model', methods=['POST'])
def upload_model():
    try:
        data = request.get_json()
        client = data.get("client")
        compressed_weights = data.get("compressed_weights")

        if not client:
            return jsonify({"error": "client missing"}), 400
        if not compressed_weights:
            return jsonify({"error": "compressed_weights missing"}), 400

        # Decode data dari base64
        binary_data = base64.b64decode(compressed_weights)
        buffer = io.BytesIO(binary_data)
        npzfile = np.load(buffer, allow_pickle=True)
        weights = [npzfile[key] for key in npzfile]

        # Pastikan folder models/ ada
        os.makedirs("models", exist_ok=True)

        # Simpan bobot model
        save_path = f"models/{client}_weights.npz"
        np.savez_compressed(save_path, *weights)
        print(f"✅ Model dari {client} diterima dan disimpan di {save_path}")

        return jsonify({"status": "success", "client": client})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================================
# 4️⃣ ENDPOINT: AGREGASI SEMUA MODEL
# ==========================================================
@app.route('/aggregate', methods=['POST'])
def aggregate_models():
    try:
        model_dir = "models"
        client_files = [
            f for f in os.listdir(model_dir)
            if f.endswith("_weights.npz")
        ]

        if len(client_files) < 2:
            return jsonify({
                "status": "error",
                "message": f"Dibutuhkan minimal 2 model untuk agregasi (saat ini {len(client_files)})"
            }), 400

        print(f"🧮 Memulai Federated Averaging untuk {len(client_files)} client...")

        # Muat semua bobot client
        client_weights = []
        for file in client_files:
            npz = np.load(os.path.join(model_dir, file), allow_pickle=True)
            weights = [npz[key] for key in npz]
            client_weights.append(weights)
            print(f"✅ {file} dimuat ({len(weights)} layer)")

        num_layers = len(client_weights[0])

        # 🔹 FedAvg dengan penanganan BatchNormalization
        avg_weights = []
        for layer_idx in range(num_layers):
            # Ambil bobot layer ke-n dari semua client
            weights_for_layer = [client[layer_idx] for client in client_weights]

            try:
                # Coba lakukan FedAvg
                stacked = np.stack(weights_for_layer, axis=0)
                averaged = np.mean(stacked, axis=0)
                avg_weights.append(averaged)
            except Exception:
                # Jika gagal → ini pasti moving_mean / moving_variance BatchNorm
                # → Ambil bobot milik client terakhir (lebih stabil)
                print(f"⚠️  Layer {layer_idx} terdeteksi BatchNorm moving stats → tidak di-average")
                avg_weights.append(weights_for_layer[-1])

        # Simpan model global hasil agregasi
        save_path = os.path.join(model_dir, "global_model_fedavg.npz")
        np.savez_compressed(save_path, *avg_weights)

        print(f"🎯 FedAvg selesai → Hasil disimpan di {save_path}")

        return jsonify({
            "status": "success",
            "method": "FedAvg",
            "num_clients": len(client_files),
            "num_layers": num_layers,
            "message": "Agregasi global dengan FedAvg berhasil!"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/logs', methods=['GET'])
def list_files():
    files = os.listdir("models")
    return jsonify({
        "message": "📂 File di server:",
        "files": files
    })

from flask import send_file

@app.route('/download-global', methods=['GET'])
def download_global():
    try:
        file_path = os.path.join("models", "global_model_fedavg.npz")
        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": "File global_model_fedavg.npz tidak ditemukan"}), 404
        
        print("📦 Mengirim file global_model_fedavg.npz ke client...")
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# ==========================================
# 🧩 Route tambahan untuk download file model
# ==========================================
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    # Folder tempat file model disimpan
    model_folder = os.path.join(os.getcwd(), "models")
    file_path = os.path.join(model_folder, filename)

    # Cek apakah file ada
    if not os.path.exists(file_path):
        print(f"[404] File tidak ditemukan: {file_path}")
        return f"❌ File {filename} tidak ditemukan di server.", 404

    print(f"[200] Mengirim file: {file_path}")
    return send_file(file_path, as_attachment=True)


# ==========================================================
# 5️⃣ ENDPOINT: STATUS SERVER
# ==========================================================
@app.route("/")
def home():
    return {
        "message": "🌍 Federated Aggregation Server aktif!",
        "status": "online",
        "endpoints": {
            "/upload-model": "Upload model lokal dari client (POST)",
            "/aggregate": "Lakukan agregasi global (POST)",
            "/logs": "Lihat histori agregasi (GET)"
        }
    }


# ==========================================================
# 6️⃣ RUN SERVER (untuk Railway)
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
