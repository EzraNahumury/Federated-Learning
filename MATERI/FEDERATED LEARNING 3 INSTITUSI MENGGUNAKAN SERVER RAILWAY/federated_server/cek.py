import os
import numpy as np
import requests

SERVER_URL = "https://federatedserver-production.up.railway.app"
TEMP_DIR = "temp_models"
os.makedirs(TEMP_DIR, exist_ok=True)

# ============================================================
# 🧩 1️⃣ Ambil daftar file di server
# ============================================================
print("📡 Mengambil daftar file model dari server...")
res = requests.get(f"{SERVER_URL}/logs")
if res.status_code != 200:
    print("❌ Gagal ambil daftar file:", res.text)
    exit()

data = res.json()
files = data.get("files", [])
npz_files = [f for f in files if f.endswith("_weights.npz")]
print("📂 File model ditemukan di server:", npz_files)

# ============================================================
# 📥 2️⃣ Unduh dan cek isi layer tiap file
# ============================================================
for filename in npz_files:
    print(f"\n🔽 Mengunduh {filename} dari server...")
    url = f"{SERVER_URL}/models/{filename}"  # sesuaikan jika endpoint berbeda
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Gagal download {filename}: {r.text}")
        continue

    local_path = os.path.join(TEMP_DIR, filename)
    with open(local_path, "wb") as f:
        f.write(r.content)
    print(f"✅ File {filename} disimpan ke lokal sementara.")

    # Baca isi layer
    npz = np.load(local_path, allow_pickle=True)
    shapes = [npz[key].shape for key in npz.files]
    print(f"📘 {filename} — {len(shapes)} tensor")
    for i, s in enumerate(shapes):
        print(f"   Layer {i+1}: {s}")
