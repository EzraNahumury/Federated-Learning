import os
import requests

# ============================================================
# ⚙️ KONFIGURASI
# ============================================================
SERVER_URL = "https://federatedserver-production.up.railway.app"
DOWNLOAD_DIR = "models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ============================================================
# 📡 1️⃣ Ambil daftar file dari server
# ============================================================
print("📡 Mengambil daftar file model dari server...")
res = requests.get(f"{SERVER_URL}/logs")
if res.status_code != 200:
    print(f"❌ Gagal mengambil daftar file: {res.status_code} - {res.text}")
    exit()

data = res.json()
server_files = data.get("files", [])
print(f"📂 File di server: {server_files}")

# Filter hanya file client model
target_files = [f for f in server_files if f.endswith("_weights.npz")]
if not target_files:
    print("❌ Tidak ditemukan file *_weights.npz di server!")
    exit()

print(f"🎯 File yang akan diunduh: {target_files}")

# ============================================================
# 📥 2️⃣ Unduh setiap file satu per satu
# ============================================================
for filename in target_files:
    print(f"\n🔽 Mengunduh {filename} ...")

    # Biasanya file tersimpan di folder 'models' di server,
    # tapi tidak ada route /models/<filename> terbuka, jadi kita gunakan endpoint umum.
    # Endpoint '/download/<filename>' harus tersedia di app.py server kamu.
    # Jika belum, pastikan app.py punya route:
    #   @app.route('/download/<path:filename>')
    #   def download_file(filename):
    #       return send_file(os.path.join('models', filename), as_attachment=True)
    #
    # Jika sudah ada, gunakan:
    url = f"{SERVER_URL}/download/{filename}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            save_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"✅ File berhasil disimpan: {save_path}")
        else:
            print(f"❌ Gagal download {filename}: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error koneksi saat download {filename}: {e}")

print("\n✅ Semua file model selesai diunduh ke folder 'models/'")
