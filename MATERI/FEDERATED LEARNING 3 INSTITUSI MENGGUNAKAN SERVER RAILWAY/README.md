#  Federated Learning – Prediksi Kelayakan Subsidi (3 Institusi)

## 📘 Deskripsi
Proyek ini merupakan implementasi **Federated Learning (FL)** yang melibatkan tiga instansi pemerintah:  
- **DINSOS** (Program Bantuan Pangan Non Tunai / Kartu Sembako)  
- **DUKCAPIL** (Data Kependudukan)  
- **KEMENKES** (Kartu Indonesia Sehat / KIS)  

Setiap instansi melatih model lokal secara terpisah menggunakan data internal mereka.  
Bobot hasil pelatihan dikirim ke server pusat untuk digabungkan menggunakan metode **Federated Averaging (FedAvg)**.  
Hasilnya adalah **model global** yang mampu melakukan prediksi kelayakan subsidi lintas instansi tanpa harus membagikan data mentah antar lembaga.

---



## 🧩 Cara Clone Repository

```bash
git clone --no-checkout https://github.com/EzraNahumury/Federated-Learning.git
cd Federated-Learning
git sparse-checkout init --cone
git sparse-checkout set "MATERI/FEDERATED LEARNING 3 INSTITUSI MENGGUNAKAN SERVER RAILWAY"
git checkout main





====================================================================================================================================================================

🌸 **Panduan Instalasi TensorFlow Federated (TFF) di Windows (WSL)**


## 📘 Deskripsi
Panduan ini berisi langkah-langkah lengkap untuk menginstal dan menyiapkan **TensorFlow Federated (TFF)** di komputer berbasis **Windows**.  
Karena TensorFlow Federated lebih stabil di sistem operasi **Linux**, maka digunakan **WSL (Windows Subsystem for Linux)** untuk membuat lingkungan pengembangan yang kompatibel.

Dengan panduan ini, kamu dapat menjalankan proyek **Federated Learning (FL)** secara lokal tanpa perlu migrasi ke sistem operasi lain.

---

## 🧩 1️⃣ Instalasi WSL / Ubuntu di Windows

1. Buka **Microsoft Store**, cari **Ubuntu**, lalu klik **Install**.  
   Jalankan Ubuntu setelah instalasi selesai untuk membuat username dan password Linux.
2. Pastikan terminal **VS Code** sudah menggunakan **WSL: Ubuntu**  
   (atau bisa juga **Git Bash / PowerShell**, namun disarankan WSL).  
3. Atur terminal default di VS Code:
   - Klik ikon `˅` di pojok kanan atas terminal.  
   - Pilih **Select Default Profile**.  
   - Pilih **WSL**, lalu restart VS Code.

> 💡 Gunakan **Ubuntu 22.04** untuk hasil terbaik dan kompatibilitas penuh dengan TensorFlow.

---

## 🧮 2️⃣ Cek Versi Python

Pastikan Python sudah terinstal di dalam WSL dengan perintah berikut:

```bash
python3 --version
```

Jika Python belum terinstal, lanjut ke langkah berikut untuk memasang versi yang sesuai.

> ⚠️ TensorFlow Federated hanya stabil di **Python 3.11**  
> Jangan gunakan **Python 3.12** karena belum sepenuhnya didukung.

---

## 🔢 3️⃣ Instal Python 3.11 di WSL

Jalankan perintah berikut di terminal Ubuntu (WSL):

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y
```

Setelah selesai, verifikasi instalasi:

```bash
python3.11 --version
```

Jika muncul seperti `Python 3.11.x`, berarti instalasi berhasil ✅

---

## 🌐 4️⃣ Buat Virtual Environment

Langkah berikutnya adalah membuat *virtual environment* agar setiap proyek terisolasi dari sistem global.

```bash
python3.11 -m venv venv
source venv/bin/activate
```

> Jika environment aktif, terminal akan menampilkan prefix `(venv)` di depan nama folder.

Untuk menonaktifkan environment:

```bash
deactivate
```

---

## 🔧 5️⃣ Install TensorFlow Federated (TFF)

Setelah environment aktif, jalankan perintah berikut untuk menginstal TensorFlow Federated versi terbaru:

```bash
pip install --upgrade pip
pip install --upgrade tensorflow-federated
```

> Jika koneksi internet lambat, instalasi bisa memakan waktu beberapa menit karena akan mengunduh dependensi TensorFlow.

---

## ✅ 6️⃣ Verifikasi Instalasi

Cek apakah TensorFlow Federated berhasil diinstal dengan benar:

```bash
python
>>> import tensorflow_federated as tff
>>> tff.federated_sum
```

Jika tidak muncul error, maka instalasi sudah sukses 🎉  
Ketik `exit()` untuk keluar dari mode Python.

---

## 🧪 7️⃣ Uji Coba Program TFF Sederhana

Untuk memastikan TensorFlow Federated dapat berfungsi, jalankan contoh berikut:

```python
import tensorflow_federated as tff

@tff.federated_computation
def average_example():
    return tff.federated_mean([1.0, 2.0, 3.0])

print(average_example())
```

Output yang diharapkan:

```
2.0
```

Berarti TensorFlow Federated sudah aktif sepenuhnya 🚀

---

## 📎 8️⃣ Rangkuman Instalasi

| Komponen | Versi / Tools yang Disarankan |
|-----------|-------------------------------|
| **OS** | Windows 10/11 |
| **Linux Environment** | Ubuntu 22.04 (via WSL) |
| **Python** | 3.11 |
| **TensorFlow Federated** | Versi terbaru |
| **IDE** | Visual Studio Code |
| **Terminal** | WSL / Bash |

---

## 💡 9️⃣ Tips Tambahan

- Jalankan semua instalasi dalam **environment (venv)**, bukan global system.
- Untuk membuka kembali environment yang pernah dibuat:
  ```bash
  source venv/bin/activate
  ```
- Untuk memperbarui TensorFlow Federated:
  ```bash
  pip install --upgrade tensorflow-federated
  ```
- Simpan file proyek di folder WSL (misalnya `/home/username/project`) agar performa lebih cepat dibanding di `C:\`.

---

## ⚡ 🔟 Troubleshooting Umum

| Masalah | Penyebab | Solusi |
|----------|-----------|--------|
| `ImportError: No module named 'tensorflow_federated'` | Environment belum aktif | Jalankan `source venv/bin/activate` |
| `ERROR: Could not find a version that satisfies the requirement tensorflow-federated` | Versi Python tidak cocok | Gunakan Python 3.11 |
| `ModuleNotFoundError: No module named 'grpc'` | Dependensi belum lengkap | Jalankan `pip install grpcio grpcio-tools` |
| Error CUDA / GPU | TensorFlow Federated tidak memakai GPU | Abaikan, karena TFF berjalan di CPU |
| `Permission denied` saat install | Tidak pakai sudo di Ubuntu | Gunakan perintah `sudo` jika diperlukan |

---

## 📂 11️⃣ Struktur Folder (Contoh Proyek Federated Learning)

```
📦 FEDERATED LEARNING 3 INSTITUSI MENGGUNAKAN SERVER RAILWAY
├── 📂 FEATURE COL               → berisi file fitur global (fitur_global.pkl)
├── 📂 Flask                     → aplikasi Flask untuk prediksi berbasis web
├── 📂 client_dinsos             → pelatihan lokal DINSOS
├── 📂 client_dukcapil           → pelatihan lokal DUKCAPIL
├── 📂 client_kemenkes           → pelatihan lokal KEMENKES
├── 📂 federated_server          → server Railway untuk agregasi FedAvg
└── 📄 DOKUMENTASI INSTALLASI TFF.pdf
```

---

## 🧩 12️⃣ Clone Repository (Hanya Folder Proyek)

Jika hanya ingin mengunduh folder proyek TFF tanpa seluruh repo utama:

```bash
git clone --no-checkout https://github.com/EzraNahumury/Federated-Learning.git
cd Federated-Learning
git sparse-checkout init --cone
git sparse-checkout set "MATERI/FEDERATED LEARNING 3 INSTITUSI MENGGUNAKAN SERVER RAILWAY"
git checkout main
```

> ✅ Cara ini hanya mengunduh folder tersebut saja tanpa seluruh isi repo (lebih ringan).

---

## 🧾 Penutup

Selamat! 🎉  
Kamu sudah berhasil menyiapkan **lingkungan TensorFlow Federated (TFF)** di Windows menggunakan **WSL Ubuntu**.  
Lingkungan ini sudah siap digunakan untuk pengembangan proyek **Federated Learning** baik secara lokal maupun terhubung ke server seperti **Railway** atau **Google Cloud**.

---

📄 **Dibuat oleh:**  
**Ezra K. Nahumury**  
*Federated Learning – 3 Institusi (DINSOS, DUKCAPIL, KEMENKES)*  
Tahun 2025
