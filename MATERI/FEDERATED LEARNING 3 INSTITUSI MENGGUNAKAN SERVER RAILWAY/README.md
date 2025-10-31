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

## ⚙️ Struktur Folder
├── 📂 FEATURE COL               → berisi file fitur global (fitur_global.pkl)
├── 📂 Flask                     → aplikasi Flask untuk prediksi berbasis web
├── 📂 client_dinsos             → pelatihan lokal DINSOS
├── 📂 client_dinsos2            → pelatihan iterasi ke-2 DINSOS
├── 📂 client_dukcapil           → pelatihan lokal DUKCAPIL
├── 📂 client_dukcapil2          → pelatihan iterasi ke-2 DUKCAPIL
├── 📂 client_kemenkes           → pelatihan lokal KEMENKES
├── 📂 client_kemenkes2          → pelatihan iterasi ke-2 KEMENKES
├── 📂 federated_server          → server Railway untuk agregasi FedAvg
├── 📄 DOKUMENTASI ARCHITECTURE.pdf
└── 📄 DOKUMENTASI FEDERATED LEARNING 3 LEMBAGA.pdf
