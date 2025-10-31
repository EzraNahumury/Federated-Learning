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


