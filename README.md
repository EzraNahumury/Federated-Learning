# 📚 Federated Learning Roadmap

[![Python](https://img.shields.io/badge/Python-3.10%2F3.11-blue)](https://www.python.org/)  
[![TensorFlow Federated](https://img.shields.io/badge/TensorFlow-Federated-orange)](https://www.tensorflow.org/federated)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

Repositori ini berisi lesson plan, catatan, eksperimen, dan dokumentasi pembelajaran **Federated Learning (FL)** menggunakan **TensorFlow Federated (TFF)**.  
Tujuannya adalah memahami konsep dasar, arsitektur, implementasi, serta integrasi dengan **Differential Privacy (DP)** dalam sebuah pipeline end-to-end.  

---

## 🚀 Struktur Pembelajaran

### 📌 Minggu 1 – Dasar Federated Learning
- **1.1** Pengenalan FL → ringkasan perbandingan centralized vs federated + use case  
- **1.2** Arsitektur FL → diagram server-client  
- **1.3** Instalasi TFF → setup & hello world  
- **1.4** Federated Primitives (`tff.federated_map`, `tff.federated_mean`)  
- **1.5** Review & Dokumentasi  

### 📌 Minggu 2 – Federated Learning untuk Image Data
- **2.1** FL dengan dataset EMNIST  
- **2.2** Federated Averaging Algorithm → ubah jumlah client & epochs  
- **2.3** Evaluasi & Plotting → grafik akurasi/loss per round  
- **2.4** Modifikasi Model → CNN custom sederhana vs default  
- **2.5** Review & Dokumentasi  

### 📌 Minggu 3 – Dataset Tabular
- **3.1** Dataset dummy subsidi (Dinsos, Dukcapil, Kemenkes) → CSV per client  
- **3.2** Model Custom (LogReg/NN) → wrap ke `tff.learning.from_keras_model`  
- **3.3** Training FL Tabular → 5–10 round + logging akurasi  
- **3.4** Bandingkan FL vs Centralized → tabel akurasi & risiko privasi  
- **3.5** Review & Dokumentasi  

### 📌 Minggu 4 – Differential Privacy (DP)
- **4.1** Konsep DP → noise, ε (epsilon), trade-off akurasi  
- **4.2** Integrasi TF Privacy ke Model → DP-SGD optimizer  
- **4.3** Evaluasi Trade-off → plot hasil akurasi vs ε  
- **4.4** Finalisasi Prototipe → pipeline end-to-end FL + DP  
- **4.5** Laporan & Video Demo  

---


## 🛠️ Tools & Library
- **Python** 3.10 / 3.11  
- **TensorFlow Federated (TFF)**  
- **TensorFlow Privacy**  
- **Pandas, NumPy, Scikit-learn**  
- **Matplotlib** (visualisasi)  
- **VS Code, GitHub, OBS Studio**  

---

## 📌 Output yang Diharapkan
1. Ringkasan & dokumentasi konsep FL + DP  
2. Notebook eksperimen bersih & dapat direplikasi  
3. Perbandingan performa model Federated vs Centralized  
4. Pipeline end-to-end FL + DP  
5. Laporan akhir + video demo (5 menit)  

---

## 📜 Lisensi
Repositori ini menggunakan lisensi **MIT** – silakan gunakan, modifikasi, dan kembangkan.  

