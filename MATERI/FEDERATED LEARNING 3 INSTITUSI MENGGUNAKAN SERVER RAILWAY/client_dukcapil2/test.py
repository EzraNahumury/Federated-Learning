import os
import joblib
import numpy as np
import pandas as pd
from keras.layers import TFSMLayer
from colorama import Fore, Style, init
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

init(autoreset=True)

# ============================================================
# ⚙️ 1️⃣ KONFIGURASI MODEL GLOBAL
# ============================================================
THRESHOLD_MODE = "AUTO"   # "AUTO" atau "MANUAL"
THRESHOLD_MANUAL = 0.55   # jika mode MANUAL

MODEL_GLOBAL = {
    "name": "Model Global (Iterasi 1)",
    "path": "Models/saved_global_tff",
    "preproc": "Models/fitur_global_test.pkl"  
}

# ============================================================
# 🧩 2️⃣ Fungsi Preprocessing Input
# ============================================================
def preprocess_input(data, preproc):
    """Melakukan preprocessing input sesuai fitur global."""
    if not isinstance(preproc, dict):
        raise TypeError("File PKL tidak valid. Harus dictionary dengan FEATURE_COLS, mins, rng.")

    feature_cols = preproc["FEATURE_COLS"]
    mins = pd.Series(preproc["mins"]).reindex(feature_cols).fillna(0)
    rng = pd.Series(preproc["rng"]).reindex(feature_cols).replace(0, 1)

    df_final = pd.DataFrame(columns=feature_cols)
    df_final.loc[0] = 0

    numeric_cols = [
        "penghasilan", "jumlah_tanggungan", "lama_tinggal_tahun",
        "jumlah_anggota_kk", "usia_kepala_keluarga"
    ]
    for col in numeric_cols:
        if col in data and col in df_final.columns:
            df_final[col] = float(data[col])

    for col in feature_cols:
        base = col.split("_")[0]
        if base in data:
            val = str(data[base]).lower().strip()
            if f"{base}_{val}".lower() == col.lower():
                df_final[col] = 1

    df_scaled = ((df_final - mins) / rng).fillna(0.0)
    return df_scaled.astype("float32").to_numpy()

# ============================================================
# 🧠 3️⃣ Util: Cari Threshold Otomatis
# ============================================================
def auto_threshold_from_probs(y_true, y_prob):
    """Gabungkan dua pendekatan: PR/F1 dan ROC (Youden's J)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Deteksi varian probabilitas rendah
    if np.std(y_prob) < 1e-4:
        print(Fore.YELLOW + "⚠️ Variasi probabilitas sangat rendah — hasil AUTO mungkin kurang akurat." + Style.RESET_ALL)

    # PR/F1
    try:
        precision, recall, pr_thres = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve mengembalikan n+1 threshold; selaraskan panjang
        pr_thres = np.append(pr_thres, pr_thres[-1] if len(pr_thres) else 0.5)
        f1s = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
        t_f1 = pr_thres[np.nanargmax(f1s)]
    except Exception:
        t_f1 = 0.5

    # ROC / Youden's J
    try:
        fpr, tpr, roc_thres = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        t_roc = roc_thres[int(np.argmax(j_scores))]
    except Exception:
        t_roc = 0.5

    t_final = float(np.clip((t_f1 + t_roc) / 2.0, 0.0, 1.0))
    return t_final

# ============================================================
# 🧠 4️⃣ Fungsi Pengujian Model Global
# ============================================================
def run_global_test(model_path, preproc_path, cases, label):
    print(f"\n{'=' * 70}")
    print(f"🌍 TEST MODEL GLOBAL di Data {label.upper()}")
    print(f"{'=' * 70}")

    try:
        model = TFSMLayer(model_path, call_endpoint="serving_default")
        preproc = joblib.load(preproc_path)
        print("✅ Model dan preprocessing berhasil dimuat!")
    except Exception as e:
        print(Fore.RED + "❌ Gagal memuat model atau preproc!" + Style.RESET_ALL)
        print(e)
        return

    probs, labels = [], []

    for i, (data, expected) in enumerate(cases):
        try:
            X = preprocess_input(data, preproc)
            outputs = model(X, training=False)
            y_pred = list(outputs.values())[0].numpy().flatten()[0]
            probs.append(y_pred)
            labels.append(expected)
        except Exception as e:
            print(f"{Fore.RED}⚠️ Gagal memproses kasus {i + 1}:{Style.RESET_ALL}", e)

    if not probs:
        print(Fore.YELLOW + "⚠️ Tidak ada hasil prediksi valid." + Style.RESET_ALL)
        return

    probs, labels = np.array(probs), np.array(labels)

    # Penentuan threshold
    if THRESHOLD_MODE == "AUTO":
        # 1) Jika preproc sudah menyimpan threshold hasil training → gunakan
        if isinstance(preproc, dict) and "threshold" in preproc:
            threshold = float(preproc["threshold"])
        else:
            # 2) Hitung AUTO dari data uji (fallback)
            threshold = auto_threshold_from_probs(labels, probs)
    else:
        threshold = THRESHOLD_MANUAL


    benar = 0
    for i, (prob, expected) in enumerate(zip(probs, labels), 1):
        pred = int(prob >= threshold)
        icon = Fore.GREEN + "✅" if pred == expected else Fore.RED + "❌"
        if pred == expected:
            benar += 1
        print(f"[{i:02d}] Exp={expected:<2} | Pred={pred:<2} | Prob={prob:.4f} {icon}{Style.RESET_ALL}")

    acc = benar / len(cases) * 100
    print(f"\n📊 Akurasi Model Global di Data {label.upper()}: {Fore.CYAN}{acc:.2f}% ({benar}/{len(cases)}){Style.RESET_ALL}")
    print("--------------------------------------------------\n")

# ============================================================
# 🧪 5️⃣ Test Case
# ============================================================
dinsos_cases = [
    ({"penghasilan": 1800000, "jumlah_tanggungan": 3, "kondisi_rumah": "tidak layak", "status_pekerjaan": "buruh harian"}, 1),
    ({"penghasilan": 1500000, "jumlah_tanggungan": 4, "kondisi_rumah": "sederhana", "status_pekerjaan": "petani"}, 1),
    ({"penghasilan": 1000000, "jumlah_tanggungan": 5, "kondisi_rumah": "sangat sederhana", "status_pekerjaan": "tidak bekerja"}, 1),
    ({"penghasilan": 1200000, "jumlah_tanggungan": 2, "kondisi_rumah": "tidak layak", "status_pekerjaan": "buruh harian"}, 1),
    ({"penghasilan": 1750000, "jumlah_tanggungan": 3, "kondisi_rumah": "semi permanen", "status_pekerjaan": "petani"}, 1),
    ({"penghasilan": 4000000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak", "status_pekerjaan": "pegawai tetap"}, 0),
    ({"penghasilan": 5500000, "jumlah_tanggungan": 2, "kondisi_rumah": "layak", "status_pekerjaan": "karyawan tetap"}, 0),
    ({"penghasilan": 3500000, "jumlah_tanggungan": 1, "kondisi_rumah": "sederhana", "status_pekerjaan": "pegawai tetap"}, 0),
    ({"penghasilan": 5000000, "jumlah_tanggungan": 2, "kondisi_rumah": "layak", "status_pekerjaan": "PNS"}, 0),
    ({"penghasilan": 6000000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak", "status_pekerjaan": "wirausaha"}, 0),
]

dukcapil_cases = [
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "buruh", "jumlah_anggota_kk": 5, "usia_kepala_keluarga": 40}, 1),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "janda", "pekerjaan_kepala_keluarga": "petani", "jumlah_anggota_kk": 4, "usia_kepala_keluarga": 35}, 1),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "tidak bekerja", "jumlah_anggota_kk": 6, "usia_kepala_keluarga": 42}, 1),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "belum menikah", "pekerjaan_kepala_keluarga": "pegawai tetap", "jumlah_anggota_kk": 1, "usia_kepala_keluarga": 25}, 0),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "cerai", "pekerjaan_kepala_keluarga": "PNS", "jumlah_anggota_kk": 2, "usia_kepala_keluarga": 30}, 0),
    ({"nik_valid": "tidak", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "ya", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "buruh", "jumlah_anggota_kk": 4, "usia_kepala_keluarga": 38}, 0),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "petani", "jumlah_anggota_kk": 7, "usia_kepala_keluarga": 47}, 1),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "duda", "pekerjaan_kepala_keluarga": "wirausaha", "jumlah_anggota_kk": 2, "usia_kepala_keluarga": 33}, 0),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "buruh", "jumlah_anggota_kk": 5, "usia_kepala_keluarga": 36}, 1),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "belum menikah", "pekerjaan_kepala_keluarga": "pegawai tetap", "jumlah_anggota_kk": 1, "usia_kepala_keluarga": 27}, 0),
]

kemenkes_cases = [
    ({"penghasilan": 2000000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "diabetes", "status_pekerjaan": "buruh harian", "kondisi_rumah": "sederhana"}, 1),
    ({"penghasilan": 1500000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "asma", "status_pekerjaan": "petani", "kondisi_rumah": "tidak layak"}, 1),
    ({"penghasilan": 1200000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "hipertensi", "status_pekerjaan": "tidak bekerja", "kondisi_rumah": "sangat sederhana"}, 1),
    ({"penghasilan": 3500000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "pegawai tetap", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 7000000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "PNS", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 2500000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "ginjal", "status_pekerjaan": "buruh harian", "kondisi_rumah": "tidak layak"}, 1),
    ({"penghasilan": 6000000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "wirausaha", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 1000000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "TBC", "status_pekerjaan": "tidak bekerja", "kondisi_rumah": "tidak layak"}, 1),
    ({"penghasilan": 4000000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "pegawai tetap", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 1800000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "asma", "status_pekerjaan": "buruh harian", "kondisi_rumah": "semi permanen"}, 1),
]

gabungan_cases = [
    ({"penghasilan": 1800000, "jumlah_tanggungan": 5, "kondisi_rumah": "tidak layak", "status_pekerjaan": "buruh harian",
      "nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "punya_asuransi_lain": "tidak", "penyakit_kronis": "asma"}, 1),
    ({"penghasilan": 1500000, "jumlah_tanggungan": 4, "kondisi_rumah": "sederhana", "status_pekerjaan": "petani",
      "nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "punya_asuransi_lain": "tidak", "penyakit_kronis": "diabetes"}, 1),
    ({"penghasilan": 7500000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak", "status_pekerjaan": "pegawai tetap",
      "nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada"}, 0),
    ({"penghasilan": 9000000, "jumlah_tanggungan": 2, "kondisi_rumah": "mewah", "status_pekerjaan": "wirausaha",
      "nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "tidak",
      "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada"}, 0),
]

# ============================================================
# 🚀 6️⃣ Jalankan Pengujian
# ============================================================
if __name__ == "__main__":
    run_global_test(MODEL_GLOBAL["path"], MODEL_GLOBAL["preproc"], dinsos_cases, "DINSOS")
    run_global_test(MODEL_GLOBAL["path"], MODEL_GLOBAL["preproc"], dukcapil_cases, "DUKCAPIL")
    run_global_test(MODEL_GLOBAL["path"], MODEL_GLOBAL["preproc"], kemenkes_cases, "KEMENKES")
    run_global_test(MODEL_GLOBAL["path"], MODEL_GLOBAL["preproc"], gabungan_cases, "GABUNGAN")
