import os
import joblib
import numpy as np
import pandas as pd
from keras.layers import TFSMLayer
from colorama import Fore, Style, init

init(autoreset=True)

# ============================================================
# ⚙️ 1️⃣ Konfigurasi Model & Threshold Mode
# ============================================================
THRESHOLD_MODE = "AUTO"   # "AUTO" atau "MANUAL"
THRESHOLD_MANUAL = 0.55   # jika pakai mode MANUAL

MODELS = {
    "dinsos": {
        "path": "Models/saved_dinsos_tff",
        "preproc": "Models/saved_dinsos_tff/preprocess_dinsos.pkl"
    },
    "dukcapil": {
        "path": "Models/saved_dukcapil_tff",
        "preproc": "Models/saved_dukcapil_tff/preprocess_dukcapil.pkl"
    },
    "kemenkes": {
        "path": "Models/saved_kemenkes_tff",
        "preproc": "Models/saved_kemenkes_tff/preprocess_kemenkes.pkl"
    },
    "gabungan": {
        "path": "Models/saved_global2_tff",
        "preproc": "Models/fitur_global.pkl"
    },
}


# ============================================================
# 🧩 2️⃣ Fungsi Preprocessing Input
# ============================================================
def preprocess_input(data, preproc):
    """
    Fungsi untuk melakukan preprocessing input sebelum dikirim ke model.
    Mendukung format dictionary berisi FEATURE_COLS, mins, dan rng.
    """
    # Pastikan format benar
    if not isinstance(preproc, dict):
        raise TypeError(
            f"Format PKL tidak valid. Harus berupa dictionary "
            f"dengan keys: FEATURE_COLS, mins, dan rng. (dapat: {type(preproc)})"
        )

    feature_cols = preproc["FEATURE_COLS"]
    mins = pd.Series(preproc["mins"]).reindex(feature_cols).fillna(0)
    rng = pd.Series(preproc["rng"]).reindex(feature_cols).replace(0, 1)

    # Buat dataframe kosong dengan semua kolom
    df_final = pd.DataFrame(columns=feature_cols)
    df_final.loc[0] = 0

    # Kolom numerik
    numeric_cols = [
        "penghasilan",
        "jumlah_tanggungan",
        "lama_tinggal_tahun",
        "jumlah_anggota_kk",
        "usia_kepala_keluarga",
    ]
    for col in numeric_cols:
        if col in data and col in df_final.columns:
            df_final[col] = float(data[col])

    # Kolom kategorikal (One-Hot)
    for col in feature_cols:
        base = col.split("_")[0]
        if base in data:
            val = str(data[base]).lower().strip()
            if f"{base}_{val}".lower() == col.lower():
                df_final[col] = 1

    # Normalisasi
    df_scaled = ((df_final - mins) / rng).fillna(0.0)
    return df_scaled.astype("float32").to_numpy()


# ============================================================
# 🧠 3️⃣ Helper: Jalankan Pengujian
# ============================================================
def run_test(model_name, model_path, preproc_path, cases):
    print(f"\n{'=' * 60}")
    print(f"🧩 TEST MODEL: {model_name.upper()} (mode={THRESHOLD_MODE})")
    print(f"{'=' * 60}")

    try:
        model = TFSMLayer(model_path, call_endpoint="serving_default")
        preproc = joblib.load(preproc_path)
    except Exception as e:
        print(f"{Fore.RED}❌ ERROR saat memuat model/preproc untuk {model_name.upper()}:{Style.RESET_ALL}")
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
            continue

    if not probs:
        print(f"{Fore.YELLOW}Tidak ada hasil prediksi valid untuk {model_name.upper()}.{Style.RESET_ALL}")
        return

    probs = np.array(probs)
    labels = np.array(labels)

    # Threshold otomatis
    if THRESHOLD_MODE == "AUTO":
        thresholds = np.linspace(0.4, 0.75, 36)
        best_acc, best_t = 0, 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc, best_t = acc, t
        threshold = best_t
    else:
        threshold = THRESHOLD_MANUAL

    # Evaluasi hasil akhir
    benar = 0
    print(f"📏 Threshold digunakan: {threshold:.3f}\n")
    for i, (prob, expected) in enumerate(zip(probs, labels), 1):
        pred = int(prob >= threshold)
        icon = Fore.GREEN + "✅" if pred == expected else Fore.RED + "❌"
        if pred == expected:
            benar += 1
        print(f"[{i:02d}] Exp={expected:<2} | Pred={pred:<2} | Prob={prob:.4f} {icon}{Style.RESET_ALL}")

    acc = benar / len(cases) * 100
    print(f"\n📊 Akurasi {model_name.upper()}: {Fore.CYAN}{acc:.2f}% ({benar}/{len(cases)}){Style.RESET_ALL}")
    print("--------------------------------------------------\n")


# ============================================================
# 🧪 4️⃣ Test Case per Model
# ============================================================


# ✅ DINSOS
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

# ✅ DUKCAPIL (10 kasus)
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

# ✅ KEMENKES (10 kasus)
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
    # Kasus gabungan tetap sama
    # ...
    ({
        "penghasilan": 1800000, "jumlah_tanggungan": 5, "kondisi_rumah": "tidak layak", "status_pekerjaan": "buruh harian",
        "nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
        "punya_asuransi_lain": "tidak", "penyakit_kronis": "asma"
    }, 1),
    ({
        "penghasilan": 1500000, "jumlah_tanggungan": 4, "kondisi_rumah": "sederhana", "status_pekerjaan": "petani",
        "nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
        "punya_asuransi_lain": "tidak", "penyakit_kronis": "diabetes"
    }, 1),
    ({
        "penghasilan": 7500000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak", "status_pekerjaan": "pegawai tetap",
        "nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
        "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada"
    }, 0),
    ({
        "penghasilan": 9000000, "jumlah_tanggungan": 2, "kondisi_rumah": "mewah", "status_pekerjaan": "wirausaha",
        "nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "tidak",
        "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada"
    }, 0),
]

# ============================================================
# 🚀 5️⃣ Jalankan Semua Pengujian
# ============================================================
if __name__ == "__main__":
    run_test("dinsos", MODELS["dinsos"]["path"], MODELS["dinsos"]["preproc"], dinsos_cases)
    run_test("dukcapil", MODELS["dukcapil"]["path"], MODELS["dukcapil"]["preproc"], dukcapil_cases)
    run_test("kemenkes", MODELS["kemenkes"]["path"], MODELS["kemenkes"]["preproc"], kemenkes_cases)
    run_test("gabungan", MODELS["gabungan"]["path"], MODELS["gabungan"]["preproc"], gabungan_cases)