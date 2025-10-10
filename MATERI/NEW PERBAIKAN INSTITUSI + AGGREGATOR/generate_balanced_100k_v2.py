import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# ==============================
# KONFIGURASI
# ==============================
SAVE_DIR = Path("BALANCED_DATASET_100K_REALISTIC")
SAVE_DIR.mkdir(exist_ok=True)
N_ROWS = 100_000
RATIO = 0.5  # 50% layak : 50% tidak layak
NOISE_PCT = 0.05  # 5% data dibalik label untuk mensimulasikan inkonsistensi dunia nyata


# ==============================
# HELPER
# ==============================
def flip_labels(df, noise_pct):
    """Balik sebagian kecil label untuk simulasi kesalahan labeling di dunia nyata."""
    n_flip = int(len(df) * noise_pct)
    flip_idx = np.random.choice(df.index, n_flip, replace=False)
    df.loc[flip_idx, "layak_subsidi"] = 1 - df.loc[flip_idx, "layak_subsidi"]
    return df


# ==============================
# 1️⃣ DINSOS
# ==============================
def generate_dinsos(n, ratio):
    n_layak = int(n * ratio)
    n_tidak = n - n_layak

    layak = pd.DataFrame({
        "penghasilan": np.random.randint(500_000, 2_000_000, n_layak),
        "jumlah_tanggungan": np.random.randint(2, 8, n_layak),
        "kondisi_rumah": np.random.choice(["tidak layak", "sederhana"], n_layak, p=[0.65, 0.35]),
        "status_pekerjaan": np.random.choice(["buruh harian", "petani", "tidak bekerja"], n_layak, p=[0.5, 0.3, 0.2]),
        "pendidikan": np.random.choice(["SD", "SMP", "SMA"], n_layak, p=[0.5, 0.3, 0.2]),
        "lama_tinggal_tahun": np.random.randint(5, 25, n_layak),
        "layak_subsidi": 1
    })

    tidak = pd.DataFrame({
        "penghasilan": np.random.randint(3_500_000, 15_000_000, n_tidak),
        "jumlah_tanggungan": np.random.randint(0, 3, n_tidak),
        "kondisi_rumah": np.random.choice(["layak", "mewah"], n_tidak, p=[0.7, 0.3]),
        "status_pekerjaan": np.random.choice(["PNS", "pegawai tetap", "wirausaha"], n_tidak, p=[0.4, 0.4, 0.2]),
        "pendidikan": np.random.choice(["SMA", "D3", "S1"], n_tidak, p=[0.4, 0.3, 0.3]),
        "lama_tinggal_tahun": np.random.randint(1, 10, n_tidak),
        "layak_subsidi": 0
    })

    df = pd.concat([layak, tidak], ignore_index=True)
    return flip_labels(df, NOISE_PCT)


# ==============================
# 2️⃣ DUKCAPIL
# ==============================
def generate_dukcapil(n, ratio):
    n_layak = int(n * ratio)
    n_tidak = n - n_layak

    def rand_bool(true_prob):
        return np.random.choice(["ya", "tidak"], p=[true_prob, 1 - true_prob])

    layak = pd.DataFrame({
        "nik_valid": [rand_bool(0.95) for _ in range(n_layak)],
        "memiliki_kk": [rand_bool(0.95) for _ in range(n_layak)],
        "domisili_tetap": [rand_bool(0.9) for _ in range(n_layak)],
        "data_ganda": [rand_bool(0.05) for _ in range(n_layak)],
        "masuk_dtks": [rand_bool(0.9) for _ in range(n_layak)],
        "status_perkawinan": np.random.choice(["menikah", "janda", "duda"], n_layak, p=[0.8, 0.1, 0.1]),
        "pekerjaan_kepala_keluarga": np.random.choice(["buruh", "petani", "tidak bekerja"], n_layak, p=[0.4, 0.3, 0.3]),
        "jumlah_anggota_kk": np.random.randint(3, 7, n_layak),
        "usia_kepala_keluarga": np.random.randint(25, 60, n_layak),
        "layak_subsidi": 1
    })

    tidak = pd.DataFrame({
        "nik_valid": [rand_bool(0.4) for _ in range(n_tidak)],
        "memiliki_kk": [rand_bool(0.5) for _ in range(n_tidak)],
        "domisili_tetap": [rand_bool(0.5) for _ in range(n_tidak)],
        "data_ganda": [rand_bool(0.8) for _ in range(n_tidak)],
        "masuk_dtks": [rand_bool(0.2) for _ in range(n_tidak)],
        "status_perkawinan": np.random.choice(["belum menikah", "cerai"], n_tidak),
        "pekerjaan_kepala_keluarga": np.random.choice(["PNS", "pegawai tetap", "wirausaha"], n_tidak, p=[0.3, 0.4, 0.3]),
        "jumlah_anggota_kk": np.random.randint(1, 3, n_tidak),
        "usia_kepala_keluarga": np.random.randint(25, 60, n_tidak),
        "layak_subsidi": 0
    })

    df = pd.concat([layak, tidak], ignore_index=True)
    return flip_labels(df, NOISE_PCT)


# ==============================
# 3️⃣ KEMENKES
# ==============================
def generate_kemenkes(n, ratio):
    n_layak = int(n * ratio)
    n_tidak = n - n_layak

    layak = pd.DataFrame({
        "penghasilan": np.random.randint(500_000, 2_500_000, n_layak),
        "punya_asuransi_lain": np.random.choice(["tidak", "ya"], n_layak, p=[0.9, 0.1]),
        "penyakit_kronis": np.random.choice(["diabetes", "asma", "hipertensi", "ginjal kronis"], n_layak),
        "status_pekerjaan": np.random.choice(["buruh harian", "petani", "tidak bekerja"], n_layak),
        "kondisi_rumah": np.random.choice(["sederhana", "tidak layak"], n_layak, p=[0.4, 0.6]),
        "status_gizi": np.random.choice(["baik", "kurang", "stunting"], n_layak, p=[0.5, 0.3, 0.2]),
        "layak_subsidi": 1
    })

    tidak = pd.DataFrame({
        "penghasilan": np.random.randint(4_000_000, 15_000_000, n_tidak),
        "punya_asuransi_lain": np.random.choice(["ya", "tidak"], n_tidak, p=[0.85, 0.15]),
        "penyakit_kronis": np.random.choice(["tidak ada", "ringan"], n_tidak, p=[0.9, 0.1]),
        "status_pekerjaan": np.random.choice(["PNS", "pegawai tetap", "wirausaha"], n_tidak),
        "kondisi_rumah": np.random.choice(["layak", "mewah"], n_tidak, p=[0.7, 0.3]),
        "status_gizi": np.random.choice(["baik", "gizi buruk"], n_tidak, p=[0.8, 0.2]),
        "layak_subsidi": 0
    })

    df = pd.concat([layak, tidak], ignore_index=True)
    return flip_labels(df, NOISE_PCT)


# ==============================
# 4️⃣ GABUNGAN
# ==============================
def generate_gabungan(n, ratio):
    n_layak = int(n * ratio)
    n_tidak = n - n_layak

    layak = pd.DataFrame({
        "penghasilan": np.random.randint(500_000, 2_500_000, n_layak),
        "jumlah_tanggungan": np.random.randint(3, 7, n_layak),
        "kondisi_rumah": np.random.choice(["sederhana", "tidak layak"], n_layak, p=[0.4, 0.6]),
        "status_pekerjaan": np.random.choice(["buruh harian", "petani"], n_layak),
        "nik_valid": np.random.choice(["ya", "tidak"], n_layak, p=[0.9, 0.1]),
        "memiliki_kk": np.random.choice(["ya", "tidak"], n_layak, p=[0.9, 0.1]),
        "masuk_dtks": np.random.choice(["ya", "tidak"], n_layak, p=[0.9, 0.1]),
        "punya_asuransi_lain": np.random.choice(["tidak", "ya"], n_layak, p=[0.9, 0.1]),
        "penyakit_kronis": np.random.choice(["asma", "diabetes", "hipertensi"], n_layak),
        "layak_subsidi": 1
    })

    tidak = pd.DataFrame({
        "penghasilan": np.random.randint(5_000_000, 12_000_000, n_tidak),
        "jumlah_tanggungan": np.random.randint(0, 3, n_tidak),
        "kondisi_rumah": np.random.choice(["layak", "mewah"], n_tidak, p=[0.7, 0.3]),
        "status_pekerjaan": np.random.choice(["PNS", "pegawai tetap"], n_tidak),
        "nik_valid": np.random.choice(["ya", "tidak"], n_tidak, p=[0.2, 0.8]),
        "memiliki_kk": np.random.choice(["ya", "tidak"], n_tidak, p=[0.3, 0.7]),
        "masuk_dtks": np.random.choice(["ya", "tidak"], n_tidak, p=[0.2, 0.8]),
        "punya_asuransi_lain": np.random.choice(["ya", "tidak"], n_tidak, p=[0.8, 0.2]),
        "penyakit_kronis": np.random.choice(["tidak ada", "ringan"], n_tidak, p=[0.9, 0.1]),
        "layak_subsidi": 0
    })

    df = pd.concat([layak, tidak], ignore_index=True)
    return flip_labels(df, NOISE_PCT)


# ==============================
# 5️⃣ SIMPAN SEMUA CSV
# ==============================
datasets = {
    "dinsos_balanced.csv": generate_dinsos,
    "dukcapil_balanced.csv": generate_dukcapil,
    "kemenkes_balanced.csv": generate_kemenkes,
    "gabungan_balanced.csv": generate_gabungan,
}

for name, func in datasets.items():
    print(f"🚀 Membuat {name} ...")
    df = func(N_ROWS, RATIO)
    df.to_csv(SAVE_DIR / name, index=False)
    n1 = df['layak_subsidi'].sum()
    n0 = len(df) - n1
    print(f"✅ {name}: {len(df)} baris | Layak={n1:,} | Tidak Layak={n0:,}\n")
