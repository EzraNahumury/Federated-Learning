import pandas as pd, joblib

print(" Membaca dataset dari ketiga instansi...")
dfs = [
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/dinsos_balanced.csv"),
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/dukcapil_balanced.csv"),
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/kemenkes_balanced.csv"),
]

# ============================================================
# 1.  Gabungkan semua fitur unik dari ketiga instansi
# ============================================================
print(" Menggabungkan semua fitur unik dari ketiga instansi...")
all_features = pd.concat(
    [df.drop(columns=["layak_subsidi"]) for df in dfs],
    axis=0,
    ignore_index=True
)

# ============================================================
# 2.  One-hot encoding untuk seluruh kolom kategorikal
# ============================================================
global_encoded = pd.get_dummies(all_features, drop_first=False)

# Pastikan semua kolom numerik bertipe float32
global_encoded = global_encoded.astype("float32")

# Ambil daftar kolom global
feature_cols = list(global_encoded.columns)

print(f" Total fitur unik: {len(feature_cols)} kolom ditemukan.")

# ============================================================
# 3. Hitung statistik normalisasi (min dan range)
# ============================================================
print(" Menghitung statistik normalisasi (min dan range) untuk setiap fitur...")
mins = global_encoded.min()
rng = (global_encoded.max() - global_encoded.min()).replace(0, 1)

# ============================================================
# 4. Buat dictionary preprocessing global
# ============================================================
fitur_global = {
    "FEATURE_COLS": feature_cols,
    "mins": mins.to_dict(),
    "rng": rng.to_dict()
}

# ============================================================
# 5.  Simpan hasil ke file .pkl
# ============================================================
joblib.dump(fitur_global, "Models/fitur_global_test.pkl")

print(" Fitur global (DICTIONARY) tersimpan di 'Models/fitur_global.pkl'")
print(" Format PKL berisi:")
print("   - FEATURE_COLS : daftar semua kolom fitur global")
print("   - mins          : nilai minimum tiap fitur")
print("   - rng           : range (max - min) tiap fitur")
