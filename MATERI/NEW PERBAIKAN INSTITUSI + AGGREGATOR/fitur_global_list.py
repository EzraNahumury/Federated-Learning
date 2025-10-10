import pandas as pd, joblib

print("📂 Membaca dataset dari ketiga instansi...")
dfs = [
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/dinsos_balanced.csv"),
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/dukcapil_balanced.csv"),
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/kemenkes_balanced.csv"),
]

# Gabungkan semua kolom (tanpa kolom target)
print("🔧 Menggabungkan semua fitur unik...")
all_features = pd.concat(
    [df.drop(columns=["layak_subsidi"]) for df in dfs],
    axis=0,
    ignore_index=True
)

# One-hot encoding ke seluruh fitur kategorikal
global_encoded = pd.get_dummies(all_features, drop_first=False)

# 🔧 Perbaikan penting: pastikan semua kolom bertipe numerik (float)
global_encoded = global_encoded.astype("float32")

# Daftar kolom global
feature_cols = list(global_encoded.columns)

# Hitung nilai min dan range (max - min) untuk setiap kolom
print("📊 Menghitung statistik normalisasi (min dan range)...")
mins = global_encoded.min()
rng = (global_encoded.max() - global_encoded.min()).replace(0, 1)

# Buat dictionary preprocessing global
fitur_global = {
    "FEATURE_COLS": feature_cols,
    "mins": mins.to_dict(),
    "rng": rng.to_dict()
}

# Simpan ke file
joblib.dump(fitur_global, "Models/fitur_global.pkl")

print("✅ Fitur global tersimpan di 'Models/fitur_global.pkl'")
print(f"Total fitur unik: {len(feature_cols)} kolom")
print("🔒 Format PKL sekarang berisi dictionary lengkap dengan min dan range.")
