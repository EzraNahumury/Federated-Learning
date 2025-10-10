import pandas as pd, joblib

print("📂 Membaca dataset dari ketiga instansi...")
dfs = [
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/dinsos_balanced.csv"),
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/dukcapil_balanced.csv"),
    pd.read_csv("BALANCED_DATASET_100K_REALISTIC/kemenkes_balanced.csv"),
]

# Gabungkan semua kolom (tanpa kolom target)
print("🔧 Menggabungkan semua fitur unik...")
all_features = pd.concat([df.drop(columns=["layak_subsidi"]) for df in dfs], axis=0)
global_encoded = pd.get_dummies(all_features, drop_first=False)
global_cols = list(global_encoded.columns)

# Simpan ke file
joblib.dump(global_cols, "Models/fitur_global_dict.pkl")
print(f"✅ Fitur global tersimpan di 'Models/fitur_global.pkl'")
print(f"Total fitur unik: {len(global_cols)} kolom")
