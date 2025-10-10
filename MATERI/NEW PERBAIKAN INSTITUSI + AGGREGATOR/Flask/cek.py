import joblib
import pandas as pd

path = "Models/saved_dinsos_tff/preprocess_dinsos.pkl"

print("🔍 Membaca file:", path)
preproc = joblib.load(path)

print("\n=== KUNCI YANG ADA DALAM FILE ===")
print(list(preproc.keys()))

# tampilkan ringkasan isi tiap bagian
for key, val in preproc.items():
    print(f"\n--- {key.upper()} ---")
    print("Tipe:", type(val))
    if isinstance(val, (list, tuple)):
        print(f"Panjang: {len(val)} | Contoh: {val[:5]}")
    elif isinstance(val, dict):
        print(f"Banyak key: {len(val)} | Contoh keys: {list(val.keys())[:5]}")
    elif isinstance(val, (pd.Series, pd.DataFrame)):
        print(val.head())
    else:
        print(val)
