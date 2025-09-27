import pandas as pd
from pathlib import Path

DATA_DIR = Path("DATASET")
OUT_FILE = DATA_DIR / "gabungan.csv"

# Baca semua CSV
df_dinsos   = pd.read_csv(DATA_DIR / "dinsos.csv")
df_dukcapil = pd.read_csv(DATA_DIR / "dukcapil.csv")
df_kemenkes = pd.read_csv(DATA_DIR / "kemenkes.csv")

# Reset index supaya baris align
df_dinsos.reset_index(drop=True, inplace=True)
df_dukcapil.reset_index(drop=True, inplace=True)
df_kemenkes.reset_index(drop=True, inplace=True)

# Gabungkan secara horizontal (union by row index)
df_all = pd.concat([df_dinsos, df_dukcapil, df_kemenkes], axis=1)

# Simpan hasil
df_all.to_csv(OUT_FILE, index=False)
print(f"Gabungan CSV disimpan di {OUT_FILE} dengan {len(df_all)} baris dan {len(df_all.columns)} kolom")
print("Kolom:", list(df_all.columns))
