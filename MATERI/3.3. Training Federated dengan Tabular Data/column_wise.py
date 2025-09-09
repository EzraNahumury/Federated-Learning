import pandas as pd

# Load CSV per instansi
dinsos = pd.read_csv("dinsos_100.csv")
dukcapil = pd.read_csv("dukcapil_100.csv")
kemenkes = pd.read_csv("kemenkes_100.csv")


# Pastikan jumlah baris sama
assert len(dinsos) == len(dukcapil) == len(kemenkes), "Jumlah baris dataset berbeda!"

# Gabung berdasarkan index (side by side)
gabungan = pd.concat([dinsos.reset_index(drop=True),
                      dukcapil.reset_index(drop=True),
                      kemenkes.reset_index(drop=True)], axis=1)

# Simpan ke CSV baru
gabungan.to_csv("dataset_gabungan.csv", index=False)