# labeled_gabungan.py
import pandas as pd
from pathlib import Path

DATA_IN = Path("DATASET") / "gabungan.csv"
DATA_OUT = Path("NEW_DATASET") / "gabungan_labeled.csv"
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Aturan Gabungan
# -----------------------------
KRONIS_SET = {"kronis", "jantung", "diabetes", "asma", "disabilitas"}

def label_gabungan(row: pd.Series) -> int:
    # Ambil nilai
    jt = int(row.get("jumlah_tanggungan", 0) or 0)
    ph = float(row.get("penghasilan", 0) or 0)
    kr = str(row.get("kondisi_rumah", "")).strip().lower()

    u  = int(row.get("umur", 0) or 0)
    sp = str(row.get("status_pekerjaan", "")).strip().lower()
    st = str(row.get("status_pernikahan", "")).strip().lower()

    rp = str(row.get("riwayat_penyakit", "")).strip().lower()
    sg = str(row.get("status_gizi", "")).strip().lower()
    t  = float(row.get("tinggi_cm", 0) or 0)
    b  = float(row.get("berat_kg", 0) or 0)

    bmi = None
    if pd.notna(t) and pd.notna(b) and t > 0 and b > 0:
        bmi = b / ((t/100.0) ** 2)

    # ✅ Layak Subsidi
    if (ph < 3_000_000) and (jt >= 3) and (u > 60) and \
       (sp in {"pengangguran", "buruh", "pekerja informal"}) and \
       (rp in KRONIS_SET):
        return 1

    if (u > 65) and (st in {"janda", "duda", "cerai"}) and \
       (ph < 4_000_000) and (sg in {"kurang", "stunting", "gizi buruk"}):
        return 1

    if (kr in {"tidak layak", "semi permanen", "sangat sederhana"}) and \
       (sp != "pegawai tetap") and (bmi is not None) and ((bmi < 17) or (bmi > 35)):
        return 1

    if (jt >= 5) and (ph < 6_000_000) and (rp != "sehat"):
        return 1

    # ❌ Tidak Layak
    if (ph >= 8_000_000) and (jt <= 2) and \
       (sp in {"pegawai tetap", "pns", "karyawan tetap"}) and \
       (st == "menikah") and (sg == "baik") and (rp == "sehat") and \
       (bmi is not None) and (18.5 <= bmi <= 25):
        return 0

    if (u < 25) and (sp == "wirausaha") and (ph >= 5_000_000) and (kr == "layak"):
        return 0

    if (ph >= 5_000_000) and (jt <= 2) and (kr == "layak") and \
       (sp == "pegawai tetap") and (st == "menikah") and \
       (sg == "baik") and (rp == "sehat") and \
       (bmi is not None) and (18.5 <= bmi <= 25):
        return 0

    return 0

# -----------------------------
# Main Program
# -----------------------------
def main():
    if not DATA_IN.exists():
        raise FileNotFoundError(f"{DATA_IN} tidak ditemukan")

    df = pd.read_csv(DATA_IN)

    # Tambahkan kolom label
    df["label_gabungan"] = df.apply(label_gabungan, axis=1)

    # Simpan hasil
    df.to_csv(DATA_OUT, index=False)
    print(f"File labeled disimpan di {DATA_OUT} dengan {len(df)} baris")

    # Statistik ringkas
    print("Ringkasan label:")
    print(df["label_gabungan"].value_counts())

if __name__ == "__main__":
    main()
