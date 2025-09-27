import pandas as pd
from pathlib import Path

# ============================================================
# 0) Load CSV
# ============================================================
def load_or_fail(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(p)

# arahkan ke folder dataset
dinsos   = load_or_fail("DATASET/dinsos.csv")
dukcapil = load_or_fail("DATASET/dukcapil.csv")
kemenkes = load_or_fail("DATASET/kemenkes.csv")

# ============================================================
# 1) Aturan Labeling (REVISI)
# ============================================================
def label_dinsos(row):
    jt = row.get("jumlah_tanggungan", 0)
    ph = row.get("penghasilan", 0)
    kr = str(row.get("kondisi_rumah", "")).lower()

    # Aturan Layak Subsidi
    # 1. Penghasilan < 2 jt → Layak jika ada tanggungan & rumah sederhana
    if (ph < 2_000_000) and (jt >= 1) and (kr in {"tidak layak", "semi permanen", "sangat sederhana"}):
        return 1
    
    # 2. Penghasilan 2–3.5 jt & tanggungan >= 4 → Layak
    if (2_000_000 <= ph < 3_500_000) and (jt >= 4):
        return 1
    
    # 3. Penghasilan < 5 jt & rumah sederhana → Layak
    if (ph < 5_000_000) and (kr in {"tidak layak", "semi permanen", "sangat sederhana"}):
        return 1
    
    # 4. Tanggungan >= 5 & penghasilan < 6 jt → Layak
    if (jt >= 5) and (ph < 6_000_000):
        return 1

    # Aturan Tidak Layak
    # 5. Penghasilan >= 6 jt & tanggungan <= 2 & rumah layak → Tidak Layak
    if (ph >= 6_000_000) and (jt <= 2) and (kr == "layak"):
        return 0
    
    # 6. Penghasilan >= 8 jt → Tidak Layak
    if ph >= 8_000_000:
        return 0

    # Default: Tidak Layak
    return 0



def label_dukcapil(row):
    u  = row.get("umur", 0)
    sp = str(row.get("status_pekerjaan", "")).lower()
    st = str(row.get("status_pernikahan", "")).lower()

    # Layak
    if (u > 65) and (sp in {"pengangguran", "buruh", "pekerja informal"}) and (st in {"janda", "duda", "cerai"}):
        return 1
    if (55 <= u <= 65) and (sp == "wirausaha") and (st != "menikah"):
        return 1
    if (40 <= u <= 60) and (sp == "buruh") and (st == "menikah"):
        return 1

    # Tidak Layak
    if (25 <= u <= 55) and (sp in {"pegawai tetap", "pns", "karyawan tetap"}) and (st == "menikah"):
        return 0
    if (u < 25) and (sp == "wirausaha") and (st in {"lajang", "menikah"}):
        return 0

    return 0


def label_kemenkes(row):
    rp = str(row.get("riwayat_penyakit", "")).lower()
    sg = str(row.get("status_gizi", "")).lower()
    t  = row.get("tinggi_cm", None)
    b  = row.get("berat_kg", None)

    # Hitung BMI jika data lengkap
    bmi = None
    if pd.notna(t) and pd.notna(b) and t > 0:
        bmi = b / ((t / 100.0) ** 2)

    # ===== Aturan Layak Subsidi =====
    # 1. Penyakit kronis + gizi buruk/kurang/stunting + BMI rendah
    if (rp in {"kronis", "jantung", "asma", "diabetes", "disabilitas"}) \
       and (sg in {"kurang", "stunting", "gizi buruk"}) \
       and (bmi is not None and bmi < 18.5):
        return 1

    # 2. Hipertensi dengan kelebihan berat badan
    if (rp == "hipertensi") and (bmi is not None and bmi >= 25):
        return 1

    # 3. Status gizi baik tapi BMI ekstrem
    if (sg == "baik") and (bmi is not None) and (bmi < 17 or bmi > 35):
        return 1

    # 4. Tanpa penyakit kronis, gizi baik, tapi BMI masih di bawah normal
    if (rp not in {"kronis", "jantung", "asma", "diabetes", "disabilitas"}) \
       and (sg == "baik") \
       and (bmi is not None and bmi < 19):
        return 1

    # 5. Penyakit kronis + BMI sangat rendah/tinggi (meskipun gizi baik)
    if (rp in {"kronis", "jantung", "asma", "diabetes", "disabilitas"}) \
       and (sg == "baik") \
       and (bmi is not None and (bmi < 18.5 or bmi > 30)):
        return 1

    # ===== Aturan Tidak Layak =====
    # Orang sehat, gizi baik, dan BMI normal
    if (rp == "sehat") and (sg == "baik") and (bmi is not None and 18.5 <= bmi <= 25):
        return 0

    # Default: Tidak Layak
    return 0



# ============================================================
# 2) Tambahkan kolom "layak_subsidi"
# ============================================================
dinsos["layak_subsidi"]   = dinsos.apply(label_dinsos, axis=1)
dukcapil["layak_subsidi"] = dukcapil.apply(label_dukcapil, axis=1)
kemenkes["layak_subsidi"] = kemenkes.apply(label_kemenkes, axis=1)

# ============================================================
# 3) Simpan sebagai file CSV baru di folder dataset
# ============================================================
Path("NEW_DATASET").mkdir(parents=True, exist_ok=True)
dinsos.to_csv("NEW_DATASET/dinsos_labeled.csv", index=False)
dukcapil.to_csv("NEW_DATASET/dukcapil_labeled.csv", index=False)
kemenkes.to_csv("NEW_DATASET/kemenkes_labeled.csv", index=False)

print("✅ Semua file berhasil dibuat di folder NEW_DATASET/")
