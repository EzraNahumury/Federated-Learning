# test_gabungan.py
import joblib
import tensorflow as tf
import pandas as pd
from pathlib import Path

BASE_DIR = Path("models")

# =========================
# Helper Functions
# =========================
def load_model(preproc_pkl, model_path):
    """Load preprocessing info & trained model (Keras3 compatible)"""
    preproc = joblib.load(BASE_DIR / preproc_pkl)
    model_dir = BASE_DIR / model_path

    if model_dir.is_dir():
        # SavedModel format -> gunakan TFSMLayer (untuk Keras 3)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(preproc["FEATURE_COLS"]),)),
            tf.keras.layers.TFSMLayer(str(model_dir), call_endpoint="serving_default")
        ])
    else:
        # Kalau ada .h5 / .keras
        model = tf.keras.models.load_model(model_dir)

    return preproc, model


def preprocess_input(data, preproc):
    """Scale input sesuai FEATURE_COLS"""
    X = pd.DataFrame([data])
    for col in preproc["FEATURE_COLS"]:
        if col not in X:
            X[col] = 0.0
    X = X[preproc["FEATURE_COLS"]]
    X_scaled = (X - preproc["mins"]) / preproc["rng"]
    return X_scaled.fillna(0.0).astype("float32").values


def test_model(preproc_pkl, model_path, test_cases, expected_labels):
    """Run test cases untuk model gabungan"""
    print(f"\n===== TESTING MODEL GABUNGAN =====")
    preproc, model = load_model(preproc_pkl, model_path)
    benar, total = 0, len(test_cases)

    for i, (case, expected) in enumerate(zip(test_cases, expected_labels), 1):
        X = preprocess_input(case, preproc)
        out = model.predict(X, verbose=0)

        # Handle output dict dari TFSMLayer
        if isinstance(out, dict):
            pred = list(out.values())[0][0][0]
        else:
            pred = out[0][0]

        label = int(pred > 0.5)
        status = "✅ BENAR" if label == expected else "❌ SALAH"
        if label == expected:
            benar += 1

        print(f"Case {i:02d} | Input={case} | Pred={pred:.4f} "
              f"| Label={label} | Expected={expected} | {status}")

    print(f"\nSummary Gabungan: {benar}/{total} benar ({benar/total:.2%})")


# =========================
# Test Cases Gabungan
# =========================
test_cases_gabungan = [
    # Layak 1: Ekonomi & Demografi & Kesehatan
    {"jumlah_tanggungan": 4, "penghasilan": 2000000, "kondisi_rumah": "semi permanen",
     "umur": 65, "status_pekerjaan": "buruh", "status_pernikahan": "menikah",
     "riwayat_penyakit": "jantung", "status_gizi": "kurang",
     "tinggi_cm": 160, "berat_kg": 50},

    # Layak 2: Lansia Miskin
    {"jumlah_tanggungan": 2, "penghasilan": 3000000, "kondisi_rumah": "permanen",
     "umur": 70, "status_pekerjaan": "pengangguran", "status_pernikahan": "janda",
     "riwayat_penyakit": "sehat", "status_gizi": "gizi buruk",
     "tinggi_cm": 155, "berat_kg": 45},

    # Layak 3: Rumah Tidak Layak + Risiko Kesehatan
    {"jumlah_tanggungan": 3, "penghasilan": 4000000, "kondisi_rumah": "tidak layak",
     "umur": 40, "status_pekerjaan": "buruh", "status_pernikahan": "menikah",
     "riwayat_penyakit": "sehat", "status_gizi": "baik",
     "tinggi_cm": 160, "berat_kg": 100},  # BMI > 35

    # Layak 4: Anak Banyak + Penyakit
    {"jumlah_tanggungan": 6, "penghasilan": 5000000, "kondisi_rumah": "semi permanen",
     "umur": 50, "status_pekerjaan": "buruh", "status_pernikahan": "menikah",
     "riwayat_penyakit": "asma", "status_gizi": "baik",
     "tinggi_cm": 170, "berat_kg": 60},

    # Tidak Layak 1: Mapan & Sehat
    {"jumlah_tanggungan": 2, "penghasilan": 9000000, "kondisi_rumah": "layak",
     "umur": 40, "status_pekerjaan": "pegawai tetap", "status_pernikahan": "menikah",
     "riwayat_penyakit": "sehat", "status_gizi": "baik",
     "tinggi_cm": 170, "berat_kg": 65},  # BMI normal

    # Tidak Layak 2: Wirausaha muda
    {"jumlah_tanggungan": 1, "penghasilan": 6000000, "kondisi_rumah": "layak",
     "umur": 22, "status_pekerjaan": "wirausaha", "status_pernikahan": "lajang",
     "riwayat_penyakit": "sehat", "status_gizi": "baik",
     "tinggi_cm": 168, "berat_kg": 60},
]

expected_gabungan = [1, 1, 1, 1, 0, 0]

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    test_model("preprocess_gabungan.pkl", "saved_model_gabungan",
               test_cases_gabungan, expected_gabungan)
