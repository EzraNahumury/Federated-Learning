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
        # SavedModel format -> gunakan TFSMLayer
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(preproc["FEATURE_COLS"]),)),
            tf.keras.layers.TFSMLayer(str(model_dir), call_endpoint="serving_default")
        ])
    else:
        # Kalau nanti ada .h5 atau .keras
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


def test_model(preproc_pkl, model_path, test_cases, model_name, expected_labels):
    """Run test cases untuk 1 model"""
    print(f"\n===== TESTING {model_name.upper()} =====")
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

    print(f"Summary {model_name}: {benar}/{total} benar ({benar/total:.2%})")
    return benar, total


# =========================
# Test Cases
# =========================

# --- Dinsos ---
test_cases_dinsos = [
    {"penghasilan": 1500000, "jumlah_tanggungan": 2, "kondisi_rumah": "semi permanen"}, # Layak
    {"penghasilan": 2800000, "jumlah_tanggungan": 4, "kondisi_rumah": "permanen"},     # Layak
    {"penghasilan": 4500000, "jumlah_tanggungan": 3, "kondisi_rumah": "tidak layak"},  # Layak
    {"penghasilan": 5500000, "jumlah_tanggungan": 5, "kondisi_rumah": "permanen"},     # Layak
    {"penghasilan": 8500000, "jumlah_tanggungan": 2, "kondisi_rumah": "layak"},        # Tidak
    {"penghasilan": 7000000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak"},        # Tidak
]
expected_dinsos = [1, 1, 1, 1, 0, 0]

# --- Dukcapil ---
test_cases_dukcapil = [
    {"umur": 70, "status_pekerjaan": "pengangguran", "status_pernikahan": "janda"},   # Layak
    {"umur": 60, "status_pekerjaan": "buruh", "status_pernikahan": "menikah"},        # Layak
    {"umur": 58, "status_pekerjaan": "wirausaha", "status_pernikahan": "cerai"},      # Layak
    {"umur": 40, "status_pekerjaan": "pegawai tetap", "status_pernikahan": "menikah"},# Tidak
    {"umur": 22, "status_pekerjaan": "wirausaha", "status_pernikahan": "lajang"},     # Tidak
]
expected_dukcapil = [1, 1, 1, 0, 0]

# --- Kemenkes ---
test_cases_kemenkes = [
    {"tinggi_cm": 160, "berat_kg": 45, "riwayat_penyakit": "diabetes", "status_gizi": "gizi buruk"}, # Layak
    {"tinggi_cm": 170, "berat_kg": 40, "riwayat_penyakit": "jantung",  "status_gizi": "kurang"},     # Layak
    {"tinggi_cm": 165, "berat_kg": 50, "riwayat_penyakit": "sehat",    "status_gizi": "baik"},       # Layak
    {"tinggi_cm": 160, "berat_kg": 90, "riwayat_penyakit": "diabetes", "status_gizi": "baik"},       # Layak
    {"tinggi_cm": 170, "berat_kg": 63, "riwayat_penyakit": "sehat",    "status_gizi": "baik"},       # Tidak (BMI ≈ 21.8 normal)
    {"tinggi_cm": 180, "berat_kg": 78, "riwayat_penyakit": "sehat",    "status_gizi": "baik"},       # Tidak (BMI ≈ 24.1 normal)
]

expected_kemenkes = [1, 1, 1, 1, 0, 0]












# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    total_benar, total_uji = 0, 0

    b, t = test_model("preprocess_dinsos_rule.pkl", "saved_dinsos_tff",
                      test_cases_dinsos, "Dinsos", expected_dinsos)
    total_benar += b; total_uji += t

    b, t = test_model("preprocess_dukcapil_rule.pkl", "saved_dukcapil_tff",
                      test_cases_dukcapil, "Dukcapil", expected_dukcapil)
    total_benar += b; total_uji += t

    b, t = test_model("preprocess_kemenkes_rule.pkl", "saved_kemenkes_tff",
                      test_cases_kemenkes, "Kemenkes", expected_kemenkes)
    total_benar += b; total_uji += t

    print("\n===== HASIL AKHIR =====")
    print(f"Total: {total_benar}/{total_uji} benar ({total_benar/total_uji:.2%})")
