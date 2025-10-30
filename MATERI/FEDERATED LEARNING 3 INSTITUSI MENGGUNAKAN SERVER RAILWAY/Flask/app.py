import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from keras.layers import TFSMLayer
from sklearn.metrics import precision_recall_curve, roc_curve

# ============================================================
# ⚙️ Konfigurasi Flask
# ============================================================
app = Flask(__name__, template_folder="templates")

# ============================================================
# 📦 Konfigurasi Model Global
# ============================================================
THRESHOLD_MODE = "AUTO"   # "AUTO" atau "MANUAL"
THRESHOLD_MANUAL = 0.55   # Jika mode MANUAL, pakai ini

MODEL_GLOBAL = {
    "path": "Models/saved_global_tff",
    "preproc": "Models/fitur_global_test.pkl"
}

# ============================================================
# 🔧 Fungsi Threshold Otomatis (PR/F1 + ROC)
# ============================================================
def auto_threshold_from_probs(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    precision, recall, pr_thres = precision_recall_curve(y_true, y_prob)
    pr_thres = np.append(pr_thres, pr_thres[-1] if len(pr_thres) else 0.5)
    f1s = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    t_f1 = pr_thres[np.nanargmax(f1s)]

    fpr, tpr, roc_thres = roc_curve(y_true, y_prob)
    t_roc = roc_thres[int(np.argmax(tpr - fpr))]

    return float((t_f1 + t_roc) / 2.0)

# ============================================================
# 🧩 Preprocessing Input
# ============================================================
def preprocess_input(data, preproc):
    feature_cols = list(preproc["FEATURE_COLS"])
    mins = pd.Series(preproc["mins"]).reindex(feature_cols).fillna(0)
    rng = pd.Series(preproc["rng"]).reindex(feature_cols).replace(0, 1)

    df = pd.DataFrame([0] * len(feature_cols), index=feature_cols).T

    for col in ["penghasilan", "jumlah_tanggungan", "lama_tinggal_tahun",
                "jumlah_anggota_kk", "usia_kepala_keluarga"]:
        if col in data and str(data[col]).strip() != "":
            df[col] = float(data[col])

    for col in feature_cols:
        if "_" in col:
            base, cat = col.rsplit("_", 1)
            if base in data and str(data[base]).lower().strip() == cat.lower():
                df[col] = 1

    df = ((df - mins) / rng).fillna(0.0)
    return df.astype("float32").to_numpy()

# ============================================================
# 🧠 Prediksi Global (Semua Lembaga Pakai Model yang Sama)
# ============================================================
# def predict_global(lembaga, data):
#     preproc = joblib.load(MODEL_GLOBAL["preproc"])
#     model = TFSMLayer(MODEL_GLOBAL["path"], call_endpoint="serving_default")

#     X = preprocess_input(data, preproc)
#     y_prob = float(list(model(X, training=False).values())[0].numpy().flatten()[0])

#     # =====================================================
#     # ✅ PRIORITAS 1 → kalau preproc sudah punya threshold → gunakan
#     # =====================================================
#     if isinstance(preproc, dict) and "threshold" in preproc:
#         best_t = float(preproc["threshold"])

#     # =====================================================
#     # ✅ PRIORITAS 2 → AUTO seperti test.py (hitung sekali)
#     # =====================================================
#     elif THRESHOLD_MODE == "AUTO":
#         # threshold dihitung dari data training/validasi yang disimpan di preproc
#         if "y_true" in preproc and "y_prob" in preproc:
#             best_t = auto_threshold_from_probs(preproc["y_true"], preproc["y_prob"])
#             # simpan threshold ke preproc biar konsisten di request berikutnya
#             preproc["threshold"] = float(best_t)
#             joblib.dump(preproc, MODEL_GLOBAL["preproc"])
#         else:
#             best_t = 0.5   # fallback aman

#     # =====================================================
#     # ✅ PRIORITAS 3 → MANUAL MODE
#     # =====================================================
#     else:
#         best_t = THRESHOLD_MANUAL

#     y_pred = int(y_prob >= best_t)

#     return {
#         "lembaga": lembaga.upper(),
#         "prediksi": y_pred,
#         "probabilitas": round(y_prob, 4),
#         "threshold": round(best_t, 4)  
#     }


def predict_global(lembaga, data):
    model = TFSMLayer(MODEL_GLOBAL["path"], call_endpoint="serving_default")
    preproc = joblib.load(MODEL_GLOBAL["preproc"])

    X = preprocess_input(data, preproc)
    y_prob = float(list(model(X, training=False).values())[0].numpy().flatten()[0])

    # ✅ Threshold Default
    best_t = 0.50

    # ✅ Threshold khusus KEMENKES (hasil dari test.py)
    if lembaga == "kemenkes":
        best_t = 0.4998   

    y_pred = int(y_prob >= best_t)

    return {
        "lembaga": lembaga.upper(),
        "prediksi": y_pred,
        "probabilitas": round(y_prob, 4),
        "threshold": round(best_t, 4)
    }


# ============================================================
# 🌐 Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/<lembaga>", methods=["POST"])
def predict(lembaga):
    allowed = ["dinsos", "dukcapil", "kemenkes", "gabungan"]
    if lembaga not in allowed:
        return jsonify({"error": "Model lembaga tidak dikenali"}), 400

    data = request.get_json(force=True)
    return jsonify(predict_global(lembaga, data))

# ============================================================
# 🚀 MAIN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
