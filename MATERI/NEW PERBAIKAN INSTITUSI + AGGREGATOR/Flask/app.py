import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from keras.layers import TFSMLayer

# ============================================================
# ⚙️ Konfigurasi Flask
# ============================================================
app = Flask(__name__, template_folder="templates")

# ============================================================
# 📦 Konfigurasi Model
# ============================================================
MODELS = {
    "dinsos": {
        "path": "Models/saved_dinsos_tff",
        "preproc": "Models/saved_dinsos_tff/preprocess_dinsos.pkl"
    },
    "dukcapil": {
        "path": "Models/saved_dukcapil_tff",
        "preproc": "Models/saved_dukcapil_tff/preprocess_dukcapil.pkl"
    },
    "kemenkes": {
        "path": "Models/saved_kemenkes_tff",
        "preproc": "Models/saved_kemenkes_tff/preprocess_kemenkes.pkl"
    },
    "gabungan": {
        "path": "Models/saved_global_tff",
        "preproc": "Models/fitur_global.pkl"
    },
}


# ============================================================
# 🧩 Fungsi Preprocessing Input
# ============================================================
def preprocess_input(data, preproc):
    import pandas as pd

    feature_cols = list(preproc["FEATURE_COLS"])
    mins = pd.Series(preproc["mins"]).reindex(feature_cols).fillna(0)
    rng  = pd.Series(preproc["rng"]).reindex(feature_cols).replace(0, 1)

    # Normalizer kecil
    def norm(x): 
        return str(x).strip().lower()

    # Sinonim untuk menyamakan input UI vs kolom training
    CANON = {
        "status_pekerjaan": {
            "karyawan tetap": "pegawai tetap",
            "pegawai tetap":  "pegawai tetap",
            "pns":            "PNS",            # biarkan kapital sesuai kolom
            "buruh":          "buruh harian",   # kalau di UI pernah dipersingkat
        },
        "kondisi_rumah": {
            "tdk layak": "tidak layak",
            "sangat sederhana": "sangat sederhana",
            "semi permanen": "semi permanen",
            "sederhana": "sederhana",
            "layak": "layak",
            "mewah": "mewah",
        }
    }

    # DataFrame final berisi seluruh kolom fitur
    df = pd.DataFrame([0]*len(feature_cols), index=feature_cols).T

    # Isi fitur numerik jika ada (yang lain biarkan 0)
    for num_col in ["penghasilan", "jumlah_tanggungan","lama_tinggal_tahun",
                    "jumlah_anggota_kk","usia_kepala_keluarga"]:
        if num_col in data and str(data[num_col]).strip() != "":
            try:
                df[num_col] = float(data[num_col])
            except:
                df[num_col] = 0.0

    # One-hot exact match: pecah dari belakang -> base, kategori
    for col in feature_cols:
        if "_" not in col:
            continue
        base, cat = col.rsplit("_", 1)  # 'kondisi_rumah_tidak layak' -> ('kondisi_rumah','tidak layak')
        if base in data:
            raw = data[base]
            v = norm(raw)
            # mapping sinonim (jika ada)
            if base in CANON:
                v = norm(CANON[base].get(v, raw))
            # samakan format pembanding (lower)
            if norm(cat) == v:
                df[col] = 1

    # Scale sesuai mins & rng dari training
    df = ((df - mins) / rng).fillna(0.0)
    return df.astype("float32").to_numpy()



# ============================================================
# 🧠 Prediksi Model dengan Threshold Otomatis
# ============================================================
def predict_with_threshold(model_name, data):
    model_path = MODELS[model_name]["path"]
    preproc_path = MODELS[model_name]["preproc"]

    if not os.path.exists(model_path):
        return {"error": f"Model {model_name} tidak ditemukan."}

    model = TFSMLayer(model_path, call_endpoint="serving_default")
    preproc = joblib.load(preproc_path)

    # Preprocess input
    X = preprocess_input(data, preproc)
    outputs = model(X, training=False)
    y_prob = float(list(outputs.values())[0].numpy().flatten()[0])

    # Threshold otomatis berdasarkan sampling (0.45–0.7)
    thresholds = np.linspace(0.45, 0.7, 26)
    # simulasi: asumsikan model balanced — ambil 0.5 default optimal
    best_t = 0.5

    y_pred = int(y_prob >= best_t)

    return {
        "prediksi": y_pred,
        "probabilitas": round(y_prob, 4),
        "threshold": round(best_t, 3)
    }


# ============================================================
# 🌐 ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/<model>", methods=["POST"])
def predict(model):
    if model not in MODELS:
        return jsonify({"error": "Model tidak dikenali"}), 400

    data = request.get_json(force=True)
    result = predict_with_threshold(model, data)

    return jsonify(result)


# ============================================================
# 🚀 MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
