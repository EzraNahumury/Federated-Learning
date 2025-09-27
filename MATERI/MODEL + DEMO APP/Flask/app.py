import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from keras.layers import TFSMLayer

app = Flask(__name__)

# ============================================================
# 1. Load model & preprocessing
# ============================================================
MODELS = {
    "dinsos": {
        "model": TFSMLayer("models/saved_dinsos_tff", call_endpoint="serving_default"),
        "preproc": joblib.load("models/preprocess_dinsos_rule.pkl"),
    },
    "dukcapil": {
        "model": TFSMLayer("models/saved_dukcapil_tff", call_endpoint="serving_default"),
        "preproc": joblib.load("models/preprocess_dukcapil_rule.pkl"),
    },
    "kemenkes": {
        "model": TFSMLayer("models/saved_kemenkes_tff", call_endpoint="serving_default"),
        "preproc": joblib.load("models/preprocess_kemenkes_rule.pkl"),
    },
    "gabungan": {
        "model": TFSMLayer("models/saved_model_gabungan", call_endpoint="serving_default"),
        "preproc": joblib.load("models/preprocess_gabungan.pkl"),
    },
}

# ============================================================
# 2. Helper untuk preprocessing input
# ============================================================
def preprocess_input(data, preproc):
    feature_cols = preproc["FEATURE_COLS"]
    mins = preproc["mins"]
    rng = preproc["rng"]

    converted = {}
    for k, v in data.items():
        if v is None or v == "":
            converted[k] = 0.0
        else:
            try:
                converted[k] = float(v)
            except ValueError:
                converted[k] = str(v).strip().lower()

    df = pd.DataFrame([converted])

    for col in feature_cols:
        if col not in df:
            df[col] = 0.0

    df = df[feature_cols]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df_scaled = (df - mins) / rng
    df_scaled = df_scaled.fillna(0.0)

    return df_scaled.astype("float32").to_numpy()


# ============================================================
# 3. Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in MODELS:
        return jsonify({"error": "Model tidak ditemukan"}), 400

    model_info = MODELS[model_name]
    preproc = model_info["preproc"]
    model = model_info["model"]

    data = request.get_json(force=True)

    try:
        X = preprocess_input(data, preproc)
        outputs = model(X, training=False)

        # ambil output pertama dari dict
        y_pred = list(outputs.values())[0].numpy().flatten()[0]

        result = {
            "prediksi": int(y_pred >= 0.5),
            "probabilitas": float(y_pred),
            "status": "Layak" if y_pred >= 0.5 else "Tidak Layak",
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 4. Run server
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
