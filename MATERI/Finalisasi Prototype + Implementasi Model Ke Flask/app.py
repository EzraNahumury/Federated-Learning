import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# === Load trained model ===
keras_model = tf.keras.models.load_model("trained_model.h5", compile=False)

# === Load preprocessing info ===
preproc = joblib.load("preprocess.pkl")
FEATURE_COLS = preproc["FEATURE_COLS"]
mins = preproc["mins"]
rng = preproc["rng"]

def preprocess_input(form_data):
    df = pd.DataFrame([form_data])

    # Tambahkan kolom kosong untuk semua FEATURE_COLS yang tidak ada di input
    for col in FEATURE_COLS:
        if col not in df:
            df[col] = 0.0

    # Scale sesuai global
    df_scaled = (df[FEATURE_COLS] - mins) / rng
    df_scaled = df_scaled.fillna(0.0)
    return df_scaled.values.astype("float32")

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # Mulai dengan data numeric
    input_data = {
        "jumlah_tanggungan": int(form["jumlah_tanggungan"]),
        "penghasilan": float(form["penghasilan"]),
        "umur": int(form["umur"]),
        "tinggi_cm": float(form["tinggi_cm"]),
        "berat_kg": float(form["berat_kg"]),
    }

    # Tambahkan kategori one-hot sesuai pilihan user
    input_data[f"kondisi_rumah__{form['kondisi_rumah']}"] = 1.0
    input_data[f"status_pekerjaan__{form['status_pekerjaan']}"] = 1.0
    input_data[f"status_pernikahan__{form['status_pernikahan']}"] = 1.0
    input_data[f"riwayat_penyakit__{form['riwayat_penyakit']}"] = 1.0
    input_data[f"status_gizi__{form['status_gizi']}"] = 1.0

    X_user = preprocess_input(input_data)
    pred = keras_model.predict(X_user)[0][0]
    label = "✅ Layak Subsidi" if pred >= 0.5 else "❌ Tidak Layak Subsidi"
    return render_template("result.html", result=label, score=pred)

if __name__ == "__main__":
    app.run(debug=True)
