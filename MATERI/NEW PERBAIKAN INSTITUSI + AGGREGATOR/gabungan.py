import tensorflow as tf
import numpy as np
from pathlib import Path

# ==========================================================
# 1️⃣ LOAD MODEL SETIAP INSTANSI
# ==========================================================
models_dir = Path("Models")

paths = {
    "dinsos": models_dir / "saved_dinsos_tff",
    "dukcapil": models_dir / "saved_dukcapil_tff",
    "kemenkes": models_dir / "saved_kemenkes_tff"
}

print("📂 Memuat model lokal...")
models = {name: tf.keras.models.load_model(path) for name, path in paths.items()}
print("✅ Semua model lokal berhasil dimuat!")

# ==========================================================
# 2️⃣ KONFIGURASI WEIGHT (Jika ingin diberi bobot)
# ==========================================================
# Contoh: jika jumlah data pelatihan berbeda di tiap instansi
client_weights = {
    "dinsos": 0.4,   # misalnya Dinsos punya data 40%
    "dukcapil": 0.35,
    "kemenkes": 0.25
}

# ==========================================================
# 3️⃣ AMBIL BOBOT DAN LAKUKAN AGREGASI FEDAVG
# ==========================================================
def aggregate_fedavg(models_dict, weights_dict):
    names = list(models_dict.keys())
    n_models = len(names)
    
    # Ambil bobot pertama sebagai referensi struktur layer
    base_weights = models_dict[names[0]].get_weights()
    new_weights = [np.zeros_like(w) for w in base_weights]

    for name in names:
        w = models_dict[name].get_weights()
        weight_factor = weights_dict.get(name, 1.0 / n_models)
        for i in range(len(w)):
            new_weights[i] += w[i] * weight_factor
    
    return new_weights

global_weights = aggregate_fedavg(models, client_weights)
print("✅ Agregasi bobot global selesai.")

# ==========================================================
# 4️⃣ BANGUN MODEL GLOBAL DAN ASSIGN WEIGHT
# ==========================================================
sample_model = models["dinsos"]
global_model = tf.keras.models.clone_model(sample_model)
global_model.set_weights(global_weights)

# ==========================================================
# 5️⃣ SIMPAN MODEL GLOBAL
# ==========================================================
save_dir = models_dir / "saved_global_tff"
save_dir.mkdir(parents=True, exist_ok=True)
global_model.save(save_dir)
print(f"🌍 Model global tersimpan di: {save_dir}")

# ==========================================================
# 6️⃣ OPSIONAL: CEK INFORMASI BOBOT
# ==========================================================
print("\n========== INFO BOBOT GLOBAL ==========")
print("Total parameter:", global_model.count_params())
for layer in global_model.layers:
    w = layer.get_weights()
    if w:
        print(f"Layer: {layer.name}, shape: {[p.shape for p in w]}")
