import tensorflow as tf
import numpy as np
from pathlib import Path

# ==========================================================
# KONFIGURASI
# ==========================================================
models_dir = Path("models")
paths = {
    "global_prev": models_dir / "saved_global_tff",
    "dinsos_new": models_dir / "saved_dinsos2_tff"
}

print("\n📂 Memuat model lokal & global sebelumnya...")
models = {name: tf.keras.models.load_model(path) for name, path in paths.items()}
print("✅ Model global lama & model lokal baru berhasil dimuat!")

# ==========================================================
# 1️⃣ FEDERATED AVERAGING (FEDAVG)
# ==========================================================
def aggregate_fedavg(models_dict, weights_dict=None):
    names = list(models_dict.keys())
    base_weights = models_dict[names[0]].get_weights()
    new_weights = [np.zeros_like(w) for w in base_weights]

    if weights_dict is None:
        weights_dict = {name: 1.0 / len(names) for name in names}

    for name in names:
        w = models_dict[name].get_weights()
        weight_factor = weights_dict.get(name, 1.0 / len(names))
        for i in range(len(w)):
            new_weights[i] += w[i] * weight_factor
    return new_weights

# ==========================================================
# 2️⃣ HITUNG AGREGASI (50:50 antara global lama dan model baru)
# ==========================================================
client_weights = {
    "global_prev": 0.5,
    "dinsos_new": 0.5
}

global_weights = aggregate_fedavg(models, client_weights)
print("✅ Agregasi bobot selesai (rasio 50:50).")

# ==========================================================
# 3️⃣ SIMPAN MODEL GLOBAL BARU
# ==========================================================
global_model = tf.keras.models.clone_model(models["global_prev"])
global_model.set_weights(global_weights)

save_dir = models_dir / "saved_global2_tff"
save_dir.mkdir(parents=True, exist_ok=True)
global_model.save(save_dir)

print(f"🌍 Model global baru berhasil disimpan di: {save_dir}")
print("\n========== INFO BOBOT GLOBAL BARU ==========")
print("Total parameter:", global_model.count_params())
for layer in global_model.layers:
    w = layer.get_weights()
    if w:
        print(f"Layer: {layer.name}, shape: {[p.shape for p in w]}")
