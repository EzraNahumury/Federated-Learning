import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1. Load Dataset CSV
# =========================================================
dinsos = pd.read_csv("dinsos_100.csv")
dukcapil = pd.read_csv("dukcapil_100.csv")
kemenkes = pd.read_csv("kemenkes_100.csv")

# =========================================================
# 2. Preprocessing: Encoding kategori + samakan jumlah fitur
# =========================================================

# ---- DINSOS ----
enc_dinsos = LabelEncoder()
dinsos['kondisi_rumah'] = enc_dinsos.fit_transform(dinsos['kondisi_rumah'])
# label sudah ada: layak_subsidi (0/1)
dinsos['dummy'] = 0   # biar total 4 fitur

# ---- DUKCAPIL ----
enc_pekerjaan = LabelEncoder()
enc_nikah = LabelEncoder()
dukcapil['status_pekerjaan'] = enc_pekerjaan.fit_transform(dukcapil['status_pekerjaan'])
dukcapil['status_pernikahan'] = enc_nikah.fit_transform(dukcapil['status_pernikahan'])
dukcapil['layak_subsidi'] = (dukcapil['umur'] > 40).astype(int)  # dummy rule
dukcapil['dummy'] = 0   # biar total 4 fitur

# ---- KEMENKES ----
enc_penyakit = LabelEncoder()
enc_gizi = LabelEncoder()
kemenkes['riwayat_penyakit'] = enc_penyakit.fit_transform(kemenkes['riwayat_penyakit'])
kemenkes['status_gizi'] = enc_gizi.fit_transform(kemenkes['status_gizi'])
kemenkes['layak_subsidi'] = (kemenkes['berat_kg'] > 60).astype(int)  # dummy rule
# sudah 4 fitur: riwayat_penyakit, status_gizi, tinggi_cm, berat_kg

print(" Data siap (contoh Dinsos):\n", dinsos.head())

# =========================================================
# 3. Fungsi Convert DataFrame -> tf.data.Dataset
# =========================================================
def make_dataset_from_df(df, label_col):
    labels = df[label_col].astype('int32').values   # pastikan int32
    features = df.drop(columns=[label_col]).astype('float32').values
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(20)
    return dataset

# =========================================================
# 4. Buat Federated Data (3 Client)
# =========================================================
client_dinsos = make_dataset_from_df(dinsos, 'layak_subsidi')
client_dukcapil = make_dataset_from_df(dukcapil, 'layak_subsidi')
client_kemenkes = make_dataset_from_df(kemenkes, 'layak_subsidi')

federated_train_data = [client_dinsos, client_dukcapil, client_kemenkes]
print(f"Total clients: {len(federated_train_data)}")

# =========================================================
# 5. Definisikan Model TFF
# =========================================================
input_dim = federated_train_data[0].element_spec[0].shape[1]  # 4 fitur

def model_fn():
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # 4 fitur
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # biner: 0/1
    ])

    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# =========================================================
# 6. Federated Training
# =========================================================
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.01),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
)

state = iterative_process.initialize()

for round_num in range(1, 6):
    result = iterative_process.next(state, federated_train_data)
    state, metrics = result.state, result.metrics
    acc = metrics['client_work']['train']['sparse_categorical_accuracy']
    loss = metrics['client_work']['train']['loss']
    print(f'Round {round_num} - Loss: {loss:.4f}, Accuracy: {acc:.4f}')
