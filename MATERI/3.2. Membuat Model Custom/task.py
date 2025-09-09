import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.preprocessing import MinMaxScaler

# ===================== Konfigurasi =====================
CSV_FILE = "dinsos_100.csv"
BATCH_SIZE = 10
CLIENT_EPOCHS = 5
SHUFFLE_BUFFER = 100
LABEL_NAME = "layak_subsidi"   # target biner

# ===================== Loader dataset =====================
def load_client_dataset(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != LABEL_NAME]

    # Encode kolom string jadi angka
    for col in feature_cols:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # Normalisasi fitur numerik
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Pastikan label 0/1 integer
    df[LABEL_NAME] = df[LABEL_NAME].astype(int)

    # Debug distribusi label
    print("Distribusi label:")
    print(df[LABEL_NAME].value_counts())

    def preprocess(x, y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (df[feature_cols].values, df[LABEL_NAME].values)
    )
    ds = ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).repeat(CLIENT_EPOCHS)
    return ds

# ===================== Dataset =====================
client_dataset = load_client_dataset(CSV_FILE)
federated_train_data = [client_dataset]

# Ambil input_spec
input_spec = federated_train_data[0].element_spec

# ===================== Logistic Regression Model =====================
def create_logreg_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# ===================== Neural Network Model =====================
def create_nn_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# ===================== Wrap ke TFF =====================
def model_fn():
    input_dim = input_spec[0].shape[-1]

    # pilih model
    keras_model = create_nn_model(input_dim)       # NN kecil
    # keras_model = create_logreg_model(input_dim) # Logistic Regression

    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.AUC()]
    )

# ===================== Federated Averaging =====================
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=0.01),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
)

state = iterative_process.initialize()

# ===================== Training Loop =====================
NUM_ROUNDS = 10
for round_num in range(1, NUM_ROUNDS + 1):
    result = iterative_process.next(state, federated_train_data)
    state = result.state

    # ambil metrik dari client_work → train
    train_metrics = result.metrics['client_work']['train']
    loss = float(train_metrics['loss'])
    acc = float(train_metrics['binary_accuracy'])

    print(f"Round {round_num} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
