# centralized.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 1) Load data
df = pd.read_csv("gabungan.csv")

# 2) Feature/label
y = df["layak_subsidi"]
X = df.drop(columns=["layak_subsidi"])

# 3) Definisikan kolom numerik & kategorikal
num_cols = ["jumlah_tanggungan", "penghasilan", "umur", "tinggi_cm", "berat_kg"]
cat_cols = ["kondisi_rumah", "status_pekerjaan", "status_pernikahan",
            "riwayat_penyakit", "status_gizi"]

# (Opsional) jika ada NaN, isi sederhana
X[num_cols] = X[num_cols].fillna(0)
X[cat_cols] = X[cat_cols].fillna("unknown")

# 4) Preprocessor:
#    - Numerik: StandardScaler(with_mean=False) agar aman bila gabung sparse
#    - Kategorikal: OneHotEncoder(handle_unknown='ignore') biar robust
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 5) Model pipeline: preprocessor -> LogisticRegression
#    liblinear/saga cocok untuk data sparse; tambah max_iter biar konvergen
clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogisticRegression(solver="liblinear", max_iter=200))
])

# 6) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Train
clf.fit(X_train, y_train)

# 8) Evaluasi
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)
