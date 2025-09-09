import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('data_dummy.csv')




# # --- Label Encoding ---
# le = LabelEncoder()
# df['kota_label'] = le.fit_transform(df['kota'])

# --- One-Hot Encoding ---
df = pd.get_dummies(df, columns=['Kota'])


scaler = StandardScaler()
df[['Umur', 'Pendapatan']] = scaler.fit_transform(df[['Umur', 'Pendapatan']])


# print(df)
print(df.to_string())
