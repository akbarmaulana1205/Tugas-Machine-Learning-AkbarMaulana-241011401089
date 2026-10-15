# Tugas Pemrosesan Data dan Machine Learning
# Nama: Akbar Maulana
# NIM: 241011401089
# Kelas: 05TPLE014

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. load dataset
print("Load dataset...")
df = pd.read_csv('Hospital_Indonesia_datasets.csv', sep=';')

print("Dimensi data:", df.shape)

# 2. data cleaning
print("Mulai proses cleaning data...")
kolom_drop = ['id', 'nama', 'alamat', 'kab', 'propinsi']
df = df.drop(columns=kolom_drop)

kolom_angka = ['total_tempat_tidur', 'total_layanan', 'total_tenaga_kerja']
for col in kolom_angka:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in kolom_angka:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=['kelas'])

# 3. EDA ringan (visualisasi)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='kelas', order=df['kelas'].value_counts().index)
plt.title('Distribusi Kelas Rumah Sakit')
plt.xlabel('Kelas')
plt.ylabel('Jumlah')
plt.savefig('distribusi_kelas.png')

# 4. preprocessing / encoding
print("Encoding variabel kategori...")
le_jenis = LabelEncoder()
le_status = LabelEncoder()
le_pemilik = LabelEncoder()

df['jenis'] = le_jenis.fit_transform(df['jenis'].astype(str))
df['status_blu'] = le_status.fit_transform(df['status_blu'].astype(str))
df['kepemilikan'] = le_pemilik.fit_transform(df['kepemilikan'].astype(str))

X = df.drop(columns=['kelas'])
y = df['kelas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. evaluasi hasil
print("Evaluasi model...")
y_pred = model.predict(X_test)

akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {akurasi:.2f}")
print("\nDetail Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, zero_division=0))

df.to_csv('cleaned_hospital_data.csv', index=False)
print("Selesai! Data bersih udah disave ke cleaned_hospital_data.csv")