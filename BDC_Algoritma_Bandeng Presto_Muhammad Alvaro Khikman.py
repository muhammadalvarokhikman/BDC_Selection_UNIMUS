from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import xgboost as xgb
import pandas as pd
import numpy as np

class CustomStopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        if val_acc is not None and val_loss is not None:
            if val_acc >= 0.99 and val_loss <= 0.03:
                print("\nAkurasi mencapai >= 0.99 dan validation loss <= 0.03. Pelatihan Selesai.\n")
                self.model.stop_training = True

def load_data(train_path, test_path, sample_submission_path):
    df_train = pd.read_csv(train_path, sep=';')
    df_test = pd.read_csv(test_path, sep=';')
    sample_submission = pd.read_csv(sample_submission_path, sep=';')
    df_train.drop(columns=["ID_Siswa"], inplace=True)
    test_ids = df_test["ID_Siswa"]
    df_test.drop(columns=["ID_Siswa"], inplace=True)
    return df_train, df_test, test_ids

def preprocess_data(df_train, df_test):
    # Encoding label target
    label_encoder = LabelEncoder()
    df_train["Kategori_Gizi"] = label_encoder.fit_transform(df_train["Kategori_Gizi"])
    # Encoding fitur kategorikal
    categorical_columns = ["Tingkat_Kesulitan", "Jenis_Kelamin", "Riwayat_Penyakit"]
    df_train = pd.get_dummies(df_train, columns=categorical_columns)
    df_test = pd.get_dummies(df_test, columns=categorical_columns)
    # Menyamakan fitur dengan data training
    missing_cols = set(df_train.columns) - set(df_test.columns) - {"Kategori_Gizi"}
    for col in missing_cols:
        df_test[col] = 0  # Tambahkan kolom yang hilang dengan nilai default 0
    df_test = df_test[df_train.drop(columns=["Kategori_Gizi"]).columns]
    # Normalisasi fitur numerik
    numerical_columns = ["Usia", "Berat_Badan", "Tinggi_Badan", "IMT", "Asupan_Kalori",
                         "Aktivitas_Fisik", "Frekuensi_Makan", "Konsumsi_Sayur_Buah",
                         "Durasi_Tidur (Menit)", "Fast_Food_Per_Minggu"]
    scaler = StandardScaler()
    df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
    df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])
    return df_train, df_test, label_encoder

# Fungsi load_data dan preprocess_data
train_path = 'dataset/data_training.csv'
test_path = 'dataset/data_testing_no_label.csv'
sample_submission_path = 'dataset/Sample_Submission.csv'
df_train, df_test, test_ids = load_data(train_path, test_path, sample_submission_path)
df_train, df_test, label_encoder = preprocess_data(df_train, df_test)

# Pemisahan fitur dan target
X = df_train.drop(columns=["Kategori_Gizi"])
y = df_train["Kategori_Gizi"]

# Split data untuk validasi
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model XGBoost
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(y.unique()),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

# Training model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_valid)

# Evaluasi accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy: {accuracy:.4f}")

def predict_and_save(model, df_test, test_ids, label_encoder, output_file):
    y_test_pred = model.predict(df_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    submission = pd.DataFrame({"ID": test_ids, "Kategori_Gizi_Prediksi": y_test_pred_labels})
    submission.to_csv(output_file, index=False, sep=',')
    print(f"Submission file saved as {output_file}")

if __name__ == '__main__':
    output_file = "BDC_Prediksi_Bandeng Presto_Muhammad Alvaro Khikman.csv"
    model_save_path = "model.h5"
    predict_and_save(model, df_test, test_ids, label_encoder, output_file)