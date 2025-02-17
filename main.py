import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_and_evaluate(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")

    return model


def predict_and_save(model, df_test, test_ids, label_encoder, output_file):
    y_pred = np.argmax(model.predict(df_test), axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    submission = pd.DataFrame({"ID_Siswa": test_ids, "Kategori_Gizi": y_pred_labels})
    submission.to_csv(output_file, index=False, sep=';')
    print(f"Submission file saved as {output_file}")


if __name__ == '__main__':
    train_path = "dataset/data_training.csv"
    test_path = "dataset/data_testing_no_label.csv"
    sample_submission_path = "dataset/Sample_Submission.csv"
    output_file = "BDC_Prediksi_Alon_Alon_Muhammad_Alvaro_Khikman.csv"
    model_save_path = "model.h5"

    df_train, df_test, test_ids = load_data(train_path, test_path, sample_submission_path)
    df_train, df_test, label_encoder = preprocess_data(df_train, df_test)

    X = df_train.drop(columns=["Kategori_Gizi"])
    y = df_train["Kategori_Gizi"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1], len(label_encoder.classes_))
    model = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    model.save(model_save_path)
    print(f"Model saved as {model_save_path}")

    predict_and_save(model, df_test, test_ids, label_encoder, output_file)
