from keras import backend
#backend.clear_session()
import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv1D, Flatten, LSTM, Dropout, MaxPooling1D, BatchNormalization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
dataset_path = "c:\\Users\\Pathuru\\Desktop\\DeepIDS2\\Dataset1"
model_save_path = "c:\\Users\\Pathuru\\Desktop\\DeepIDS2\\Trained Models"
csv_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.csv')]
def load_and_merge_csv(csv_files, drop_columns=None):
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce')
        if drop_columns:
            df.drop(columns=drop_columns, errors='ignore', inplace=True)
        dataframes.append(df)
    common_columns = set.intersection(*(set(df.columns) for df in dataframes))
    dataframes = [df[common_columns] for df in dataframes]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df.fillna(combined_df.mean())
print("Loading and merging CSV files...")
data = load_and_merge_csv(csv_files, drop_columns=["Label"])
X = data.iloc[:, :-1].to_numpy(dtype=np.float32)
y = data.iloc[:, -1].to_numpy(dtype=np.float32)
y = (y > 0.5).astype(int)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
X[X < -1] = -1
X = np.log1p(X)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=0.0)
scaler = MinMaxScaler()
X = scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    return autoencoder
print("Creating and training the autoencoder...")
autoencoder = create_autoencoder(X.shape[1])
chunk_size = 2500000
for i in range(0, len(X), chunk_size):
    X_chunk = X[i:i + chunk_size]
    print(f"Training chunk {i // chunk_size + 1}/{(len(X) + chunk_size - 1) // chunk_size}")
    autoencoder.fit(X_chunk, X_chunk, batch_size=64, epochs=5, verbose=1)
    backend.clear_session()
    gc.collect()
autoencoder.save(os.path.join(model_save_path, "Autoencoder_Model.h5"))
print("Autoencoder trained and saved")
encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
X_transformed = []
for i in range(0, len(X), chunk_size):
    X_chunk = X[i:i + chunk_size]
    print(f"Transforming chunk {i // chunk_size + 1}/{(len(X) + chunk_size - 1) // chunk_size}")
    X_transformed_chunk = encoder.predict(X_chunk, batch_size=64)
    X_transformed.append(X_transformed_chunk)
    backend.clear_session()
    gc.collect()
X_transformed = np.vstack(X_transformed)
X_cnn = X_transformed.reshape(X_transformed.shape[0], X_transformed.shape[1], 1)
X_lstm = X_transformed.reshape(X_transformed.shape[0], 1, X_transformed.shape[1])
def cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.4),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
print("Training CNN model...")
cnn_model = cnn((X_cnn.shape[1], X_cnn.shape[2]))
for i in range(0, len(X_cnn), chunk_size):
    print(f"Training chunk {i // chunk_size + 1}/{(len(X_cnn) + chunk_size - 1) // chunk_size}")
    X_chunk = X_cnn[i:i + chunk_size]
    y_chunk = y[i:i + chunk_size]
    cnn_model.fit(X_chunk, y_chunk, batch_size=64, epochs=15, verbose=1)
    backend.clear_session()
    gc.collect()
cnn_model.save(os.path.join(model_save_path, "CNN_Model.h5"))
print("CNN model trained and saved")
def lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.4),
        BatchNormalization(),
        LSTM(256, dropout=0.5),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
print("Training LSTM model...")
lstm_model = lstm((X_lstm.shape[1], X_lstm.shape[2]))
for i in range(0, len(X_lstm), chunk_size):
    print(f"Training chunk {i // chunk_size + 1}/{(len(X_lstm) + chunk_size - 1) // chunk_size}")
    X_chunk = X_lstm[i:i + chunk_size]
    y_chunk = y[i:i + chunk_size]
    lstm_model.fit(X_chunk, y_chunk, batch_size=64, epochs=15, verbose=1)
    backend.clear_session()
    gc.collect()
lstm_model.save(os.path.join(model_save_path, "LSTM_Model.h5"))
print("LSTM model trained and saved")