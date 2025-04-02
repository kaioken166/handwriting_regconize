import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import cv2
import os

data_dir = "../dataset/images"
labels_file = "../labels_continue.csv"
img_height, img_width = 100, 100
max_length = 5
num_classes = 11  # 10 số (0-9) + dấu phẩy


def load_data():
    df = pd.read_csv(labels_file)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['image_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            img = img / 255.0  # Chuẩn hóa về [0,1]
            images.append(img)
            labels.append(encode_label(row['label']))

    images = np.expand_dims(np.array(images), axis=-1)
    labels = np.array(labels)
    return images, labels


def encode_label(label):
    mapping = {str(i): i for i in range(10)}
    mapping[","] = 10  # Đánh dấu dấu phẩy
    label_encoded = [mapping[c] for c in label]
    while len(label_encoded) < max_length:
        label_encoded.append(10)  # Padding bằng dấu phẩy
    return keras.utils.to_categorical(label_encoded, num_classes)


def create_cnn_lstm_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Reshape((-1, 128)),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dense(max_length * num_classes, activation='softmax'),
        layers.Reshape((max_length, num_classes))
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Load data
X, y = load_data()

# Khởi tạo model
model = create_cnn_lstm_model()
model.summary()

# Huấn luyện mô hình
from tensorflow.keras.callbacks import EarlyStopping

# Thêm callback early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Huấn luyện mô hình với early stopping
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])