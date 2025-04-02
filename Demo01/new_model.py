import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Đọc file labels.csv
df = pd.read_csv("labels.csv")

# Tách tập train và validation (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Image size và batch size
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Data Augmentation & Rescaling
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Tạo train generator
train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col="image_name",
    y_col="label",
    directory="dataset",  # Chứa cả original_images & augmented_images
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Tạo validation generator
val_generator = datagen.flow_from_dataframe(
    val_df,
    x_col="image_name",
    y_col="label",
    directory="dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Xây dựng kiến trúc CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(df["label"].unique()), activation='softmax')  # Số lớp output = số nhãn
])

# Compile mô hình
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Hiển thị kiến trúc mô hình
model.summary()

EPOCHS = 20

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Lưu mô hình sau khi train
model.save("handwritten_decimal_model.h5")

import matplotlib.pyplot as plt

# Vẽ đồ thị loss và accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")

plt.show()
