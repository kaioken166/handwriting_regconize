import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Hàm tạo và huấn luyện mô hình
def train_model(data_dir, num_classes, model_name):
    # Tạo Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Tạo generator cho tập huấn luyện
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(28, 66),
        color_mode='grayscale',
        batch_size=16,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    # Tạo generator cho tập validation
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(28, 66),
        color_mode='grayscale',
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    # Xây dựng mô hình
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 66, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    # Biên dịch mô hình
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Sử dụng Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )

    # Đánh giá mô hình
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f"Độ chính xác trên tập kiểm tra ({model_name}): {test_acc}")

    # Lưu mô hình
    model.save(f"{model_name}.h5")

if __name__ == "__main__":
    # Huấn luyện mô hình cho phần nguyên (0-10)
    print("Huấn luyện mô hình cho phần nguyên...")
    train_model('integer_part', num_classes=11, model_name='integer_model.h5')

    # Huấn luyện mô hình cho phần thập phân (0-9)
    print("\nHuấn luyện mô hình cho phần thập phân...")
    train_model('decimal_part', num_classes=10, model_name='decimal_model.h5')