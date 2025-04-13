import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# Tắt MKL để tránh lỗi (nếu cần)
os.environ['TF_DISABLE_MKL'] = '1'

# Tính class weights
counts = [3571, 1577, 2636, 1483, 1265, 2915, 1271, 2808, 756, 1957]
labels = np.arange(10)
class_weights = compute_class_weight('balanced', classes=labels, y=np.repeat(labels, counts))
class_weight_dict = dict(enumerate(class_weights))

# Tạo Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    zoom_range=0.4,
    shear_range=0.3,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

# Tạo generator cho tập huấn luyện
train_generator = datagen.flow_from_directory(
    'decimal_part',
    target_size=(56, 132),
    color_mode='grayscale',
    batch_size=8,
    class_mode='categorical',
    subset='training',
    seed=42
)

# Tạo generator cho tập validation
validation_generator = datagen.flow_from_directory(
    'decimal_part',
    target_size=(56, 132),
    color_mode='grayscale',
    batch_size=8,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Xây dựng mô hình
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(56, 132, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Sử dụng Early Stopping và ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Độ chính xác trên tập kiểm tra (decimal_model): {test_acc}")

# Lưu mô hình
model.save('decimal_model.keras')  # Lưu ở định dạng Keras theo khuyến nghị