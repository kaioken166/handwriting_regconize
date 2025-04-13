from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Tạo Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.0,
    fill_mode='nearest',
    validation_split=0.2  # Chia 80% train, 20% validation
)

# Tạo generator cho tập huấn luyện
train_generator = datagen.flow_from_directory(
    'decimal_digits',
    target_size=(28, 66),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='training',
    seed=42
)

# Tạo generator cho tập validation
validation_generator = datagen.flow_from_directory(
    'decimal_digits',
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
    layers.Dense(101, activation='softmax')
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
print(f"Độ chính xác trên tập kiểm tra: {test_acc}")

# Lưu mô hình
model.save('decimal_digit_model_2.h5')
