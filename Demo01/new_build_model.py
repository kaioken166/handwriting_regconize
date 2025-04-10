import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


def load_data(data_dir, target_size=(28, 66)):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            # Trích xuất nhãn từ tên file (ví dụ: "3,5_123.png" -> 3,5)
            label_str = filename.split('_')[0]
            if ',' in label_str:
                integer_part, decimal_part = label_str.split(',')
                label = float(integer_part + '.' + decimal_part)
            else:
                label = float(label_str)
            # Đọc ảnh
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Resize ảnh về kích thước cố định
            img_resized = cv2.resize(img, target_size)
            images.append(img_resized)
            labels.append(label)
    return np.array(images), np.array(labels)


# Giả sử thư mục chứa ảnh của bạn là "decimal_digits"
data_dir = "decimal_digits"
images, labels = load_data(data_dir)

# Chuẩn hóa nhãn thành các chỉ số từ 0 đến 100
unique_labels = np.unique(labels)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
labels_idx = np.array([label_to_index[label] for label in labels])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(images, labels_idx, test_size=0.2, random_state=42)

# Reshape ảnh và chuẩn hóa giá trị pixel
X_train = X_train.reshape(-1, 28, 66, 1) / 255.0
X_test = X_test.reshape(-1, 28, 66, 1) / 255.0

# Mã hóa one-hot cho nhãn
y_train_onehot = to_categorical(y_train, num_classes=101)
y_test_onehot = to_categorical(y_test, num_classes=101)

model = models.Sequential([
    # Lớp tích chập đầu tiên
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 66, 1)),
    layers.MaxPooling2D((2, 2)),
    # Lớp tích chập thứ hai
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Lớp tích chập thứ ba
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Flatten để chuyển từ tensor 2D sang vector 1D
    layers.Flatten(),
    # Lớp fully connected
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Thêm Dropout với tỷ lệ 50%
    # Lớp đầu ra với 101 lớp (0.0 đến 10.0)
    layers.Dense(101, activation='softmax')
])

# Xem tóm tắt kiến trúc mô hình
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Huấn luyện mô hình
history = model.fit(X_train, y_train_onehot, epochs=20, validation_split=0.1, callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f"Độ chính xác trên tập kiểm tra: {test_acc}")

model.save('decimal_digit_model.h5')
