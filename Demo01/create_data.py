import os

import cv2
import numpy as np
import pandas as pd

# Đọc file labels.csv
labels_df = pd.read_csv("labels.csv")

# Thư mục chứa ảnh
image_folder = "dataset/images"
output_size = (32, 32)  # Resize ảnh về kích thước mong muốn

# Tạo danh sách chứa ảnh và nhãn
images = []
labels = []

# Lặp qua từng dòng trong CSV để load ảnh tương ứng
for index, row in labels_df.iterrows():
    image_path = os.path.join(image_folder, row["image_name"])

    if os.path.exists(image_path):  # Kiểm tra xem file có tồn tại không
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Chuyển về grayscale
        img = cv2.resize(img, output_size)  # Resize ảnh về kích thước cố định
        img = img / 255.0  # Chuẩn hóa pixel về khoảng [0, 1]

        images.append(img)
        labels.append(row["label"])  # Lưu nhãn tương ứng

# Chuyển danh sách thành numpy array
images = np.array(images).reshape(-1, 32, 32, 1)  # Thêm kênh màu (grayscale)
labels = np.array(labels, dtype=np.float32)  # Chuyển nhãn thành số thực

print(f"Dataset size: {len(images)} samples")

from sklearn.model_selection import train_test_split

# Chia dữ liệu thành 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

from tensorflow.keras import layers, models

# Xây dựng mô hình CNN đơn giản
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Dự đoán số thực
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Sử dụng MSE vì nhãn là số thực

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

import matplotlib.pyplot as plt

# Chọn một ảnh ngẫu nhiên từ tập test
idx = np.random.randint(len(X_test))
sample_img = X_test[idx]

# Dự đoán
predicted_value = model.predict(sample_img.reshape(1, 32, 32, 1))[0][0]
true_value = y_test[idx]

# Hiển thị ảnh
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f"True: {true_value:.2f}, Predicted: {predicted_value:.2f}")
plt.show()
