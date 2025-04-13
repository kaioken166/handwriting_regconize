import numpy as np
import cv2
import os
from tensorflow.keras.datasets import mnist

# Tải tập dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Tạo thư mục lưu ảnh
save_dir = "../decimal_digits"
os.makedirs(save_dir, exist_ok=True)

# Lấy tất cả các chỉ số của số 1 và số 0
idx_1 = np.where(y_train == 1)[0]  # Chỉ số của số 1
idx_0 = np.where(y_train == 0)[0]  # Chỉ số của số 0

# Giới hạn số lượng ảnh tạo ra là 2000
num_samples = 2000

# Chọn ngẫu nhiên các chỉ số để ghép số 1 với số 0
selected_1 = np.random.choice(idx_1, num_samples, replace=True)  # Chọn ngẫu nhiên số 1
selected_0 = np.random.choice(idx_0, num_samples, replace=True)  # Chọn ngẫu nhiên số 0

# Tạo và lưu ảnh
for i in range(num_samples):
    digit1 = x_train[selected_1[i]]  # Ảnh của số 1
    digit2 = x_train[selected_0[i]]  # Ảnh của số 0

    # Ghép ảnh của số 1 và số 0 lại để tạo thành số 10
    combined_image = np.hstack([digit1, digit2])

    # Tạo tên file (ví dụ: "10,0_123.png")
    filename = f"{save_dir}/10,0_{i}.png"

    # Lưu ảnh
    cv2.imwrite(filename, combined_image)

print(f"Đã tạo {num_samples} ảnh số 10 và lưu vào {save_dir}")
