# Load_Mnist.py
import os

import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

# Tải tập dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Hàm ghép số thập phân (giới hạn số nguyên từ 0 đến 10 và phần thập phân từ 0-9)
def create_decimal_image():
    idx1 = np.random.choice(np.where(y_train < 10)[0])  # Chọn số nguyên từ 0 đến 9

    # Chọn phần thập phân từ 0 đến 9
    idx2 = np.random.choice(np.where(y_train < 10)[0])  # Chọn phần thập phân từ 0 đến 9

    digit1 = x_train[idx1]  # Phần nguyên
    digit2 = x_train[idx2]  # Phần thập phân

    # Tạo ảnh dấu phẩy (dấu ',')
    comma = np.zeros((28, 10), dtype=np.uint8)  # Một cột đen
    comma[20:28, 4:6] = 255  # Chấm trắng mô phỏng dấu phẩy

    # Ghép ảnh theo chiều ngang
    decimal_image = np.hstack([digit1, comma, digit2])

    # Trả về ảnh và nhãn tương ứng
    return decimal_image, y_train[idx1], y_train[idx2]


# Tạo thư mục lưu ảnh
save_dir = "../decimal_digits"
os.makedirs(save_dir, exist_ok=True)

# Số lượng ảnh cần tạo
num_samples = 1000

# Tạo và lưu ảnh
for i in range(num_samples):
    img, label1, label2 = create_decimal_image()

    # Tạo tên file (ví dụ: "10,2_123.png" nếu label1=10)
    filename = f"{save_dir}/{label1},{label2}_{i}.png"

    # Lưu ảnh
    cv2.imwrite(filename, img)

print(f"Đã tạo {num_samples} ảnh số thập phân và lưu vào {save_dir}")
