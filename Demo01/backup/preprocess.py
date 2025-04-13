import os

import cv2
import numpy as np

# Bảng ký tự cho tập số từ 0,0 đến 10 (gồm các số và dấu phẩy)
characters = "0123456789,"

# Tạo ánh xạ ký tự sang chỉ số
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Thông số ảnh
img_width = 128
img_height = 32


def encode_label(label):
    """Chuyển label dạng chuỗi (ví dụ '9,4') thành mảng số"""
    return [char_to_index[c] for c in label]


def preprocess_image(image_path):
    """Đọc ảnh, resize và chuẩn hóa"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    return img


def load_dataset(data_dir):
    """Load tất cả ảnh và label"""
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            label = filename.split("_")[0]  # Ví dụ: '9,4_001.png' -> '9,4'
            img_path = os.path.join(data_dir, filename)
            X.append(preprocess_image(img_path))
            y.append(encode_label(label))
    return np.array(X), y


if __name__ == "__main__":
    X, y = load_dataset("../decimal_digits")
    print("Tổng số ảnh:", len(X))
    print("Ảnh shape:", X[0].shape)
    print("1 vài nhãn sau encode:", y[:5])
