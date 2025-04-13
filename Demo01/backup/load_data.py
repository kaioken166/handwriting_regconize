import os
import cv2
import numpy as np

def load_decimal_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám
            img = img / 255.0  # Chuẩn hóa
            label = filename.split("_")[0]  # Lấy "label1,label2"
            label1, label2 = map(int, label.split(","))  # Tách nhãn
            images.append(img)
            labels.append([label1, label2])
    return np.array(images), np.array(labels)

X, y = load_decimal_data("../decimal_digits")