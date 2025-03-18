import numpy as np
import cv2
import os
import itertools
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageDraw, ImageFont

# Tạo thư mục lưu dữ liệu
os.makedirs("dataset/images", exist_ok=True)

# Tải tập MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chuyển ảnh về định dạng chuẩn
X_train = X_train.astype(np.uint8)
X_test = X_test.astype(np.uint8)


def create_decimal_image(digit1, digit2, idx):
    """Ghép 2 chữ số từ MNIST thành số thập phân dạng 'a,b' và lưu ảnh"""
    img1 = X_train[digit1]  # Lấy chữ số đầu
    img2 = X_train[digit2]  # Lấy chữ số sau dấu phẩy

    # Chuyển ảnh sang PIL để ghép
    img1 = Image.fromarray(img1).resize((28, 28))
    img2 = Image.fromarray(img2).resize((28, 28))

    # Tạo ảnh trắng nền 90x28 (chứa cả số và dấu phẩy)
    img_final = Image.new('L', (90, 28), 255)

    # Vẽ dấu phẩy
    draw = ImageDraw.Draw(img_final)
    draw.text((32, 8), ",", fill=0)  # Vị trí dấu phẩy

    # Ghép ảnh vào
    img_final.paste(img1, (0, 0))
    img_final.paste(img2, (60, 0))

    # Lưu ảnh
    img_final.save(f"dataset/images/{digit1}_{digit2}_{idx}.png")
    return f"{digit1},{digit2}"  # Trả về nhãn số thập phân

num_samples = 5000  # Số lượng mẫu
labels = []  # Danh sách nhãn

for i in range(num_samples):
    digit1, digit2 = np.random.randint(0, 10, size=2)  # Chọn 2 số ngẫu nhiên
    label = create_decimal_image(digit1, digit2, i)
    labels.append(label)

# Lưu nhãn vào file CSV
with open("dataset/labels.csv", "w") as f:
    for i, label in enumerate(labels):
        f.write(f"{i},{label}\n")

print("Tạo dữ liệu hoàn tất!")
