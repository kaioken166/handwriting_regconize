import cv2
import numpy as np
from tensorflow import keras
import os

# Load model chỉ một lần
model = keras.models.load_model("my_model2.h5")

# Danh sách các ảnh cần xử lý
# image_paths = ["img/new_image4.jpg", "img/new_image5.jpg", "img/new_image6.jpg", "img/new_image7.png"]  # Cập nhật danh sách ảnh của bạn

image_dir = "img"
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".jpg", ".png"))]

# Danh sách lưu kết quả dự đoán
results = []

# Lặp qua từng ảnh để xử lý
for image_path in image_paths:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}. Bỏ qua...")
        continue

    # Tiền xử lý ảnh
    img = cv2.resize(img, (28, 28))  # Resize về 28x28
    img = cv2.bitwise_not(img)  # Đảo màu nếu nền đen - chữ trắng
    img = img.astype("float32") / 255.0  # Chuẩn hóa về [0,1]
    img = img.reshape(1, 28, 28, 1)  # Thêm batch dimension

    # Dự đoán chữ số
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    # Lưu kết quả
    results.append((image_path, digit))

# Hiển thị toàn bộ kết quả
print("\nKết quả dự đoán:")
for img_path, pred in results:
    print(f"Ảnh {img_path} ➝ Dự đoán: {pred}")
