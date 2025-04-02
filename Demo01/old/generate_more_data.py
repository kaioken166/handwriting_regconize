from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Cấu hình tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=15,  # Xoay ảnh ±15 độ
    width_shift_range=0.2,  # Dịch ngang tối đa 20% ảnh
    height_shift_range=0.2,  # Dịch dọc tối đa 20% ảnh
    zoom_range=0.2,  # Phóng to/thu nhỏ tối đa 20%
    shear_range=10,  # Biến dạng góc tối đa 10 độ
    brightness_range=[0.8, 1.2],  # Điều chỉnh độ sáng ngẫu nhiên
    fill_mode='nearest'  # Điền các pixel bị mất khi xoay/dịch chuyển
)

# Thư mục chứa ảnh gốc
image_folder = "dataset/images"
output_folder = "dataset/images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load ảnh và tạo thêm dữ liệu
for image_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, image_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = img.reshape((1, img.shape[0], img.shape[1], 1))  # Thêm batch dimension
    save_prefix = image_name.split(".")[0]

    # Tạo thêm 10 ảnh từ mỗi ảnh gốc
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=output_folder,
                              save_prefix=save_prefix, save_format="png"):
        i += 1
        if i >= 10:
            break  # Dừng sau khi tạo 10 ảnh mới
