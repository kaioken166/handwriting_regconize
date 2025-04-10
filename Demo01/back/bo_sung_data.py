import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tạo bộ biến đổi dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,  # Xoay ±10 độ
    width_shift_range=0.1,  # Dịch chuyển ngang
    height_shift_range=0.1,  # Dịch chuyển dọc
    shear_range=0.1,  # Biến dạng
    zoom_range=0.1  # Phóng to / thu nhỏ nhẹ
)

# Thư mục chứa ảnh
save_dir = "../decimal_digits"
target_count = 250  # Số lượng ảnh mong muốn cho mỗi số

# Tạo danh sách các số có ít hơn 200 ảnh
numbers_to_augment = []
all_files = os.listdir(save_dir)

# Duyệt qua tất cả số từ 0.0 đến 10.0
for i in range(11):
    for j in range(10):
        num_label = f"{i},{j}"  # Định dạng số (ví dụ: "3,7")
        count = sum(1 for f in all_files if f.startswith(num_label))

        if count < target_count:
            numbers_to_augment.append((num_label, count))

# Nếu có số cần bổ sung dữ liệu
if numbers_to_augment:
    print("Các số cần tăng cường dữ liệu:", numbers_to_augment)

    # Tăng cường dữ liệu
    for num_label, count in numbers_to_augment:
        needed = target_count - count
        image_files = [f for f in all_files if f.startswith(num_label)]

        while len(image_files) < target_count:
            for file in image_files:
                img = cv2.imread(os.path.join(save_dir, file), cv2.IMREAD_GRAYSCALE)

                # Chuyển ảnh thành dạng (28, width, 1)
                img = np.expand_dims(img, axis=-1)  # Thêm kênh màu (grayscale)
                img = np.expand_dims(img, axis=0)  # Thêm batch dimension (1, 28, width, 1)

                # Tạo ảnh mới và tự lưu với tên tùy chỉnh
                for batch in datagen.flow(img, batch_size=1):
                    print("Vào\n")
                    # batch[0] có shape (28, width, 1), cần chuyển về dạng ảnh 2D
                    new_img = batch[0].astype(np.uint8).squeeze()

                    # Tạo tên file mới: ví dụ "0,0_aug123.png"
                    save_name = f"{num_label}_{len(image_files)}.png"
                    save_path = os.path.join(save_dir, save_name)

                    cv2.imwrite(save_path, new_img)

                    break  # chỉ tạo 1 ảnh mỗi lần

                # Cập nhật lại danh sách ảnh sau khi thêm ảnh mới
                all_files = os.listdir(save_dir)
                image_files = [f for f in all_files if f.startswith(num_label)]

                # Nếu đủ số lượng thì dừng
                if len(image_files) >= target_count:
                    break

    print("Đã tăng cường dữ liệu cho các số có ít ảnh.")
else:
    print("Không có số nào cần bổ sung dữ liệu.")
