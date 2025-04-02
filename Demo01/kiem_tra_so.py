import matplotlib.pyplot as plt
import cv2
import os
import random

# Lấy danh sách các tệp ảnh trong thư mục dữ liệu
save_dir = "decimal_digits"
image_files = os.listdir(save_dir)

# Chọn ngẫu nhiên 5 ảnh để hiển thị
sample_files = random.sample(image_files, 5)

# Hiển thị ảnh
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, file in enumerate(sample_files):
    img = cv2.imread(os.path.join(save_dir, file), cv2.IMREAD_GRAYSCALE)
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(file)  # Hiển thị tên file (nhãn)
    axes[i].axis("off")

plt.show()

from collections import Counter

# Đếm số lượng ảnh của mỗi nhãn
labels = [file.split("_")[0] for file in image_files]  # Lấy phần nhãn từ tên file
label_counts = Counter(labels)

# Hiển thị số lượng mỗi nhãn
for label, count in sorted(label_counts.items()):
    print(f"Số {label}: {count} ảnh")
