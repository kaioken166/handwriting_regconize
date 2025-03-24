import os
import pandas as pd

# Đọc file nhãn gốc
labels_df = pd.read_csv("labels.csv")  # Giả sử có cột: image_name, label

# Thư mục chứa ảnh tăng cường
augmented_folder = "dataset/augmented_images"

# Danh sách ảnh trong thư mục ảnh tăng cường
augmented_data = []

for aug_image in os.listdir(augmented_folder):
    if aug_image.endswith(".png"):  # Đảm bảo chỉ lấy ảnh
        if aug_image in labels_df["image_name"].values:
            label = labels_df[labels_df["image_name"] == aug_image]["label"].values[0]
            augmented_data.append([aug_image, label])

# Tạo DataFrame cho ảnh đã tăng cường
augmented_df = pd.DataFrame(augmented_data, columns=["image_name", "label"])

# Gộp cả ảnh gốc và ảnh tăng cường thành 1 dataset
full_dataset = pd.concat([labels_df, augmented_df])

# Lưu thành file mới
full_dataset.to_csv("augmented_labels.csv", index=False)

print(f"Tổng số ảnh sau khi tăng cường dữ liệu: {len(full_dataset)}")
