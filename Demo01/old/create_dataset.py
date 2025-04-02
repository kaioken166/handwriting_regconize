import os
import pandas as pd

# Thư mục ảnh gốc và ảnh tăng cường
original_folder = "dataset/images"
augmented_folder = "dataset/augmented_images"

# Danh sách ảnh từ cả hai thư mục
original_images = [f for f in os.listdir(original_folder) if f.endswith(".png") or f.endswith(".jpg")]
augmented_images = [f for f in os.listdir(augmented_folder) if f.endswith(".png")]

# Hàm trích xuất nhãn từ tên ảnh
def extract_label(image_name):
    try:
        # Lấy phần sau "digit_" và trước dấu "_" đầu tiên
        label = image_name.split("digit_")[1].split("_")[0]
        return label
    except IndexError:
        return None  # Nếu lỗi, trả về None để bỏ qua ảnh đó

# Tạo danh sách dữ liệu ảnh và nhãn
data = []

for image in original_images + augmented_images:
    label = extract_label(image)
    if label is not None:  # Đảm bảo ảnh có nhãn hợp lệ
        data.append([image, label])

# Chuyển dữ liệu thành DataFrame
labels_df = pd.DataFrame(data, columns=["image_name", "label"])

# Lưu thành file labels.csv
labels_df.to_csv("labels.csv", index=False)

print(f"Đã tạo labels.csv với {len(labels_df)} ảnh.")
print(labels_df.head())  # Xem 5 dòng đầu tiên
