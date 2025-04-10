import os
import cv2
import numpy as np

# Đường dẫn thư mục chứa ảnh
data_dir = "../decimal_digits"

# Danh sách lưu ảnh và nhãn
images = []
labels = []

# Duyệt qua toàn bộ file ảnh
for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        # Đọc ảnh ở dạng grayscale
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Thêm ảnh vào danh sách
        images.append(img)

        # Tách nhãn từ tên file
        label = filename.split("_")[0]  # Lấy phần '3,7' từ '3,7_102.png'
        labels.append(label)

print(f"Đã đọc {len(images)} ảnh và {len(labels)} nhãn.")

# Kích thước ảnh đầu vào cho mô hình
img_height = 32
img_width = 128

# Resize và chuẩn hóa ảnh
processed_images = []

for img in images:
    # Resize về kích thước cố định
    resized_img = cv2.resize(img, (img_width, img_height))  # cv2.resize dùng (width, height)

    # Normalize pixel về [0.0, 1.0]
    normalized_img = resized_img.astype(np.float32) / 255.0

    # Thêm kênh chiều sâu (channel=1) để phù hợp với CNN (H, W, 1)
    normalized_img = np.expand_dims(normalized_img, axis=-1)

    processed_images.append(normalized_img)

# Chuyển sang numpy array
X = np.array(processed_images)

print("Kích thước X:", X.shape)  # (số ảnh, 32, 128, 1)

# Lấy tất cả ký tự xuất hiện trong nhãn
unique_chars = set(''.join(labels))  # Tập ký tự: {'0', '1', ..., '9', ','}

# Sắp xếp để ánh xạ dễ
char_list = sorted(list(unique_chars))

# Tạo từ điển ánh xạ ký tự → index
char_to_index = {char: idx for idx, char in enumerate(char_list)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

print("Tập ký tự:", char_list)
print("Số lượng ký tự:", len(char_list))

# Chuyển chuỗi '3,7' → [3, ',', 7] → [index_3, index_comma, index_7]
y_encoded = []

for label in labels:
    encoded = [char_to_index[c] for c in label]
    y_encoded.append(encoded)

print("Ví dụ nhãn gốc:", labels[0])
print("Đã mã hóa:", y_encoded[0])