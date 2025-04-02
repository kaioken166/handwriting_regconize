import os
import random

# Thư mục chứa ảnh
save_dir = "decimal_digits"

# Lấy danh sách ảnh số 10,0
image_files = [f for f in os.listdir(save_dir) if f.startswith("10,0")]

# Giữ lại 400 ảnh, xóa bớt phần còn lại
keep_count = 300  # Giữ lại 300 ảnh
files_to_delete = random.sample(image_files, len(image_files) - keep_count)

for file in files_to_delete:
    os.remove(os.path.join(save_dir, file))

print(f"Đã giảm số lượng ảnh 10,0 xuống còn {keep_count} ảnh")
