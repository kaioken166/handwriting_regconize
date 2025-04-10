import cv2
import numpy as np
import os

def split_image(image_path, output_folder, rows=10, cols=5):  # Đúng số hàng và cột theo thực tế
    # Đọc ảnh gốc
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Không thể đọc ảnh.")
        return
    
    # Lấy kích thước ảnh
    height, width = image.shape
    
    # Tính kích thước mỗi ô (mỗi số là một hình vuông)
    cell_height = height // rows
    cell_width = width // cols
    
    # Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
     # Cắt ảnh thành từng ô và lưu lại
    for i in range(rows):  # Duyệt theo hàng
        for j in range(cols):  # Duyệt theo cột
            # Tính toán tọa độ của ô hiện tại
            y_start, y_end = i * cell_height, (i + 1) * cell_height
            x_start, x_end = j * cell_width, (j + 1) * cell_width
            
            # Cắt ảnh
            cell_image = image[y_start:y_end, x_start:x_end]
            
            # Đổi cách đánh số để đúng định dạng 0,0 -> 4,9
            new_i = i // 2 + 5 # Nhóm thành 5 dòng
            new_j = j + (i % 2) * cols  # Điều chỉnh vị trí cột dựa vào hàng lẻ/chẵn
            
            # Lưu ảnh số thập phân theo thứ tự mong muốn
            filename = os.path.join(output_folder, f"digit2_{new_i},{new_j}.png")
            cv2.imwrite(filename, cell_image)
    
    print(f"Đã tách và lưu {rows * cols} ảnh vào thư mục '{output_folder}'")

# Sử dụng hàm
image_path = "../img/photo_2025-03-24_18-45-14.jpg"  # Thay bằng đường dẫn thực tế
output_folder = "split_digits"
split_image(image_path, output_folder)
