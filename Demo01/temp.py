import os
import cv2


def remove_special_files(directory):
    """
    Xóa tất cả các file có định dạng X,Y_0_Z.png trong thư mục được chỉ định.

    Args:
        directory (str): Đường dẫn đến thư mục chứa các file.
    """
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra xem file có phải là .png không
        if filename.endswith('.png'):
            # Tách các phần của tên file bằng dấu "_"
            parts = filename.split('_')
            # Kiểm tra định dạng: X,Y_0_Z.png (phải có đúng 3 phần)
            if len(parts) == 3 and parts[1] == '0':
                # Lấy đường dẫn đầy đủ của file
                file_path = os.path.join(directory, parts[0] + '_0_' + parts[2])
                try:
                    # Xóa file
                    os.remove(file_path)
                    print(f"Đã xóa: {file_path}")
                except Exception as e:
                    print(f"Lỗi khi xóa {file_path}: {e}")


def check_special_files(folder_path):
    # folder_path = "../decimal_digits"  # Đường dẫn tới thư mục chứa ảnh
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}  # Các định dạng ảnh hợp lệ

    # Duyệt qua tất cả file trong thư mục
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Kiểm tra phần mở rộng hợp lệ
        _, ext = os.path.splitext(filename)
        if ext.lower() not in valid_extensions:
            print(f"Không phải ảnh hợp lệ: {filename}")
            continue

        # Cố gắng đọc ảnh bằng OpenCV
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Không thể mở ảnh: {filename}")
        else:
            print(f"✅ OK: {filename}")

if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa dữ liệu
    directory = 'decimal_digits'  # Thay bằng đường dẫn thư mục của bạn

    # Gọi hàm để xóa các file
    # remove_special_files(directory)
    # check_special_files(directory)
