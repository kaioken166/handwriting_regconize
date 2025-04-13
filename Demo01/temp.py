import os
import shutil
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def rename_files(directory):
    """
    Đổi tên các file từ định dạng 'digit_X,Y_A_B.png' thành 'X,Y_B.png'.

    Args:
        directory (str): Đường dẫn đến thư mục chứa các file.
    """
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra xem file có đúng định dạng không
        if filename.startswith('digit2_') and filename.endswith('.png'):
            # Tách các phần của tên file
            parts = filename.split('_')
            if len(parts) == 4:  # digit_X,Y_A_B.png
                label = parts[1]  # Lấy X,Y (ví dụ: 4,5)
                index = parts[3].split('.')[0]  # Lấy B (ví dụ: 6443), bỏ .png
                new_filename = f"{label}_{index}.png"  # Tạo tên mới: X,Y_B.png
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                try:
                    os.rename(old_path, new_path)
                    print(f"Đã đổi tên: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Lỗi khi đổi tên {filename}: {e}")


def rename_files2(directory):
    """
    Đổi tên các file trong thư mục:
    - Từ 'digit_X,Y_A_B.png' thành 'X,Y_B.png'.
    - Từ 'digit_X,Y.png' thành 'X,Y_{i}.png' với i là chỉ số tuần tự.

    Args:
        directory (str): Đường dẫn đến thư mục chứa các file.
    """
    for filename in os.listdir(directory):
        if filename.startswith('digit2_') and filename.endswith('.png'):
            parts = filename.split('_')
            if len(parts) == 4:  # Trường hợp: digit_X,Y_A_B.png
                label = parts[1]  # Lấy X,Y (ví dụ: 4,5)
                index = parts[3].split('.')[0]  # Lấy B (ví dụ: 6443)
                new_filename = f"{label}_{index}.png"
            elif len(parts) == 2:  # Trường hợp: digit_X,Y.png
                label = parts[1].split('.')[0]  # Lấy X,Y (ví dụ: 0,9)
                # Đếm số file hiện có với label để tạo chỉ số i
                i = len([f for f in os.listdir(directory) if f.startswith(label)])
                new_filename = f"{label}_{i + 11}.png"
            else:
                print(f"Bỏ qua file không hợp lệ: {filename}")
                continue

            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            try:
                os.rename(old_path, new_path)
                print(f"Đã đổi tên: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Lỗi khi đổi tên {filename}: {e}")


def move_and_rename_files(source_dir, dest_dir):
    """
    Chuyển file từ thư mục nguồn sang thư mục đích, đổi tên nếu trùng.

    Args:
        source_dir (str): Đường dẫn đến thư mục nguồn (dataset/images).
        dest_dir (str): Đường dẫn đến thư mục đích (decimal_digits).
    """
    # Đảm bảo thư mục đích tồn tại
    os.makedirs(dest_dir, exist_ok=True)

    # Duyệt qua tất cả file trong thư mục nguồn
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            # Đường dẫn đầy đủ của file nguồn
            source_path = os.path.join(source_dir, filename)

            # Tách label và index từ tên file
            try:
                label, index = filename.rsplit('_', 1)
                index = int(index.split('.')[0])  # Lấy số index, bỏ .png
            except ValueError:
                print(f"Tên file không hợp lệ: {filename}. Bỏ qua.")
                continue

            # Tạo tên file ban đầu trong thư mục đích
            new_filename = filename
            new_path = os.path.join(dest_dir, new_filename)

            # Kiểm tra xem tên file đã tồn tại trong thư mục đích chưa
            while os.path.exists(new_path):
                # Nếu trùng, tăng index lên cho đến khi không trùng
                index += 1
                new_filename = f"{label}_{index}.png"
                new_path = os.path.join(dest_dir, new_filename)

            # Chuyển file từ nguồn sang đích
            try:
                shutil.move(source_path, new_path)
                print(f"Đã chuyển: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Lỗi khi chuyển {filename}: {e}")


def create_directory_structure(data_dir):
    """
    Tạo cấu trúc thư mục và di chuyển các file ảnh vào thư mục con tương ứng với nhãn.

    Args:
        data_dir (str): Đường dẫn đến thư mục chứa các ảnh (decimal_digits).
    """
    # Tạo danh sách các nhãn từ 0.0 đến 10.0
    labels = [f"{i / 10:.1f}" for i in range(101)]  # ['0.0', '0.1', ..., '10.0']

    # Tạo thư mục con cho từng nhãn nếu chưa tồn tại
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            print(f"Đã tạo thư mục: {label_dir}")

    # Duyệt qua các file trong thư mục gốc
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            # Trích xuất nhãn từ tên file (ví dụ: "0,1_1.png" -> "0.1")
            label_str = filename.split('_')[0].replace(',', '.')
            try:
                label = float(label_str)
                label_formatted = f"{label:.1f}"  # Định dạng nhãn thành "0.1", "10.0", ...
            except ValueError:
                print(f"Tên file không hợp lệ: {filename}")
                continue

            # Đường dẫn file gốc và đích
            src_path = os.path.join(data_dir, filename)
            dst_dir = os.path.join(data_dir, label_formatted)
            dst_path = os.path.join(dst_dir, filename)

            # Di chuyển file vào thư mục con
            if os.path.isfile(src_path):  # Kiểm tra file có tồn tại
                shutil.move(src_path, dst_path)
                print(f"Đã di chuyển: {filename} -> {dst_dir}")


def organize_data_for_two_step(data_dir, integer_dir, decimal_dir):
    """
    Tổ chức lại dữ liệu thành hai thư mục: integer_part và decimal_part.

    Args:
        data_dir (str): Đường dẫn đến thư mục decimal_digits.
        integer_dir (str): Đường dẫn đến thư mục integer_part.
        decimal_dir (str): Đường dẫn đến thư mục decimal_part.
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(integer_dir):
        os.makedirs(integer_dir)
    if not os.path.exists(decimal_dir):
        os.makedirs(decimal_dir)

    # Tạo thư mục con cho phần nguyên (0-10)
    for i in range(11):
        integer_subdir = os.path.join(integer_dir, str(i))
        if not os.path.exists(integer_subdir):
            os.makedirs(integer_subdir)

    # Tạo thư mục con cho phần thập phân (0-9)
    for i in range(10):
        decimal_subdir = os.path.join(decimal_dir, str(i))
        if not os.path.exists(decimal_subdir):
            os.makedirs(decimal_subdir)

    # Duyệt qua tất cả ảnh trong decimal_digits
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if os.path.isdir(label_path):
            # Lấy phần nguyên và phần thập phân từ nhãn
            num = float(label_folder)
            integer_part = int(num)  # Phần nguyên: 0, 1, ..., 10
            decimal_part = int((num - integer_part) * 10)  # Phần thập phân: 0, 1, ..., 9

            # Sao chép ảnh vào thư mục tương ứng
            for filename in os.listdir(label_path):
                if filename.endswith(".png"):
                    src_path = os.path.join(label_path, filename)

                    # Sao chép vào integer_part
                    integer_dest_dir = os.path.join(integer_dir, str(integer_part))
                    integer_dest_path = os.path.join(integer_dest_dir, filename)
                    shutil.copy(src_path, integer_dest_path)

                    # Sao chép vào decimal_part
                    decimal_dest_dir = os.path.join(decimal_dir, str(decimal_part))
                    decimal_dest_path = os.path.join(decimal_dest_dir, filename)
                    shutil.copy(src_path, decimal_dest_path)

def check_number():
    decimal_dir_2 = "decimal_part"
    for label in os.listdir(decimal_dir_2):
        label_path = os.path.join(decimal_dir_2, label)
        if os.path.isdir(label_path):
            num_images = len([f for f in os.listdir(label_path) if f.endswith('.png')])
            print(f"Lớp {label}: {num_images} ảnh")

if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa dữ liệu
    # directory = 'decimal_digits'  # Thay bằng đường dẫn thư mục của bạn

    # Gọi hàm để xóa các file
    # remove_special_files(directory)
    # check_special_files(directory)
    # rename_files('dataset/images')
    # rename_files2('dataset/images')
    # source_dir = 'dataset/images'  # Thư mục chứa ảnh viết tay
    # dest_dir = 'decimal_digits'  # Thư mục chứa ảnh MNIST

    # Gọi hàm để chuyển file
    # move_and_rename_files(source_dir, dest_dir)
    # create_directory_structure(directory)

    # Tạo generator để kiểm tra ánh xạ nhãn
    # datagen = ImageDataGenerator()
    # generator = datagen.flow_from_directory(
    #     'decimal_digits',
    #     target_size=(28, 66),
    #     color_mode='grayscale',
    #     batch_size=1,
    #     class_mode='categorical',
    #     shuffle=False
    # )

    # # In ánh xạ nhãn
    # print("Ánh xạ nhãn từ thư mục:")
    # print(generator.class_indices)

    # data_dir = "decimal_digits"
    # integer_dir = "integer_part"
    # decimal_dir = "decimal_part"
    # organize_data_for_two_step(data_dir, integer_dir, decimal_dir)
    # print("Đã tổ chức lại dữ liệu thành công!")

    check_number()