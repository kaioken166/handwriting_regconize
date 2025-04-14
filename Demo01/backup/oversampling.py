import os
import shutil
import random


def oversample_decimal_part(decimal_dir, target_count=3500):
    """
    Oversampling dữ liệu trong decimal_part để cân bằng số lượng ảnh.

    Args:
        decimal_dir (str): Đường dẫn đến thư mục decimal_part.
        target_count (int): Số lượng ảnh mục tiêu cho mỗi lớp.
    """
    for label in os.listdir(decimal_dir):
        label_path = os.path.join(decimal_dir, label)
        if os.path.isdir(label_path):
            images = [f for f in os.listdir(label_path) if f.endswith('.png')]
            num_images = len(images)
            print(f"Lớp {label}: {num_images} ảnh (trước oversampling)")

            if num_images < target_count:
                # Tính số ảnh cần thêm
                num_to_add = target_count - num_images
                # Lặp lại các ảnh hiện có để thêm vào
                for i in range(num_to_add):
                    img_to_copy = random.choice(images)
                    src_path = os.path.join(label_path, img_to_copy)
                    new_filename = f"{img_to_copy.split('.')[0]}_oversampled_{i}.png"
                    dest_path = os.path.join(label_path, new_filename)
                    shutil.copy(src_path, dest_path)

            # Kiểm tra lại số lượng
            new_num_images = len([f for f in os.listdir(label_path) if f.endswith('.png')])
            print(f"Lớp {label}: {new_num_images} ảnh (sau oversampling)")


if __name__ == "__main__":
    decimal_dir = "../decimal_part"
    oversample_decimal_part(decimal_dir, target_count=3500)