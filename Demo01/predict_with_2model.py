import os
import random

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load hai mô hình
integer_model = load_model('integer_model.h5')
decimal_model = load_model('decimal_model.h5')


def get_all_images(data_dir):
    image_paths = []
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith(".png"):
                    image_path = os.path.join(label_path, filename)
                    image_paths.append(image_path)
    return image_paths


def predict_decimal_digit(image_path):
    # Đọc và tiền xử lý ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return None, None

    img_resized = cv2.resize(img, (66, 28))
    img_processed = img_resized.reshape(1, 28, 66, 1) / 255.0

    # Dự đoán phần nguyên
    integer_pred = integer_model.predict(img_processed, verbose=0)
    integer_class = np.argmax(integer_pred, axis=1)[0]  # 0-10

    # Dự đoán phần thập phân
    decimal_pred = decimal_model.predict(img_processed, verbose=0)
    decimal_class = np.argmax(decimal_pred, axis=1)[0]  # 0-9

    # Kết hợp thành số thập phân
    predicted_value = float(f"{integer_class}.{decimal_class}")

    # Lấy nhãn thực tế
    label_str = os.path.basename(image_path).split('_')[0].replace(',', '.')
    actual_value = float(label_str)

    return predicted_value, actual_value


def predict_random_images(data_dir, num_images=10):
    image_paths = get_all_images(data_dir)
    total_images = len(image_paths)
    print(f"Tổng số ảnh trong thư mục: {total_images}")

    if num_images > total_images:
        num_images = total_images
    selected_images = random.sample(image_paths, num_images)

    print("\nKết quả dự đoán:")
    print("-" * 50)
    correct_predictions = 0
    for idx, image_path in enumerate(selected_images, 1):
        predicted_value, actual_value = predict_decimal_digit(image_path)
        if predicted_value is None or actual_value is None:
            continue

        is_correct = predicted_value == actual_value
        if is_correct:
            correct_predictions += 1

        print(f"Ảnh {idx}: {image_path}")
        print(f"Nhãn thực tế: {actual_value}")
        print(f"Nhãn dự đoán: {predicted_value}")
        print(f"Dự đoán {'đúng' if is_correct else 'sai'}")
        print("-" * 50)

    accuracy = (correct_predictions / num_images) * 100
    print(f"\nĐộ chính xác trên {num_images} ảnh được chọn: {accuracy:.2f}%")


if __name__ == "__main__":
    data_dir = 'decimal_digits'
    num_images_to_predict = 18
    predict_random_images(data_dir, num_images_to_predict)
