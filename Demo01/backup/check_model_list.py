import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load mô hình
model_path = 'decimal_digit_model_2.h5'
model = load_model(model_path)

# Kiểm tra ánh xạ nhãn từ thư mục
datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
    'decimal_digits',
    target_size=(28, 66),
    color_mode='grayscale',
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_indices = generator.class_indices

# Tạo danh sách classes theo ánh xạ của flow_from_directory
classes = [0] * 101
for label, idx in class_indices.items():
    classes[idx] = float(label)
print("Danh sách classes:", classes)

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
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return None, None

    print(f"Kích thước ảnh gốc: {img.shape}")
    if np.mean(img) < 127:
        img = 255 - img
    img_resized = cv2.resize(img, (66, 28))
    print(f"Kích thước ảnh sau resize: {img_resized.shape}")
    img_processed = img_resized.reshape(1, 28, 66, 1) / 255.0
    print(f"Giá trị pixel sau chuẩn hóa: min={img_processed.min()}, max={img_processed.max()}")

    pred = model.predict(img_processed, verbose=0)
    pred_class_index = np.argmax(pred, axis=1)[0]
    predicted_value = classes[pred_class_index]

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
    data_dir = '../decimal_digits'
    num_images_to_predict = 20  # Giữ nguyên 18 ảnh để so sánh
    predict_random_images(data_dir, num_images_to_predict)