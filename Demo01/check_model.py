import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Đường dẫn đến file mô hình (cố định, bạn có thể thay đổi nếu cần)
model_path = 'decimal_digit_model.h5'
# Load mô hình
model = load_model(model_path)


def predict_decimal_digit(image_path):
    """
    Hàm dự đoán số thập phân từ ảnh đầu vào sử dụng mô hình đã huấn luyện.

    Args:
        image_path (str): Đường dẫn đến ảnh cần dự đoán.
    """

    # Danh sách classes (từ 0.0 đến 10.0 với bước 0.1)
    classes = [round(i * 0.1, 1) for i in range(101)]  # [0.0, 0.1, ..., 10.0]

    # Đọc ảnh dưới dạng grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Resize ảnh về kích thước 28x66 (điều chỉnh nếu mô hình của bạn yêu cầu khác)
    img_resized = cv2.resize(img, (66, 28))

    # Reshape và chuẩn hóa ảnh
    img_processed = img_resized.reshape(1, 28, 66, 1) / 255.0

    # Dự đoán lớp của ảnh
    pred = model.predict(img_processed)
    print(f"Xác suất dự đoán: {pred[0]}")
    pred_class_index = np.argmax(pred, axis=1)[0]
    print(f"Chỉ số lớp dự đoán: {pred_class_index}")

    # Ánh xạ chỉ số lớp sang giá trị thực
    predicted_value = classes[pred_class_index]

    # In kết quả ra console
    print(f"Đường dẫn ảnh: {image_path}")
    print(f"Kết quả dự đoán: {predicted_value}")


if __name__ == "__main__":
    # Đường dẫn ảnh cần dự đoán
    image_path = 'decimal_digits/7,3_455.png'  # Thay bằng đường dẫn thực tế

    # Gọi hàm dự đoán
    predict_decimal_digit(image_path)


