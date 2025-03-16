import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

model = keras.models.load_model("model/my_model2.h5")  # Load model chữ số

def extract_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)  # Chuyển nền trắng - chữ đen

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char = img[y:y + h, x:x + w]  # Cắt từng ký tự
        char = cv2.resize(char, (28, 28))  # Resize về kích thước 28x28
        char = cv2.bitwise_not(char)
        char_images.append((x, char))  # Lưu theo vị trí x để sắp xếp đúng thứ tự

    char_images.sort()  # Sắp xếp theo vị trí từ trái sang phải
    extracted_chars = [char for _, char in char_images]

    # 📌 Hiển thị các ảnh đã cắt để kiểm tra
    plt.figure(figsize=(10, 2))
    for i, char in enumerate(extracted_chars):
        plt.subplot(1, len(extracted_chars), i + 1)
        plt.imshow(char, cmap='gray')
        plt.axis("off")
    plt.show()

    return extracted_chars  # Trả về danh sách ảnh đã tiền xử lý


def recognize_characters(char_images): # tách từng ký tự
    predictions = []

    for char in char_images:
        char = char.astype("float32") / 255.0  # Chuẩn hóa ảnh
        char = char.reshape(1, 28, 28, 1)  # Định dạng đúng đầu vào của model

        # Bắt đầu đoán tại đây
        prediction = model.predict(char)
        digit = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
        predictions.append(digit)

    return predictions  # Trả về danh sách các chữ số đã nhận diện


def predict_characters(char_images): # có dấu phẩy
    result = []
    for char in char_images:
        char = char.astype("float32") / 255.0  # Chuẩn hóa
        char = char.reshape(1, 28, 28, 1)  # Định dạng đúng đầu vào model

        prediction = model.predict(char)
        digit = np.argmax(prediction)
        result.append(str(digit) if digit < 10 else ",")  # Nếu class = 10 thì là dấu `,`
    return "".join(result)


# print(f"Kết quả nhận diện: {predicted_number}")
image_path = "img/image1.jpg"  # Đường dẫn ảnh chứa số thập phân

# Bước 1: Tách ký tự trong ảnh
char_images = extract_characters(image_path)

# Bước 2: Nhận diện từng ký tự
recognized_digits = predict_characters(char_images)

# Bước 3: In kết quả từng ký tự
print(f"Kết quả nhận diện từng ký tự: {recognized_digits}")
