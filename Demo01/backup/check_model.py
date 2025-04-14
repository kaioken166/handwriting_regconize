import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load mô hình
model_path = 'decimal_digit_model_2.h5'
model = load_model(model_path)

# Danh sách classes (từ 0.0 đến 10.0 với bước 0.1)
classes = [round(i * 0.1, 1) for i in range(101)]  # [0.0, 0.1, ..., 10.0]

class DecimalDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Decimal Digit Recognition")
        self.root.geometry("400x500")  # Đặt kích thước cửa sổ

        # Tạo giao diện
        self.label = tk.Label(root, text="Tải ảnh để dự đoán số thập phân", font=("Arial", 14))
        self.label.pack(pady=10)

        # Nút tải ảnh
        self.upload_button = tk.Button(root, text="Tải ảnh", command=self.upload_image, font=("Arial", 12))
        self.upload_button.pack(pady=5)

        # Nút xóa
        self.clear_button = tk.Button(root, text="Xóa", command=self.clear, font=("Arial", 12))
        self.clear_button.pack(pady=5)

        # Nhãn hiển thị kết quả
        self.result_label = tk.Label(root, text="Kết quả: Chưa có dự đoán", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=10)

        # Nhãn hiển thị ảnh
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def upload_image(self):
        # Mở hộp thoại để chọn file ảnh
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        # Đọc và xử lý ảnh
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Lỗi", f"Không thể đọc ảnh từ: {file_path}")
            return

        # Đảo màu nếu cần
        if np.mean(img) < 127:
            img = 255 - img

        # Resize ảnh về kích thước 28x66 để dự đoán
        img_resized = cv2.resize(img, (66, 28))

        # Reshape và chuẩn hóa ảnh
        img_processed = img_resized.reshape(1, 28, 66, 1) / 255.0

        # Dự đoán
        pred = model.predict(img_processed, verbose=0)
        pred_class_index = np.argmax(pred, axis=1)[0]
        predicted_value = classes[pred_class_index]

        # Cập nhật kết quả trên giao diện
        self.result_label.config(text=f"Kết quả: {predicted_value}")

        # Hiển thị ảnh trên GUI (phóng to để dễ nhìn)
        img_display = cv2.resize(img, (132, 56))  # Phóng to gấp đôi
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display)
        self.image_label.config(image=img_display)
        self.image_label.image = img_display  # Giữ tham chiếu để ảnh không bị xóa

    def clear(self):
        # Xóa kết quả và ảnh hiển thị
        self.result_label.config(text="Kết quả: Chưa có dự đoán")
        self.image_label.config(image='')
        self.image_label.image = None  # Xóa tham chiếu ảnh

if __name__ == "__main__":
    root = tk.Tk()
    app = DecimalDigitApp(root)
    root.mainloop()