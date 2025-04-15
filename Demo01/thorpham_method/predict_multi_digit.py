# predict_multi_digit.py
import tkinter as tk
from tkinter import filedialog

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import win32gui
from PIL import Image, ImageGrab
from skimage.feature import hog


class MultiDigitRecognizerApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Vẽ và Nhận diện Nhiều Chữ số")
        self.geometry("900x600")

        self.x = self.y = 0

        # Load mô hình SVM
        try:
            self.model = joblib.load('svm_digit_recognizer.pkl')
        except FileNotFoundError:
            tk.messagebox.showerror("Lỗi", "Không tìm thấy file 'svm_digit_recognizer.pkl'!")
            self.quit()

        # Tạo giao diện
        self.canvas = tk.Canvas(self, width=800, height=400, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Vẽ các chữ số", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.button_image = tk.Button(self, text="Recognize from Image", command=self.recognize_from_image)

        # Sắp xếp giao diện
        self.canvas.grid(row=0, column=0, pady=2, sticky="W", columnspan=3)
        self.label.grid(row=1, column=0, pady=2, padx=2, columnspan=3)
        self.classify_btn.grid(row=2, column=1, pady=2, padx=2)
        self.button_clear.grid(row=2, column=0, pady=2)
        self.button_image.grid(row=2, column=2, pady=2, padx=2)

        # Bind sự kiện vẽ
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Vẽ các chữ số")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 5
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill="black")

    def classify_handwriting(self):
        # Lấy ảnh từ canvas
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        img = ImageGrab.grab(rect)

        # Nhận diện nhiều chữ số
        digits = self.predict_multiple_digits(img, debug=False)
        if digits:
            self.label.configure(text=f"Dự đoán: {''.join(map(str, digits))}")
        else:
            self.label.configure(text="Không tìm thấy chữ số")

        # Hiển thị ảnh với khung và nhãn
        self.show_result(img, digits)

    def recognize_from_image(self):
        # Mở hộp thoại chọn ảnh
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not file_path:
            return

        # Đọc ảnh
        try:
            img = Image.open(file_path)
        except Exception as e:
            tk.messagebox.showerror("Lỗi", f"Không thể mở ảnh: {str(e)}")
            return

        # Nhận diện nhiều chữ số
        digits = self.predict_multiple_digits(img, debug=True)
        if digits:
            self.label.configure(text=f"Dự đoán: {''.join(map(str, digits))}")
        else:
            self.label.configure(text="Không tìm thấy chữ số")

        # Hiển thị ảnh với khung và nhãn
        self.show_result(img, digits)

    def predict_multiple_digits(self, img, debug=False):
        # Chuyển sang grayscale
        img = img.convert('L')
        gray = np.array(img)

        # Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Nhị phân hóa
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)

        # Debug: Hiển thị ảnh nhị phân hóa
        if debug:
            plt.figure(figsize=(10, 5))
            plt.imshow(thresh, cmap='gray')
            plt.title("Ảnh nhị phân hóa (Debug)")
            plt.axis('off')
            plt.show()

        # Tìm contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sắp xếp contours từ trái sang phải
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        predictions = []
        self.contour_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Nới lỏng ngưỡng contour
            if w < 10 or h < 10 or w > 600 or h > 600:
                continue

            # Trích xuất chữ số
            digit = gray[y:y + h, x:x + w]

            # Thêm padding tỷ lệ
            padding = max(w, h) // 4
            digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding,
                                       cv2.BORDER_CONSTANT, value=255)

            # Resize về 28x28
            digit = cv2.resize(digit, (28, 28))

            # Chuẩn hóa và trích xuất HOG
            digit = digit.astype('float32') / 255.0
            digit_uint8 = (digit * 255).astype(np.uint8)
            hog_feature = hog(digit_uint8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              block_norm='L2-Hys')

            # Dự đoán
            pred = self.model.predict([hog_feature])[0]
            predictions.append(pred)
            self.contour_boxes.append((x, y, w, h))

        return predictions

    def show_result(self, img, digits):
        # Chuyển PIL Image sang OpenCV
        img_cv = np.array(img.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Vẽ khung và nhãn
        for i, (x, y, w, h) in enumerate(self.contour_boxes):
            if i < len(digits):
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_cv, str(digits[i]), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Hiển thị ảnh
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.title(f"Dự đoán: {''.join(map(str, digits))}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    app = MultiDigitRecognizerApp()
    app.mainloop()
