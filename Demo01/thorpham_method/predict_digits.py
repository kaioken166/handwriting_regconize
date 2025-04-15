# predict_digit.py
import tkinter as tk

import joblib
import numpy as np
import win32gui
from PIL import ImageGrab
from skimage.feature import hog


class DigitRecognizerApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Vẽ và Nhận diện Chữ số")
        self.geometry("500x500")

        self.x = self.y = 0

        # Load mô hình SVM
        try:
            self.model = joblib.load('svm_digit_recognizer.pkl')
        except FileNotFoundError:
            tk.messagebox.showerror("Lỗi", "Không tìm thấy file 'svm_digit_recognizer.pkl'!")
            self.quit()

        # Tạo giao diện
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Vẽ chữ số", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Sắp xếp giao diện
        self.canvas.grid(row=0, column=0, pady=2, sticky="W", columnspan=2)
        self.label.grid(row=1, column=0, pady=2, padx=2, columnspan=2)
        self.classify_btn.grid(row=2, column=1, pady=2, padx=2)
        self.button_clear.grid(row=2, column=0, pady=2)

        # Bind sự kiện vẽ
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Vẽ chữ số")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 5  # Độ dày nét vẽ
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill="black")

    def classify_handwriting(self):
        # Lấy handle của canvas
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)

        # Tiền xử lý và dự đoán
        digit = self.predict_digit(im)
        self.label.configure(text=f"Chữ số: {digit}")

    def predict_digit(self, img):
        # Chuyển ảnh về grayscale và resize về 28x28
        img = img.convert('L')
        img = img.resize((28, 28))
        img = np.array(img)

        # Chuẩn hóa và trích xuất HOG
        img = img.astype('float32') / 255.0
        img_uint8 = (img * 255).astype(np.uint8)
        hog_feature = hog(img_uint8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

        # Dự đoán
        digit = self.model.predict([hog_feature])[0]
        return digit


if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.mainloop()
