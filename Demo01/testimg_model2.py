import cv2
import numpy as np
from tensorflow import keras
from PIL import Image

# Tải mô hình đã huấn luyện (nếu chưa có sẵn)
model = keras.models.load_model("my_model2.h5")  # Lưu mô hình sau khi train bằng model.save('model_mnist.h5')

# Đọc ảnh viết tay bên ngoài (cập nhật đường dẫn ảnh)
image_path = "new_image.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Tiền xử lý ảnh
img = cv2.resize(img, (28, 28))  # Resize về 28x28
img = cv2.bitwise_not(img)  # Đảo màu nếu nền đen - chữ trắng
img = img.astype("float32") / 255.0  # Chuẩn hóa về [0,1]
img = img.reshape(1, 28, 28, 1)  # Thêm batch dimension

# Dự đoán chữ số
prediction = model.predict(img)
digit = np.argmax(prediction)

# Hiển thị kết quả
print(f"Mô hình dự đoán: {digit}")
