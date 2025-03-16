import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('model/my_model2.h5')

# Đọc và tiền xử lý hình ảnh
image = cv2.imread('img/new_image4.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.astype('float32') / 255
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

# Dự đoán chữ số
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)
print(f'Predicted digit: {prediction}')
print(f'Predicted digit: {predicted_digit}')