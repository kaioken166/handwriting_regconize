import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# 1. Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Tiền xử lý dữ liệu
x_train, x_test = x_train / 255.0, x_test / 255.0  # Chuẩn hóa về [0,1]
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape cho CNN
x_test = x_test.reshape(-1, 28, 28, 1)

model = keras.models.load_model("model/my_model2.h5")

predictions = model.predict(x_test[:15])

# Hiển thị kết quả
for i in range(15):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Dự đoán: {np.argmax(predictions[i])}, Thực tế: {y_test[i]}")
    plt.axis('off')
    plt.show()
