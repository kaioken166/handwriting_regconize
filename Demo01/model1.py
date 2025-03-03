import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Tiền xử lý dữ liệu
x_train, x_test = x_train / 255.0, x_test / 255.0  # Chuẩn hóa về [0,1]
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape cho CNN
x_test = x_test.reshape(-1, 28, 28, 1)

# 3. Xây dựng mô hình CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# 4. Compile mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Huấn luyện mô hình
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 6. Đánh giá trên tập kiểm tra
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Độ chính xác trên tập kiểm tra: {test_acc:.4f}')

# 7. Dự đoán một số hình ảnh từ tập kiểm tra
predictions = model.predict(x_test[:5])

# Lưu mô hình
model.save("my_model2.h5")

# Hiển thị kết quả
for i in range(15):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Dự đoán: {np.argmax(predictions[i])}, Thực tế: {y_test[i]}")
    plt.axis('off')
    plt.show()

