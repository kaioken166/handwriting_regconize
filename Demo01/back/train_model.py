import numpy as np
from ocr_model import build_model, ctc_loss
from preprocess import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
img_width = 128
img_height = 32
num_classes = 12  # 10 digits + comma
max_label_length = 5  # tối đa 5 ký tự cho nhãn
batch_size = 32
epochs = 10

# Load dataset
X, y = load_dataset("../decimal_digits")  # Thay đường dẫn với dữ liệu của bạn

# Tính toán input_lengths (độ dài ảnh đầu vào sau khi qua các lớp CNN)
input_lengths = np.ones(X.shape[0]) * (X.shape[1] // 4)  # Sau 2 lớp max pooling, chiều dài sẽ là X.shape[1] // 4

# Padding y để đảm bảo chiều dài cố định cho các nhãn
y_padded = pad_sequences(y, maxlen=max_label_length, padding='post', value=0)

# Tính toán label_lengths (độ dài nhãn)
label_lengths = np.array([len(label) for label in y])

# Build the model
model = build_model(img_width, img_height, num_classes, max_label_length)

# Compile the model với custom loss function
model.compile(optimizer='adam', loss=lambda y_true, y_pred: ctc_loss(y_true, y_pred, input_lengths, label_lengths))

# Train the model
model.fit(X, y_padded, batch_size=batch_size, epochs=epochs, validation_split=0.2,
          callbacks=[tf.keras.callbacks.ModelCheckpoint("ocr_model.h5", save_best_only=True)])
