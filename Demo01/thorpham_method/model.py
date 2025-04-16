# model.py
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

# 1. Tải và tiền xử lý dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu: chuyển giá trị pixel về [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# 2. Trích xuất đặc trưng HOG
def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Đảm bảo ảnh ở định dạng uint8 cho HOG
        img_uint8 = (img * 255).astype(np.uint8)
        # Tính toán HOG
        feature = hog(img_uint8, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L2-Hys')
        hog_features.append(feature)
    return np.array(hog_features)


# Trích xuất đặc trưng cho tập huấn luyện và kiểm tra
print("Trích xuất đặc trưng HOG cho tập huấn luyện...")
X_train_hog = extract_hog_features(X_train)
print("Trích xuất đặc trưng HOG cho tập kiểm tra...")
X_test_hog = extract_hog_features(X_test)

# 3. Huấn luyện mô hình SVM
print("Huấn luyện mô hình SVM...")
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(X_train_hog, y_train)

# 4. Dự đoán và đánh giá
y_pred = clf.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")


# 5. Hàm nhận diện chữ số từ ảnh đầu vào
def predict_digit(image_path, model):
    # Đọc và tiền xử lý ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh!")

    # Resize về 28x28
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0

    # Trích xuất HOG
    img_uint8 = (img * 255).astype(np.uint8)
    hog_feature = hog(img_uint8, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm='L2-Hys')

    # Dự đoán
    digit = model.predict([hog_feature])[0]
    return digit


# 6. Thử nghiệm với một ảnh ví dụ
# Lưu ý: Bạn cần chuẩn bị một ảnh chữ số viết tay (ví dụ: digit.jpg)
try:
    image_path = "../img/new_image7.png"  # Thay bằng đường dẫn ảnh của bạn
    predicted_digit = predict_digit(image_path, clf)
    print(f"Chữ số dự đoán: {predicted_digit}")

    # Hiển thị ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(f"Dự đoán: {predicted_digit}")
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print("Vui lòng cung cấp một ảnh chữ số viết tay để thử nghiệm!")

# 5. Lưu mô hình
joblib.dump(clf, 'svm_digit_recognizer.pkl')
print("Mô hình đã được lưu vào 'svm_digit_recognizer.pkl'")
