# Tổng hợp về xây dựng mô hình nhận diện số thập phân viết tay

## 1. Cách xây dựng model

### 1.1. Cách tiếp cận ban đầu

- **Mục tiêu**: Xây dựng một mô hình CNN để nhận diện số thập phân từ 0.0 đến 10.0 (bước 0.1) từ ảnh viết tay.
- **Kiến trúc ban đầu**:
  - 4 tầng `Conv2D` với số filters lần lượt là 32, 64, 64, 128.
  - 2 tầng `MaxPooling2D` để giảm kích thước đặc trưng.
  - Tầng `Flatten` để chuyển dữ liệu thành vector.
  - Tầng `Dense` với 64 units, tiếp theo là `Dropout` (0.5) để giảm overfitting.
  - Tầng đầu ra `Dense` với 101 units (tương ứng 101 lớp từ 0.0 đến 10.0) và hàm kích hoạt `softmax`.
- **Tiền xử lý ảnh**:
  - Chuẩn hóa màu sắc: Đảo màu nếu nền đen (dựa trên điều kiện `np.mean(img) < 127`).
  - Resize ảnh về kích thước 28x66.
  - Chuẩn hóa giá trị pixel về khoảng [0, 1].

### 1.2. Vấn đề gặp phải

- Độ chính xác huấn luyện thấp (từ 0.99% đến 14.72%) do lỗi trong ánh xạ nhãn và dữ liệu không đồng nhất.
- Sau khi sửa lỗi ánh xạ nhãn, độ chính xác vẫn không cải thiện đáng kể (9.31% đến 14.72%), và mô hình có xu hướng thiên lệch dự đoán về một số nhãn cụ thể (như 3.9 và 3.4).

### 1.3. Cách tiếp cận mới: Tách bài toán thành hai bước

- **Ý tưởng**: Chia bài toán thành hai mô hình riêng biệt:
  - **Mô hình 1**: Phân loại phần nguyên (0-10, 11 lớp).
  - **Mô hình 2**: Phân loại phần thập phân (0-9, 10 lớp).
- **Lợi ích**:
  - Giảm số lượng lớp cần phân loại từ 101 xuống còn 11 và 10, giúp mô hình dễ học hơn.
  - Dễ dàng cân bằng dữ liệu và cải thiện độ chính xác.

### 1.4. Kiến trúc mô hình mới

- **Mô hình 1 (phần nguyên)**:
  - 4 tầng `Conv2D` (32, 64, 64, 128 filters).
  - 2 tầng `MaxPooling2D`.
  - Tầng `Flatten`, tiếp theo là `Dense` (128 units), `Dropout` (0.3).
  - Tầng đầu ra `Dense` (11 units, `softmax`).
- **Mô hình 2 (phần thập phân)**:
  - Kiến trúc tương tự mô hình 1, nhưng tầng đầu ra là `Dense` (10 units, `softmax`).
- **Cải tiến sau đó**:
  - Tăng độ phân giải ảnh lên 56x132 để giữ thêm chi tiết.
  - Thêm `Batch Normalization` để ổn định quá trình huấn luyện.
  - Sử dụng `Learning Rate Scheduler` để điều chỉnh tốc độ học.
  - Áp dụng `class_weight` và oversampling để xử lý dữ liệu không cân bằng.

## 2. Cách tạo dữ liệu để huấn luyện

### 2.1. Dữ liệu ban đầu
- Dữ liệu được lưu trong thư mục `decimal_digits` với cấu trúc:
  - `0.0/`
    - `0,0_0.png`
    - `...`
  - `0.1/`
    - `0,1_0.png`
    - `...`
  - `...`
  - `10.0/`
    - `10,0_0.png`
    - `...`

- Ảnh có hai kiểu: chữ trắng trên nền đen và chữ đen trên nền trắng.

### 2.2. Tạo dữ liệu cho hai mô hình

- **Thư mục `integer_part`**:
  - Chứa ảnh cho phần nguyên (0-10).
  - Ví dụ: Ảnh `0,0_0.png` thuộc lớp 0, `1,0_0.png` thuộc lớp 1, ..., `10,0_0.png` thuộc lớp 10.
- **Thư mục `decimal_part`**:
  - Chứa ảnh cho phần thập phân (0-9).
  - Ví dụ: Ảnh `0,0_0.png` thuộc lớp 0, `0,1_0.png` thuộc lớp 1, ..., `0,9_0.png` thuộc lớp 9.

### 2.3. Oversampling cho `decimal_part`

- Dữ liệu ban đầu không cân bằng (ví dụ: lớp 0 có 3,571 ảnh, lớp 8 chỉ có 756 ảnh).
- Thực hiện oversampling để tăng số lượng ảnh mỗi lớp lên khoảng 3,500, đảm bảo cân bằng dữ liệu.

## 3. Lịch sử các điều chỉnh và kết quả

### 3.1. Huấn luyện ban đầu

- **Mô hình ban đầu**:
  - Độ chính xác: 0.99% đến 14.72%.
  - **Vấn đề**: Lỗi ánh xạ nhãn, dữ liệu không đồng nhất, mô hình thiên lệch dự đoán.

### 3.2. Sau khi sửa ánh xạ nhãn

- **Kết quả**: Độ chính xác dao động từ 9.31% đến 14.72%.
- **Vấn đề**: Mô hình vẫn thiên lệch về các nhãn như 3.9 và 3.4, không học tốt trên toàn bộ tập dữ liệu.

### 3.3. Huấn luyện với cách tiếp cận mới

- **Mô hình 1 (phần nguyên)**:
  - Độ chính xác: 89.09% (lần 1), 88.40% (lần 2).
  - Validation loss thấp nhất: 0.3838 (lần 1), 0.4392 (lần 2).
- **Mô hình 2 (phần thập phân)**:
  - Độ chính xác: 52.61% (lần 1), 50.47% (lần 2).
  - Validation loss thấp nhất: 1.4442 (lần 1), 1.4827 (lần 2).
  - **Vấn đề**: Mô hình gặp khó khăn trong việc nhận diện phần thập phân do đặc trưng phức tạp và dữ liệu không cân bằng.

### 3.4. Cải tiến cho mô hình 2

- **Oversampling**: Cân bằng dữ liệu từ 756 ảnh lên 3,571 ảnh mỗi lớp.
- **Tăng Data Augmentation**:
  - `rotation_range=40`.
  - `width_shift_range=0.4`, cùng các tham số khác.
- **Sử dụng `class_weight`**: Tăng trọng số cho các lớp ít dữ liệu.
- **Tăng độ phân giải ảnh**: Từ 28x66 lên 56x132.
- **Thêm Batch Normalization và Learning Rate Scheduler**.
- **Kết quả**: Độ chính xác vẫn chỉ đạt 50.47%, cần cải thiện thêm.

### 3.5. Đề xuất tiếp theo

- Kiểm tra `confusion matrix` để xác định các lớp bị nhầm lẫn nhiều nhất.
- Thử sử dụng mô hình pre-trained (như ResNet) hoặc chuyển sang huấn luyện trên GPU để tăng hiệu suất.
- Thực hiện của trang ThorPham.

## 4. Tài liệu tham khảo

- [ThorPham](https://thorpham.github.io/blog/Nh%E1%BA%ADn-d%E1%BA%A1ng-ch%E1%BB%AF-s%E1%BB%91-vi%E1%BA%BFt-tay)
- [CNN | Handwritten Digit Recognition](https://www.kaggle.com/code/itsmohammadshahid/7-cnn-handwritten-digit-recognition)
- [TopDev](https://topdev.vn/blog/thuat-toan-cnn-convolutional-neural-network/)
- [What are Convolutional Neural Networks? | IBM](https://www.ibm.com/think/topics/convolutional-neural-networks)
