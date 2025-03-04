import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image_path = "new_image6.jpg"
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cải thiện độ tương phản bằng cách dùng adaptive threshold
# img_processed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Đọc cả kênh alpha

if img.shape[-1] == 4:  # Kiểm tra nếu có 4 kênh (RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Chuyển về RGB

# Hiển thị ảnh
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
