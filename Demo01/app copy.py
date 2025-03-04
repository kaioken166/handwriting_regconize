import cv2

img = cv2.imread("new_image.png", cv2.IMREAD_UNCHANGED)
print(type(img))  # Kiểm tra xem có phải None không
