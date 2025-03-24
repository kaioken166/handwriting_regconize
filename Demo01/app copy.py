import os

augmented_folder = "dataset/augmented_images"
augmented_files = [f for f in os.listdir(augmented_folder) if f.endswith(".png")]

print(f"Số lượng ảnh trong {augmented_folder}: {len(augmented_files)}")
