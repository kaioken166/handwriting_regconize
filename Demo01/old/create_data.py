import os
import pandas as pd

dataset_folder = "dataset/images"  # Thư mục chứa ảnh gốc và ảnh tăng cường
image_files = [f for f in os.listdir(dataset_folder) if f.endswith(".png")]

def extract_label(image_name):
    try:
        label = image_name.split("digit_")[1].split("_")[0]
        return label
    except IndexError:
        return None

data = [[img, extract_label(img)] for img in image_files if extract_label(img) is not None]

labels_df = pd.DataFrame(data, columns=["image_name", "label"])
labels_df.to_csv("labels_continue.csv", index=False)
print(f"Đã tạo labels.csv với {len(labels_df)} ảnh.")
