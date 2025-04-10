import numpy as np

from preprocess import load_dataset


def pad_labels(labels, max_label_length):
    """Padding nhãn để có độ dài đồng nhất"""
    padded_labels = np.zeros((len(labels), max_label_length), dtype=np.int32)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    return padded_labels


def prepare_data(data_dir, max_label_length=4):
    """Chuẩn bị dữ liệu cho CTC loss"""
    X, y = load_dataset(data_dir)

    # Padding nhãn
    y_padded = pad_labels(y, max_label_length)

    # Tính chiều dài nhãn
    label_lengths = np.array([len(label) for label in y])

    # Tính chiều dài input (chiều dài ảnh sau khi thực hiện các thao tác với CNN)
    input_lengths = np.full((X.shape[0],), X.shape[1] // 4)  # Chia cho 4 vì sau CNN ảnh sẽ có chiều cao giảm

    return X, y_padded, input_lengths, label_lengths


if __name__ == "__main__":
    data_dir = "../decimal_digits"  # Thư mục chứa dữ liệu
    X, y_padded, input_lengths, label_lengths = prepare_data(data_dir)
    print(f"X shape: {X.shape}")
    print(f"y_padded shape: {y_padded.shape}")
    print(f"input_lengths shape: {input_lengths.shape}")
    print(f"label_lengths shape: {label_lengths.shape}")
