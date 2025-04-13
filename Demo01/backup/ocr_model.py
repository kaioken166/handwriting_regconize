import tensorflow as tf
from tensorflow.keras import layers, models


def ctc_loss(y_true, y_pred, input_lengths, label_lengths):
    """
    CTC Loss function để huấn luyện mô hình OCR
    """
    # CTC Loss yêu cầu y_true có shape là (batch_size, max_label_length)
    # CTC Loss yêu cầu y_pred có shape là (batch_size, time_steps, num_classes)

    print("Shape of y_true:", y_true.shape)  # Expected shape: (batch_size, max_label_length)
    print("Shape of y_pred:", y_pred.shape)  # Expected shape: (batch_size, time_steps, num_classes)

    return tf.reduce_mean(tf.nn.ctc_loss(
        labels=y_true, logits=y_pred, label_length=label_lengths, logit_length=input_lengths))


def build_model(img_width, img_height, num_classes, max_label_length):
    """
    Xây dựng mô hình OCR với CNN + LSTM + CTC Loss
    """
    # Input layer (32x128x1)
    input_img = layers.Input(shape=(img_height, img_width, 1), name='input_image')

    # CNN (Convolutional layers)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    # Reshape phù hợp với LSTM
    x = layers.Reshape(target_shape=(-1, 128))(x)  # Reshape thành (batch_size, time_steps, features)

    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    # Output layer (Số lượng class = 12: 10 digits + comma)
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Định nghĩa mô hình
    model = models.Model(inputs=input_img, outputs=output)

    model.summary()

    return model
