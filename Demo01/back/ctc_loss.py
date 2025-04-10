# ctc_loss.py

import tensorflow as tf

def ctc_loss(y_true, y_pred):
    """
    CTC Loss function để huấn luyện mô hình OCR
    """
    # Sử dụng CTC loss từ TensorFlow
    return tf.reduce_mean(tf.nn.ctc_loss(
        labels=y_true, logits=y_pred, label_length=label_lengths, logit_length=input_lengths))
