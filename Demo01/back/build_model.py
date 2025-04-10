import numpy as np

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from load_data import X, y

input_layer = Input(shape=(28, 66, 1))  # Kích thước ảnh: 28x(28+10+28)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

# Hai đầu ra
integer_output = Dense(11, activation='softmax', name='integer')(x)  # 0-10
decimal_output = Dense(10, activation='softmax', name='decimal')(x)  # 0-9

model = Model(inputs=input_layer, outputs=[integer_output, decimal_output])
model.compile(optimizer='adam',
              loss={'integer': 'sparse_categorical_crossentropy',
                    'decimal': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]  # Thêm chiều kênh (28, 66, 1)
X_test = X_test[..., np.newaxis]

model.fit(X_train, {'integer': y_train[:, 0], 'decimal': y_train[:, 1]},
          validation_data=(X_test, {'integer': y_test[:, 0], 'decimal': y_test[:, 1]}),
          epochs=10, batch_size=32)
