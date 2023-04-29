import tensorflow as tf
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.datasets import mnist
from pathlib import Path

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = tf.expand_dims(x_train, axis=3)
x_test = tf.expand_dims(x_test, axis=3)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# model
model = Sequential()

model.add(Conv2D(6, (5, 5), padding='same', input_shape=(28, 28, 1), activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(120, (5, 5), padding='valid', activation='tanh'))
model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# model.summary()

# train the model
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)

model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)
model.save_weights("model_weights.h5")
