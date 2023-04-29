from keras.models import Sequential
from keras.layers import RandomFlip, RandomRotation, GlobalAveragePooling2D, Dropout, Dense
from keras.applications import MobileNetV2, mobilenet_v2
from keras import Model, Input
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

def data_augmenter():
    # data_augmentation is like model=Sequential(). model(training_set) to calculate layers
    data_augmentation = Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation

IMG_SIZE = (160, 160)

def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    input_shape = image_shape + (3,)
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model

BATCH_SIZE = 32
directory = "datasets/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=42)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = data_augmenter()
model2 = alpaca_model(IMG_SIZE, data_augmentation)

base_learning_rate = 0.001
# from_logits=True, logit with range(-inf, inf); from_logits=False, probability with range(0, 1)
model2.compile(optimizer=Adam(learning_rate=base_learning_rate),
               loss=BinaryCrossentropy(from_logits=False),
               metrics=['accuracy'])

initial_epochs = 10
history = model2.fit(train_dataset,
                     validation_data=validation_dataset,
                     epochs=initial_epochs)

model_structure = model2.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)
model2.save_weights('model_weights.h5')

acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.show()
