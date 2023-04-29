from keras.applications import vgg16
from keras.utils import load_img, img_to_array
from pathlib import Path
import numpy as np
import joblib

dog_path = Path('dogs')
not_dog_path = Path('not_dogs')

images = []
labels = []

for img in dog_path.glob('*.png'):
    img = load_img(img)
    img_arr = img_to_array(img)
    images.append(img_arr)
    labels.append(1)

for img in not_dog_path.glob('*.png'):
    img = load_img(img)
    img_arr = img_to_array(img)
    images.append(img_arr)
    labels.append(0)

x_train = np.array(images)
y_train = np.array(labels)
x_train = vgg16.preprocess_input(x_train)

pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features_x = pretrained_nn.predict(x_train)

joblib.dump(features_x, 'x_train.dat')
joblib.dump(y_train, 'y_train.dat')






