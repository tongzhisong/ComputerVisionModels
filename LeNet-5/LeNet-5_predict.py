import tensorflow as tf
from keras.models import model_from_json
from pathlib import Path
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# model loading
f = Path('model_structure.json')
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights('model_weights.h5')

# data predict
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train /= 255
x_train = tf.expand_dims(x_train, axis=3)

results = model.predict(x_train)
predict_result = np.argmax(results, axis=1)
accuracy = np.sum(predict_result==y_train)/x_train.shape[0]*100
print("The accuracy of the model predicting the training set is: {:.2f}%".format(accuracy))

for i in range(x_train.shape[0]):
    sample_image = x_train[i]
    plt.imshow(sample_image)
    plt.show()

    sample = tf.expand_dims(sample_image, axis=0)
    result = int(np.argmax(model.predict(sample)))
    print(result)




