from keras.models import model_from_json
from pathlib import Path
from keras.utils import load_img, img_to_array
from keras.applications import vgg16
import numpy as np

f = Path('model_structure.json')
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

img = load_img('bay.jpg', target_size=(64, 64))
image_array = img_to_array(img)
images = np.expand_dims(image_array, axis=0)
images = vgg16.preprocess_input(images)

feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)
results = model.predict(features)
single_result = results[0][0]

print("The likelihood that the images contains a dog is: {}%".format(int(single_result * 100)))





