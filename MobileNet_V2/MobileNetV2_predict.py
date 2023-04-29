from pathlib import Path
from keras.models import model_from_json
from keras.utils import load_img, img_to_array
import numpy as np

f = Path('model_structure.json')
model_structure = f.read_text()
model2 = model_from_json(model_structure)
model2.load_weights('model_weights.h5')

img = load_img('alpaca.png', target_size=(160, 160))
image_array = img_to_array(img)
images = np.expand_dims(image_array, axis=0)

results = model2.predict(images)
print(results[0][0])
