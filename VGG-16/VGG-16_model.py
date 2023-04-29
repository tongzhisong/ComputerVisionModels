from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib

x_train = joblib.load('x_train.dat')
y_train = joblib.load('y_train.dat')

# Sequential API does not allow input included in the model, so the pretrained model has to do Predict to calculate the
# intermediate values first. While Functional API could include the pretrained model and top model all together.
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

model_structure = model.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)
model.save_weights('model_weights.h5')
