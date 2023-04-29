from keras.models import model_from_json
from keras.utils import image_dataset_from_directory
from pathlib import Path
import keras
import matplotlib.pyplot as plt

f = Path("model_structure.json")
model_structure = f.read_text()
model2 = model_from_json(model_structure)
model2.load_weights("model_weights.h5")

base_model = model2.layers[4]
base_model.trainable = True

fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

base_learning_rate = 0.001
loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.1*base_learning_rate)
metrics = ["accuracy"]

model2.compile(loss=loss_function,
               optimizer=optimizer,
               metrics=metrics)

IMG_SIZE = (160, 160)
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

# Train more 10 epochs with smaller learning rate and unfreeze the final layers
initial_epochs = 10
fine_tube_epochs = 10
total_epochs = initial_epochs + fine_tube_epochs

# Since in some of the optimizers, some of their internal values (e.g. learning rate) are set using the current epoch
# value, the initial_epoch argument let you specify the initial value of epoch to start from when training.
# This is mostly useful when you have trained your model for 10 (say) epochs, and then saved it and now want to load it
# and resume the training for another 10 epochs without disrupting the state of epoch-dependent objects (e.g. optimizer)
# You would set initial_epoch=10 and epochs=20, then everything resume as if you were initially trained the model for
# 20 epochs in one single training session.
history_fine = model2.fit(train_dataset,
                          epochs=total_epochs,
                          initial_epoch=initial_epochs,
                          validation_data=validation_dataset)

acc = [0.] + history_fine.history['accuracy']
val_acc = [0.] + history_fine.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.show()
