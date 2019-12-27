import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras.models as M
import keras.backend as K
import keras.utils as U
import keras.applications as A
import keras.engine.topology as L
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.engine.topology import Layer
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
from keras.optimizers import SGD


# Prepare data for network
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)
train_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(244, 244),
                    subset='training',
                    shuffle=True,
                    batch_size=32,
                    class_mode='categorical')
test_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(244, 244),
                    subset='validation',
                    shuffle=True,
                    batch_size=32,
                    class_mode='categorical')
print(train_data.class_indices)

# Build your own ResNet50
base_model = ResNet50(weights='imagenet', input_shape=(244, 244, 3), include_top=False)
my_model = M.Sequential()
my_model.add(ResNet50(weights='imagenet', input_shape=(244, 244, 3), include_top=False))
my_model.add(GlobalAveragePooling2D())
my_model.add(Dense(56))
my_model.add(Activation('softmax'))
my_model.layers[0].trainable = False
my_model.summary()

# Compile this model
opt = SGD(lr=1e-4, momentum=0.9)
my_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#
# Train the model, iterating on the data in batches of 32 samples
fit_history = my_model.fit_generator(train_data,
                                     epochs=2,
                                     validation_data=test_data,
                                     validation_steps=19)
print(fit_history.history.keys())
plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.show()