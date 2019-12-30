import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras.models as M
import datetime
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# Prepare data for network
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)
train_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(256, 256),
                    subset='training',
                    shuffle=True,
                    batch_size=32,
                    class_mode='categorical')
test_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(256, 256),
                    subset='validation',
                    shuffle=True,
                    batch_size=32,
                    class_mode='categorical')
print(train_data.class_indices)

# Build your own ResNet50
base_model = ResNet50(weights='imagenet', input_shape=(256, 256, 3), include_top=False)

base_model.summary()

my_model = M.Sequential()
my_model.add(base_model)
my_model.add(GlobalAveragePooling2D())
my_model.add(Dense(56, activation='softmax', kernel_initializer='glorot_normal'))
# my_model.summary()
my_model.load_weights('resnet_done/check/20191229-163004')

# Freeze all the layers before the `fine_tune_at` layer
fine_tune_at = 165
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Compile this model
opt = SGD(lr=1e-4, momentum=0.9)
my_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#
# Evaluate the model
loss0, accuracy0 = my_model.evaluate(test_data)
print("LOSS0")
print(loss0)
print("ACC0")
print(accuracy0)

# checkpoint = ModelCheckpoint('./checkpoints/finetune/',
# #                              monitor=["val_acc"],
# #                              verbose=1,
# #                              mode='max',
# #                              save_best_only=True)
# # log_dir="./logs/finetune/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
# # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# #
# #
# # callbacks_list = [checkpoint, tensorboard_callback]
# #
# # learning_process = my_model.fit_generator(train_data,
# #                                           epochs=100,
# #                                           validation_data=test_data,
# #                                           shuffle=True,
# #                                           callbacks=callbacks_list)
# # plt.figure(1, figsize=(15, 8))
# # plt.subplot(2, 2, 1)
# # plt.plot(learning_process.history['accuracy'])
# # plt.plot(learning_process.history['val_accuracy'])
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'valid'])
# # plt.subplot(2, 2, 2)
# # plt.plot(learning_process.history['loss'])
# # plt.plot(learning_process.history['val_loss'])
# # plt.title('model loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'valid'])
# # plt.savefig('acc_vs_epochs.png')
# # plt.show()