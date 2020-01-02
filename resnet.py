import keras.models as M
import datetime
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# Set batch size for learning
BATCH_SIZE = 32

# Prepare data for network
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)
train_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(256, 256),
                    subset='training',
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    class_mode='categorical')
test_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(256, 256),
                    subset='validation',
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    class_mode='categorical')
print(train_data.class_indices)

# Build your own ResNet50 network
base_model = ResNet50(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
base_model.summary()
base_model.trainable = False
my_model = M.Sequential()
my_model.add(base_model)
my_model.add(GlobalAveragePooling2D())
my_model.add(Dense(56, activation='softmax', kernel_initializer='glorot_normal'))
my_model.summary()

# Compile your model
opt = SGD(lr=1e-4, momentum=0.9)
my_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Add checkpoints to be able to restore trained weights
checkpoint = ModelCheckpoint('checkpoints\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                             monitor='val_accuracy',
                             verbose=1,
                             mode='max',
                             save_best_only=True)

# Add TensorBoard callback to be able to babysit learning process
log='logs\\resnet\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log, histogram_freq=0)

# Train your model
learning_process = my_model.fit_generator(train_data,
                                          epochs=100,
                                          validation_data=test_data,
                                          callbacks=[checkpoint, tensorboard_callback],
                                          validation_steps=len(test_data)//BATCH_SIZE,
                                          shuffle=True)

# Plot accuracy and loss improvement process
plt.figure(1, figsize=(15, 8))
plt.subplot(2, 2, 1)
plt.plot(learning_process.history['accuracy'])
plt.plot(learning_process.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.subplot(2, 2, 2)
plt.plot(learning_process.history['loss'])
plt.plot(learning_process.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.savefig('acc_vs_epochs.png')
plt.show()
