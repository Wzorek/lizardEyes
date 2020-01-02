import keras.models as M
import datetime
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
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

# Build your own ResNet50 network
base_model = ResNet50(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
my_model = M.Sequential()
my_model.add(base_model)
my_model.add(Dropout(0.25))
my_model.add(GlobalAveragePooling2D())
my_model.add(Dense(56, activation='softmax', kernel_initializer='glorot_normal'))

# Freeze all the layers except of last Residual block
fine_tune_at = 165
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Load weights from previous learning process
my_model.load_weights('checkpoints\\check_fine100_last')

# Compile your model
opt = SGD(lr=1e-4, momentum=0.9)
my_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
my_model.summary()

# Add checkpoints to be able to restore trained weights
checkpoint = ModelCheckpoint('./checkpoints/finetuneDROP/',
                             monitor=["val_accuracy"],
                             verbose=1,
                             save_best_only=True)
checkpoint_last = ModelCheckpoint('./checkpoints/finetuneDROP_last/',
                             monitor=["val_accuracy"],
                             verbose=1,
                             save_best_only=False)

# Add TensorBoard callback to be able to babysit learning process
log_dir="./logs/finetuneDROP/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks_list = [checkpoint, checkpoint_last, tensorboard_callback]

# Train your model
learning_process = my_model.fit_generator(train_data,
                                          epochs=100,
                                          validation_data=test_data,
                                          shuffle=True,
                                          callbacks=callbacks_list)

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
plt.savefig('acc_vs_epochs_fineDROP.png')
plt.show()