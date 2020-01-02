import time
import paramiko
import keras.models as M
import numpy as np
from scp import SCPClient
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import SGD
from PIL import Image

IP = '192.168.1.100'

# Prepare data for network to be able to get class indexes
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)
train_data = data_generator.flow_from_directory(
                    'input/',
                    subset='training')
print(train_data.class_indices)
key_list = list(train_data.class_indices.keys())
val_list = list(train_data.class_indices.values())

# Build your ResNet50 model
base_model = ResNet50(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
my_model = M.Sequential()
my_model.add(base_model)
my_model.add(GlobalAveragePooling2D())
my_model.add(Dropout(0.25))
my_model.add(Dense(56, activation='softmax', kernel_initializer='glorot_normal'))

# Freeze all the layers except of last Residual block
frizzed_layers = 165
for layer in base_model.layers[:frizzed_layers]:
  layer.trainable = False

# Load weights from learning process
my_model.load_weights('checkpoints\\DROP_last')

# Compile this model
opt = SGD(lr=1e-4, momentum=0.9)
my_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Open SSH connection with RaspberryPI and take the photo
client = paramiko.Transport((IP, 22))
client.connect(username='pi', password='raspberry')
session = client.open_channel(kind='session')
session.exec_command('cd CameraTest; raspistill -o cam.jpg -w 256 -h 256')
session.close()
client.close()
time.sleep(10)

# Copy photo from RaspberryPI to your working directory
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect(IP, username='pi', password='raspberry', look_for_keys=False)
with SCPClient(ssh.get_transport()) as scp:
    scp.get('/home/pi/CameraTest/cam.jpg', 'cam.jpg')

# Predict object class with trained ResNet50 model
photo = Image.open("cam.jpg")
image2class = image.img_to_array(photo)
image2class = image2class.reshape(-1, 256, 256, 3)
classify = my_model.predict(image2class)

# Open the photo and print 3 most probable labels
for x in range(3):
    class_number = np.argmax(classify[0])
    label = key_list[class_number]
    value = classify[0][class_number]
    classify[0][class_number] = 0
    print(label)
    print(value)
photo.show()