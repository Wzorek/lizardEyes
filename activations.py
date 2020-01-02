import keras.models as M
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from PIL import Image

# Build your ResNet50 model
base_model = ResNet50(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
base_model.summary()

# Set output of first CONV layer os output of the network
layer_outputs = [layer.output for layer in base_model.layers if layer.name.startswith('conv1')]
activation_model = M.Model(inputs=base_model.input, outputs=layer_outputs)

# Predict object class with trained ResNet50 model
photo = Image.open("100lat.PNG")
image2class = image.img_to_array(photo)
image2class = image2class.reshape(-1, 256, 256, 3)
activations = activation_model.predict(image2class)

# Print all 64 activations of first CONV layer
activation1 = activations[1]
for i in range(64):
    # get the filter
    f = activation1[0, :, :, i]
    f_min, f_max = f.min(), f.max()
    f = (f - f_min) / (f_max - f_min)
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(f, cmap='gray')
plt.show()
