import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50

# Load the ResNet50 model
model = ResNet50(weights='imagenet', input_shape=(256, 256, 3), include_top=False)

# Get first Convolutional Layer
for layer in model.layers:
    if layer.name == 'conv1':
        filters1 = layer.get_weights()[0]

# Plot all 64 first layers
for i in range(64):
    # get the filter
    f = filters1[:, :, :, i]
    f_min, f_max = f.min(), f.max()
    f = (f - f_min) / (f_max - f_min)
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(f)
plt.show()
