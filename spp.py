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
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.engine.topology import Layer
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
from keras.optimizers import SGD


# SpatialPyramidPooling class
class SpatialPyramidPooling(Layer):
    '''Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    '''

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.common.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'th':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'th':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'tf':
            # outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            print(outputs.shape)
            # outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            # outputs = K.permute_dimensions(outputs,(3,1,0,2))
            # outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs


# Prepare data for network
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)
train_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(244, 244),
                    subset='training',
                    shuffle=True,
                    batch_size=100,
                    class_mode='categorical')
test_data = data_generator.flow_from_directory(
                    'input/',
                    target_size=(244, 244),
                    subset='validation',
                    shuffle=True,
                    batch_size=100,
                    class_mode='categorical')
print(train_data.class_indices)
# # Build your own ResNet50
# my_model = M.Sequential()
# my_model.add(ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False))
# my_model.add(Dense(49))
# my_model.layers[0].trainable = False
# my_model.summary()
#
# # Build your own ResNet50 model with SPP
# spp_model = M.Sequential()
# spp_model.add(ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False))
# spp_model.add(SpatialPyramidPooling([1, 2, 4]))
# spp_model.add(Dense(49))
# spp_model.summary()
#
# # Compile this model
# opt = SGD(lr=1e-4, momentum=0.9)
# spp_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#
# # Generate dummy data
# data = np.random.randint(1, size=(1, 244, 244, 3))
# labels = np.random.randint(1, size=(1, 1, 1, 49))
#
# # Train the model, iterating on the data in batches of 32 samples
# spp_model.fit(preprocess_input(data), labels, epochs=10, batch_size=32)
# #label = ResNet.predict(preprocess_input(img_array2))
# #name = decode_predictions(label)
# #print(name)