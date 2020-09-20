#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:14:04 2020

@author: toddwimer
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPool2D, UpSampling2D
from random import randint
import numpy as np
# print(tf.__version__)

class MNIST_AE():
    
    def __init__(self):
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28
        self.num_channels = 1
        self.input_shape = (self.img_rows, self.img_cols, self.num_channels)
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.x_noisy = self.x_train + np.random.normal(loc=.0, scale=0.25, size=self.x_train.shape)
        self.x_all = np.append(self.x_noisy,self.x_train, axis=0)
        # print('x_all.shape: {}'.format(self.x_all.shape))
        self.x_clean = np.append(self.x_train,self.x_train, axis=0)
        self.x_test_noisy = self.x_test + np.random.normal(loc=.0, scale=0.25, size=self.x_test.shape)
        
    def plot_data(self):
        plt.figure(figsize=(10,10))
    
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i], cmap=plt.cm.binary)
            # plt.xlabel(class_names[y_train[i]])
        
        plt.show()
    
    # def build_FC_AE(input_shape):
    def build_FC_AE(self):
        inputs = Input(shape=self.input_shape, name='input')
        
        # As previously mentioned, the image flattening is done here:
        inputs_flat = Flatten()(inputs)
        
        # Encoding layers:
        enc_1 = Dense(128, activation='relu', name='enc_dense1')(inputs_flat)
        enc_2 = Dense(64, activation='relu', name='enc_dense2')(enc_1)
        code = Dense(32, activation='relu', name='enc_dense3')(enc_2)
        
        # Decoding layers:
        dec_1 = Dense(64, activation='relu', name='dec_dense1')(code)
        dec_2 = Dense(128, activation='relu', name='dec_dense2')(dec_1)
        decoded = Dense(np.prod(self.input_shape), activation='sigmoid', name='dec_dense3')(dec_2)
        # note: we use a sigmoid for the last activation, as we want the output values
        # to be between 0 and 1, like the input ones.
        
        # Finally, we reshape the decoded data so it has the same shape as the input samples:
        decoded_reshape = Reshape(self.input_shape)(decoded)
        
        # Auto-encoder model:
        # autoencoder = Model(inputs, decoded_reshape)
        self.FC_AE = Model(inputs, decoded_reshape)
        # autoencoder.summary()
        self.FC_AE.summary()
        # return autoencoder
    
    # def build_USCNN_AE(input_shape):
    def build_USCNN_AE(self):
        inputs = Input(shape=self.input_shape, name='input')
        
        # As previously mentioned, the image flattening is done here:
        # inputs_flat = Flatten()(inputs)
        
        # Encoding layers:
        enc_1 = Conv2D(filters=32, kernel_size=3, activation='relu', name='enc_conv1')(inputs)
        enc_pool_1 = MaxPool2D(pool_size=(2,2), name='enc_pool_1')(enc_1)
        enc_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name='enc_conv2')(enc_pool_1)
        enc_pool_2 = MaxPool2D(pool_size=(2,2), name='enc_pool_2')(enc_2)
        enc_3 = Conv2D(filters=64, kernel_size=3, activation='relu', name='enc_conv3')(enc_pool_2)
        code = Flatten()(enc_3)
        code = Dense(49, activation='softmax')(code)
        
        # Decoding layers:
        code_reshape = Reshape((7,7,1))(code)
        dec_1 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='same', name='dec_convTrans1')(code_reshape)
        # dec_1 = Conv2D(filters=64, kernel_size=3, activation='relu', name='dec_conv1')(code_reshape)
        upsamp_1 = UpSampling2D()(dec_1)
        dec_2 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='same', name='dec_convTrans2')(upsamp_1)
        # dec_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name='dec_conv2')(upsamp_1)
        upsamp_2 = UpSampling2D()(dec_2)
        dec_3 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', name='dec_convTrans3')(upsamp_2)
        # dec_3 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='dec_conv3')(upsamp_2)
        decoded = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='dec_conv4')(dec_3)
        # note: we use a sigmoid for the last activation, as we want the output values
        # to be between 0 and 1, like the input ones.
        
        # Finally, we reshape the decoded data so it has the same shape as the input samples:
        # decoded_reshape = Reshape(input_shape)(decoded)
        
        # Auto-encoder model:
        self.USCNN_AE = Model(inputs, decoded)
        # autoencoder = Model(inputs, decoded_reshape)
        self.USCNN_AE.summary()
        # return autoencoder
    
    # def build_TCNN_AE(input_shape):
    def build_TCNN_AE(self):
        inputs = Input(shape=self.input_shape, name='input')
        
        # As previously mentioned, the image flattening is done here:
        # inputs_flat = Flatten()(inputs)
        
        # Encoding layers:
        enc_1 = Conv2D(filters=32, kernel_size=3, activation='relu', name='enc_conv1')(inputs)
        enc_pool_1 = MaxPool2D(pool_size=(2,2), name='enc_pool_1')(enc_1)
        enc_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name='enc_conv2')(enc_pool_1)
        enc_pool_2 = MaxPool2D(pool_size=(2,2), name='enc_pool_2')(enc_2)
        enc_3 = Conv2D(filters=64, kernel_size=3, activation='relu', name='enc_conv3')(enc_pool_2)
        code = Flatten()(enc_3)
        code = Dense(49, activation='softmax')(code)
        
        # Decoding layers:
        code_reshape = Reshape((7,7,1))(code)
        dec_1 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same', name='dec_convTrans1')(code_reshape)
        dec_2 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same', name='dec_convTrans2')(dec_1)
        dec_3 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', name='dec_convTrans3')(dec_2)
        decoded = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='dec_conv1')(dec_3)
        # note: we use a sigmoid for the last activation, as we want the output values
        # to be between 0 and 1, like the input ones.
        
        # Finally, we reshape the decoded data so it has the same shape as the input samples:
        # decoded_reshape = Reshape(input_shape)(decoded)
        
        # Auto-encoder model:
        self.TCNN_AE = Model(inputs, decoded)
        # autoencoder = Model(inputs, decoded_reshape)
        self.TCNN_AE.summary()
        # return autoencoder
    
    # def build_DFCNN_AE(input_shape):
    def build_DFCNN_AE(self):
        inputs = Input(shape=self.input_shape, name='input')
        
        # As previously mentioned, the image flattening is done here:
        # inputs_flat = Flatten()(inputs)
        
        # Encoding layers:
        enc_1 = Conv2D(filters=32, kernel_size=3, activation='relu', name='enc_conv1')(inputs)
        enc_pool_1 = MaxPool2D(pool_size=(2,2), name='enc_pool_1')(enc_1)
        enc_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name='enc_conv2')(enc_pool_1)
        enc_pool_2 = MaxPool2D(pool_size=(2,2), name='enc_pool_2')(enc_2)
        enc_3 = Conv2D(filters=64, kernel_size=3, activation='relu', name='enc_conv3')(enc_pool_2)
        code = Flatten()(enc_3)
        code = Dense(49, activation='softmax')(code)
        
        # Decoding layers:
        code_reshape = Reshape((7,7,1))(code)
        dec_1 = Conv2D(filters=64, kernel_size=3, activation='relu', dilation_rate=(2,2), padding='same', name='dec_convDil1')(code_reshape)
        upsamp_1 = UpSampling2D()(dec_1)
        dec_2 = Conv2D(filters=64, kernel_size=3, activation='relu', dilation_rate=(2,2), padding='same', name='dec_convDil2')(upsamp_1)
        upsamp_2 = UpSampling2D()(dec_2)
        # dec_3 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='dec_convDil3')(upsamp_2)
        dec_3 = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', name='dec_convDil3')(upsamp_2)
        decoded = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decode_conv1')(dec_3)
        # note: we use a sigmoid for the last activation, as we want the output values
        # to be between 0 and 1, like the input ones.
        
        # Finally, we reshape the decoded data so it has the same shape as the input samples:
        # decoded_reshape = Reshape(input_shape)(decoded)
        
        # Auto-encoder model:
        self.build_DFCNN_AE = Model(inputs, decoded)
        # autoencoder = Model(inputs, decoded_reshape)
        self.build_DFCNN_AE.summary()
        # return autoencoder
    
    def compile_FC_AE(self):
        self.FC_AE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
    def fit_FC_AE(self):
        self.FC_AE.fit(self.x_noisy, self.x_train, epochs=5, verbose=1, validation_data=(self.x_test_noisy, self.x_test))

                      

# FC_autoencoder = build_FC_AE(input_shape)
# TCNN_autoencoder = build_TCNN_AE(input_shape)
# USCNN_autoencoder = build_USCNN_AE(input_shape)
# DFCNN_autoencoder = build_DFCNN_AE(input_shape)

# # just for debugging don't learn the models to debug code above
# compile_models = 1
# if compile_models == 1:
#     FC_autoencoder.compile(optimizer='adam',
#      loss='mse',
#      metrics=['accuracy'])
    
#     USCNN_autoencoder.compile(optimizer='adam',
#      loss='mse',
#      metrics=['accuracy'])
        
#     TCNN_autoencoder.compile(optimizer='adam',
#      loss='mse',
#      metrics=['accuracy'])
    
#     DFCNN_autoencoder.compile(optimizer='adam',
#      loss='mse',
#      metrics=['accuracy'])

# x_noisy = x_train + np.random.normal(loc=.0, scale=0.25, size=x_train.shape)
# x_all = np.append(x_noisy,x_train, axis=0)
# print('x_all.shape: {}'.format(x_all.shape))
# x_clean = np.append(x_train,x_train, axis=0)
# x_test_noisy = x_test + np.random.normal(loc=.0, scale=0.25, size=x_test.shape)
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_noisy[i], cmap=plt.cm.binary)
#     # plt.xlabel(class_names[y_train[i]])
    
# plt.show()

# # just for debugging don't learn the models to debug code above
# fit = 1

# if fit == 1:
#     # FC_autoencoder.fit(x_noisy, x_train, epochs=5, verbose=1, validation_data=(x_test_noisy, x_test))
#     # CNN_autoencoder.fit(x_noisy, x_train, epochs=5, verbose=1, validation_data=(x_test_noisy, x_test))
#     FC_autoencoder.fit(x_all, x_clean, epochs=5, verbose=1, validation_data=(x_test_noisy, x_test))
#     USCNN_autoencoder.fit(x_all, x_clean, epochs=5, verbose=1, validation_data=(x_test_noisy, x_test))
#     TCNN_autoencoder.fit(x_all, x_clean, epochs=5, verbose=1, validation_data=(x_test_noisy, x_test))
#     DFCNN_autoencoder.fit(x_all, x_clean, epochs=5, verbose=1, validation_data=(x_test_noisy, x_test))
#     test_val_FC_AE = FC_autoencoder.predict(x_test_noisy)
#     test_val_USCNN_AE = USCNN_autoencoder.predict(x_test_noisy)
#     test_val_TCNN_AE = TCNN_autoencoder.predict(x_test_noisy)
#     test_val_DFCNN_AE = DFCNN_autoencoder.predict(x_test_noisy)
#     tv_shape = test_val_FC_AE.shape
#     # print('test_val.shape: {}'.format(test_val_FC_AE.shape))
#     # print('x_test_noisy.shape: {}'.format(x_test_noisy.shape))
    
#     num_compare = 5
#     num_models = 5
#     for i in range(0, num_models * num_compare, num_models):
#         j = randint(0,tv_shape[0])
#         print('j: {}'.format(j))
#         plt.subplot(num_compare, num_models, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(np.squeeze(x_test_noisy[j]), cmap=plt.cm.binary)
#         plt.subplot(num_compare, num_models, i + 2)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(np.squeeze(test_val_FC_AE[j]), cmap=plt.cm.binary)
#         plt.subplot(num_compare, num_models, i + 3)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(np.squeeze(test_val_USCNN_AE[j]), cmap=plt.cm.binary)
#         plt.subplot(num_compare, num_models, i + 4)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(np.squeeze(test_val_TCNN_AE[j]), cmap=plt.cm.binary)
#         plt.subplot(num_compare, num_models, i + 5)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(np.squeeze(test_val_DFCNN_AE[j]), cmap=plt.cm.binary)
#     plt.show()



