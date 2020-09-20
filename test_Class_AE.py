#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:56:07 2020

@author: toddwimer
"""

from keras_mnist_AE_CLASS import MNIST_AE as MNIST_AE
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
# from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPool2D, UpSampling2D
# from random import randint
# import numpy as np
# print(tf.__version__)

mnist_AE = MNIST_AE()
mnist_AE.plot_data()
mnist_AE.build_FC_AE()
# FC_AE = mnist_AE.build_FC_AE()
# FC_AE.summary()
mnist_AE.compile_FC_AE()
mnist_AE.fit_FC_AE()
# FC_AE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
