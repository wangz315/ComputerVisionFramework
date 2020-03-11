from keras.models import Sequential
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input, concatenate
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
import cv2

def relu_loss(y_true, y_pred, threshold = 5):
	if not K.is_tensor(y_pred):
			y_pred = K.constant(y_pred)
	y_true = K.cast(y_true, y_pred.dtype)
	cs = K.cast(K.greater(K.abs(y_pred - y_true), threshold), K.floatx())
	cse = cs * K.abs(y_pred - y_true)
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	return K.sum(cse)**2 + mse

def cumulative_score(y_true, y_pred, threshold = 5):
	if not K.is_tensor(y_pred):
		y_pred = K.constant(y_pred)
	y_true = K.cast(y_true, y_pred.dtype)
	return K.sum(K.cast(K.abs(y_pred - y_true) < threshold, dtype='float32'))/K.cast(K.shape(y_true)[0], dtype='float32')

