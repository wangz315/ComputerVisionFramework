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
import time
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils import relu_loss, cumulative_score

def load_data():
	data = np.load('data/train.npz')
	x = data['x']
	y = data['y']
	return x,y

def split_val(rgb, xyz, y):
	ratio = 0.8
	split = int(len(y)*ratio)

	rgb_train = rgb[:split]
	xyz_train = xyz[:split]
	y_train = y[:split]

	rgb_test = rgb[split:]
	xyz_test = xyz[split:]
	y_test = y[split:]

	return rgb_train, xyz_train, y_train, rgb_test, xyz_test, y_test

def split_channel(x):
	rgb = x[:,:,:,:3]
	xyz = x[:,:,:,3:]

	return rgb,xyz

def mask(imgs):
	for i in range(len(imgs)):
		row_length = random.randint(int(imgs[i].shape[0] / 4), int(imgs[i].shape[0] / 2))
		col_length = random.randint(int(imgs[i].shape[1] / 4), int(imgs[i].shape[1] / 2))

		row_start = random.randint(0, imgs[i].shape[0] - row_length)
		row_end = row_start + row_length

		col_start = random.randint(0, imgs[i].shape[1] - col_length)
		col_end = col_start + col_length

		imgs[i, row_start:row_end, col_start:col_end, :] = 0.0
	return imgs

def data_generator(rgb, xyz, y, batch_size, seed = 12):
	rgb_gen = ImageDataGenerator(samplewise_center = False, samplewise_std_normalization = False)#, brightness_range = None)
	xyz_gen = ImageDataGenerator()

	rgb_gen.fit(rgb)
	xyz_gen.fit(xyz)

	rgb_flow = rgb_gen.flow(rgb, y, batch_size=batch_size, seed=seed)
	xyz_flow = xyz_gen.flow(xyz, y, batch_size=batch_size, seed=seed)

	while 1:
		rgbin,out = next(rgb_flow)
		xyzin,_ = next(xyz_flow)
		# rgbin = mask(rgbin)
		yield {'RGBin' : rgbin, 'XYZin' : xyzin}, out

def step_decay(epoch):
	initial_lrate = 0.00001
	drop = 0.1
	epochs_drop = 200.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def load_model(rgb, xyz):
	cin = Input(shape = rgb[0].shape, name='RGBin')
	din = Input(shape = xyz[0].shape, name='XYZin')

	#color layers
	cout = Conv2D(32, (3, 3), activation='relu')(cin)
	cout = Conv2D(32, (3, 3), activation='relu')(cout)
	cout = MaxPooling2D(pool_size=(2, 2))(cout)
	cout = Flatten()(cout)
	cout = Dense(256, activation='relu')(cout)

	#depth layers
	dout = Conv2D(32, (3, 3), activation='relu')(din)
	dout = Conv2D(32, (3, 3), activation='relu')(dout)
	dout = MaxPooling2D(pool_size=(2, 2))(dout)
	dout = Flatten()(dout)
	dout = Dense(256, activation='relu')(dout)

	#merge layers
	out = concatenate([cout, dout])
	out = Dense(256, activation='relu')(out)
	out = Dense(256, activation='relu')(out)
	out = Dense(1)(out)

	model = Model(inputs=[cin,din], outputs=out)
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', cumulative_score])
	return model

def train_model():
	x,y = load_data()
	rgb,xyz = split_channel(x)
	rgb_train, xyz_train, y_train, rgb_test, xyz_test, y_test = split_val(rgb, xyz, y)
	train_gen = data_generator(rgb_train, xyz_train, y_train, 64)
	val_gen = data_generator(rgb_test, xyz_test, y_test, 64)

	model = load_model(rgb, xyz)
	model.summary()

	reduce_lr = LearningRateScheduler(step_decay)
	model.fit_generator(train_gen, validation_data=val_gen, epochs=400, steps_per_epoch=int(len(y_train)/64), validation_steps=int(len(y_test)/64),callbacks=[reduce_lr], verbose=2)
	model.save('model/cnn.model')

def main():
	train_model()

if __name__ == '__main__':
	main()


