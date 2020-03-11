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
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.metrics import mean_absolute_error
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils import relu_loss, cumulative_score

def load_data():
	data = np.load('data/test.npz')
	x = data['x']
	y = data['y']
	return x,y

def split_data(x):
	rgb = x[:,:,:,:3]
	xyz = x[:,:,:,3:]
	return rgb,xyz

def cumulative_score_rate(y_true, y_pred, threshold = 5):
	error = abs(y_pred - y_true) < threshold
	return sum(error)/len(y_true)

def data_generator(rgb, xyz, y, batch_size, seed = 12):
	rgb_gen = ImageDataGenerator(samplewise_center = False, samplewise_std_normalization = False)
	xyz_gen = ImageDataGenerator()

	rgb_gen.fit(rgb)
	xyz_gen.fit(xyz)

	rgb_flow = rgb_gen.flow(rgb, y, batch_size=batch_size, seed=seed)
	xyz_flow = xyz_gen.flow(xyz, y, batch_size=batch_size, seed=seed)

	while 1:
		rgbin,out = next(rgb_flow)
		xyzin,_ = next(xyz_flow)
		yield [rgbin, xyzin], out

def pred_by_gen(model, pred_gen, length):
	y_true = []
	y_pred = []
	
	for i in range(length):
		[rgbin, xyzin], out = next(pred_gen)
		y_pred.append(model.predict([rgbin, xyzin])[0])
		y_true.append(out)


	y_pred = np.asarray(y_pred)
	y_true = np.asarray(y_true)
	y_pred = y_pred.reshape(-1)
	y_true = y_true.reshape(-1)

	return y_true, y_pred

def main():
	x,y = load_data()
	rgb,xyz = split_data(x)
	pred_gen = data_generator(rgb, xyz, y, 1)

	model = load_model('model/cnn.model', custom_objects={'relu_loss': relu_loss, 'cumulative_score': cumulative_score})

	y_true, y_pred = pred_by_gen(model, pred_gen, len(y))

	mae = mean_absolute_error(y_true, y_pred)
	cs = cumulative_score_rate(y_true, y_pred)
	print('mae = ',mae)
	print('cs = ',cs)

if __name__ == '__main__':
	main()





















