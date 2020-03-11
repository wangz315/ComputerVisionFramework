import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.models import load_model
from skimage import io,transform
import os
import random
import tensorflow as tf
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

w = 100
h = 100

def csv_dict_read(path):
	d = dict()
	with open(path,'r',encoding='utf-8') as f:
		reader = csv.DictReader(f,dialect='excel')
		for row in reader:
			d['face'+row['MMID']+'_1.jpg'] = int(row['glasses'])
	return d

def load_img(path, label, d):
	imgs=[]
	labels=[]
	imglist=os.listdir(path)
	for im in imglist:
		if(im[0] != '.'):
			img=io.imread(path+im)
			img=transform.resize(img,(w,h))
			imgs.append(img)
			#labels.append(label)
			labels.append(d[im])
	return imgs,labels

def process_data():
	d = csv_dict_read('label_20191125.csv')
	paths = []
	paths.append('predict_data/')
	x = []
	y = []

	for i in range(len(paths)):
		imgs,labels = load_img(paths[i], i, d)
		x.extend(imgs)
		y.extend(labels)

	return np.asarray(x,np.float32),np.asarray(y,np.int32)


x,y = process_data()
arr=np.arange(x.shape[0])
np.random.shuffle(arr)

x = x[arr]
y = y[arr]

ratio = np.int(0.8 * len(x))
x_test = x[ratio:]
y_test = y[ratio:]

y_test = keras.utils.to_categorical(y_test, num_classes=2)

model = load_model('CNN.model')
predicted = model.predict(x_test)

output = np.argmax(predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

good = 0
total = 0
for i in range(len(output)):
	if output[i] == y_test[i]:
		good += 1
	total += 1

print('acc:',1-good/total)
