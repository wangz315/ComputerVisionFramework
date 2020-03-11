import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler
from skimage import io,transform
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random

w = 100
h = 100

def load_img(path, label):
	imgs=[]
	labels=[]
	imglist=os.listdir(path)
	for im in imglist:
		if(im[0] != '.'):
			img=io.imread(path+im)
			img=transform.resize(img,(w,h))
			imgs.append(img)
			labels.append(label)
	return imgs,labels

def process_data():
	paths = []
	paths.append('class0/')
	paths.append('class1/')
	x = []
	y = []

	for i in range(len(paths)):
		imgs,labels = load_img(paths[i], i)
		x.extend(imgs)
		y.extend(labels)

	return np.asarray(x,np.float32),np.asarray(y,np.int32)

def scheduler(epoch):
	if epoch % 20 == 0 and epoch != 0:
		lr = K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr, lr * 0.1)
		print("lr changed to {}".format(lr * 0.1))
	return K.get_value(model.optimizer.lr)

x,y = process_data()
arr=np.arange(x.shape[0])
np.random.shuffle(arr)

x = x[arr]
y = y[arr]

ratio = np.int(0.8 * len(x))
x_train = x[:ratio]
y_train = y[:ratio]
x_test = x[ratio:]
y_test = y[ratio:]

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(w, h, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))

#model.load_weights("model/train_model.ckpt")

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


reduce_lr = LearningRateScheduler(scheduler)
checkpoint= keras.callbacks.ModelCheckpoint('model/cnn.model', monitor='val_acc', verbose=1, period = 10)
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=100, callbacks = [checkpoint, reduce_lr])
score = model.evaluate(x_test, y_test, batch_size=64)
#plot_model(model, to_file='modelcnn.png',show_shapes=True)
model.save("CNN.model")
