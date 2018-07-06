
import tensorflow as tf
from tfHelper import tfHelper
import common

k = tf.keras
c = common

def model():
	# model = k.models.Sequential()
	# model.add(k.layers.Conv2D(1, (2, 2), activation='tanh',
	#                  input_shape = (c.imgWidth, c.imgHeight, c.imgChanel)))
	model = k.models.Sequential()
	model.add(k.layers.Conv2D(16, (5, 5), activation='relu',
	                 input_shape = (c.imgWidth, c.imgHeight, c.imgChanel)))
	
	model.add(k.layers.BatchNormalization())
	model.add(k.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
	# model.add(k.layers.BatchNormalization())
	model.add(k.layers.MaxPool2D(strides=(2,2)))
	model.add(k.layers.Dropout(0.25))

	model.add(k.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(k.layers.BatchNormalization())
	model.add(k.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
	# model.add(k.layers.BatchNormalization())
	model.add(k.layers.MaxPool2D(strides=(2,2)))
	model.add(k.layers.Dropout(0.25))

	model.add(k.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(k.layers.BatchNormalization())
	model.add(k.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
	# model.add(k.layers.BatchNormalization())
	model.add(k.layers.MaxPool2D(strides=(2,2)))
	model.add(k.layers.Dropout(0.25))

	model.add(k.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(k.layers.BatchNormalization())
	model.add(k.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
	# model.add(k.layers.BatchNormalization())
	# model.add(k.layers.MaxPool2D(strides=(2,2)))
	# model.add(k.layers.Dropout(0.25))

	model.add(k.layers.Flatten())
	# model.add(k.layers.Dense(1024, activation='relu'))
	# model.add(k.layers.Dropout(0.2))
	# model.add(k.layers.Dense(1024, activation='relu'))
	# model.add(k.layers.Dropout(0.2))
	# model.add(k.layers.Dense(100, activation='relu'))
	# model.add(k.layers.Dropout(0.2))
	model.add(k.layers.Dense(c.num_classes, activation='relu'))
	# # model.add(k.layers.BatchNormalization())
	# model.add(k.layers.Conv2D(1, (2, 2), strides=(1,1), activation='tanh'))
	# # model.add(k.layers.Conv2D(64, (3, 3), strides=(1,1), activation='tanh'))
	# # model.add(k.layers.Conv2D(128, (2, 2), strides=(1,1), activation='tanh'))
	# # model.add(k.layers.BatchNormalization())
	# # model.add(k.layers.MaxPool2D(strides=(30,30)))
	# # model.add(k.layers.Dropout(0.2))
	# # model.add(k.layers.Conv2D(256, (2, 2), activation='tanh', padding='same'))
	# model.add(k.layers.MaxPool2D(strides=(2,2)))

	# model.add(k.layers.Flatten())
	# # model.add(k.layers.Dense(1000, activation='tanh'))
	# # model.add(k.layers.Dense(5000, activation='tanh'))
	# # model.add(k.layers.Dropout(0.2))
	# model.add(k.layers.Dense(4251, activation='tanh'))

	return model

