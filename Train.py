
import tensorflow as tf
from tfHelper import tfHelper
import numpy as np
import os

import model
import predict
import common


class Train:

	folderPath = './classed_train/'
	k = tf.keras
	c = common

	def __init__(self):
		'Train Initialized'
		tfHelper.log_level_decrease()
		# self.k.initializers.Ones()
		self.c.allOutput = tfHelper.get_all_allpout(self.folderPath)
		# self.k.initializers.RandomUniform(minval=0.7, maxval=1, seed=None)
		tfHelper.numpy_show_entire_array(self.c.imgWidth)

	def fit(self, model, x_train, y_train):

		if len(x_train) > 0:
			x_train = np.array(x_train)
			y_train = np.array(y_train)

			tensorBoard = self.k.callbacks.TensorBoard()

			learning_rate_reduction = self.k.callbacks.ReduceLROnPlateau(monitor='loss', 
																	patience=5, 
																	verbose=1, 
																	factor=0.5, 
																	min_lr=1e-09)

			datagen = self.k.preprocessing.image.ImageDataGenerator( 
																rotation_range=1,
																# width_shift_range=0.01,
																# height_shift_range=0.01,
																# shear_range=0.01,
																# zoom_range=0.01,
																# horizontal_flip=True,
																fill_mode='nearest')



			print("x_train", x_train.shape)
			print("y_train", y_train.shape)

			if y_train.shape[0] == 0:
				print("Bad dataset")
				exit(0)


			# datagen.fit(x_train)

			# for i in range(self.c.epochs):
				# print("Epoch " + str(i+1) + '/' + str(self.c.epochs))
			# model.fit_generator(datagen.flow(x_train, y_train, batch_size=10),
			model.fit(x_train, y_train,
					batch_size=32,
					# workers=8,
					# steps_per_epoch=50,
					epochs=30,
					# validation_data=(x_train, y_train),
					# validation_data=(x_test, y_test),
					shuffle=True,
					verbose=1,
					callbacks=[learning_rate_reduction, tensorBoard]
					# callbacks=[tensorBoard]
					)

			tfHelper.save_model(model, "model")
		return model

	def train(self, model):

		opt = self.k.optimizers.Adam(lr=0.00003, epsilon=0.1)
		# model.compile(loss='categorical_crossentropy',
		model.compile(loss='categorical_hinge',
				optimizer=opt,
				metrics=['accuracy'])


		print ("Load data ...")
		x_train = []
		y_train = []
		allSize = len(os.listdir(self.folderPath))
		for cur,subfolder in enumerate(os.listdir(self.folderPath)):
			if subfolder[0] != '.':
				# print ("Load folder: " + folder)
				(x, y) = tfHelper.get_dataset_with_one_folder(self.folderPath, subfolder, self.c.convertColor, self.c.allOutput)

				x = self.c.normalize(x)


				if len(y[0]) == self.c.num_classes:
					for i in x:
						x_train.append(i)
					for i in y:
						y_train.append(i)

			if (cur+1)%self.c.batchSize == 0:
				print ("Batch " + str(cur+1- self.c.batchSize) + '-' + str(cur+1)+ '/' + str(allSize))
				model = self.fit(model, x_train, y_train)
				x_train = []
				y_train = []

		print ("Batch " + str(cur+1- self.c.batchSize) + '-' + str(cur+1)+ '/' + str(allSize))
		model = self.fit(model, x_train, y_train)
		return model
