
import tensorflow as tf
import sklearn.preprocessing as skp
import pandas as pd
import numpy as np
import Image
import os


class tfHelper:

	k = tf.keras

	def __init__():
		'tfHelper Initialized'
		# tfHelper.numpy_show_entire_array(self.c.imgWidth)


	###########################################################################
	#################################### IO ###################################
	###########################################################################

	@staticmethod
	def save_model(model, path):
		path_yaml = path + ".yaml"
		path_h5 = path + ".h5"

		model_yaml = model.to_yaml()
		with open(path_yaml, "w") as yaml_file:
			yaml_file.write(model_yaml)
		# print("Saved config to: " + path_yaml)

		# serialize weights to HDF5
		model.save_weights(path_h5)
		print("Saved weight to: " + path_h5 + ", Saved config to: " + path_yaml)

	@staticmethod
	def load_model(path):
		path_yaml = path + ".yaml"
		path_h5 = path + ".h5"

		# load config
		yaml_file = open(path_yaml, 'r')
		model_yaml = yaml_file.read()
		yaml_file.close()
		model = tf.keras.models.model_from_yaml(model_yaml)
		print("Loaded model from disk: " + path_yaml)
		
		# load weights into new model
		model.load_weights(path_h5)
		print("Loaded model from disk: " + path_h5)
		return (model)

	###########################################################################
	################################## IMAGE ##################################
	###########################################################################

	@staticmethod
	def image_to_array(path, convertColor):
		img = Image.open(path).convert(convertColor)
		arr = np.array(img)
		# make a 1-dimensional view of arr
		flat_arr = arr.ravel()
		return (flat_arr)

	# @staticmethod
	# def get_dataset_with_folder_old(path, convertColor):
	# 	X_train = []
	# 	Y_train = []

	# 	for foldername in os.listdir(path):
	# 		if foldername[0] != '.':
	# 			print("Load folder: " + foldername)
	# 			for filename in os.listdir(path + foldername):
	# 				if foldername[0] != '.':
	# 					path2 = path + foldername + "/" + filename
	# 					# img = tfHelper.image_to_array_greyscale(path2)
	# 					img = tfHelper.image_to_array(path2, convertColor)
	# 					X_train.append(img)
	# 					Y_train.append(int(foldername))
	# 	Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
	# 	return (np.array(X_train), np.array(Y_train))

	@staticmethod
	def get_dataset_with_folder(path, convertColor='RGB', allOutput=None):
		X_train = []
		Y_train = []

		for foldername in os.listdir(path):
			if foldername[0] != '.':
				# print("Load folder: " + foldername)
				for filename in os.listdir(path + foldername):
					if filename[0] != '.':
						path2 = path + foldername + "/" + filename
						# img = tfHelper.image_to_array_greyscale(path2)
						img = tfHelper.image_to_array(path2, convertColor)
						X_train.append(img)
						Y_train.append(foldername)
		label = np.unique(Y_train)
		Y_train = tfHelper.to_categorical_string(Y_train, allOutput)
		return (np.array(X_train), np.array(Y_train), label)

	@staticmethod
	def get_dataset_with_one_folder(folder, subfolder, convertColor='RGB', allOutput=None):
		X_train = []
		Y_train = []

		for filename in os.listdir(folder + subfolder):
			if filename[0] != '.':
				# print("Load folder: " + folder + subfolder)
				path2 = folder + subfolder + '/' + filename
				img = tfHelper.image_to_array(path2, convertColor)
				X_train.append(img)
				Y_train.append(subfolder)
		Y_train = tfHelper.to_categorical_string(Y_train, allOutput)

		return (np.array(X_train), np.array(Y_train))

	# @staticmethod
	# def get_dataset_with_one_folder(path, convertColor):
	# 	X_train = []

	# 	for filename in os.listdir(path):
	# 		if filename[0] != '.':
	# 			path2 = path + "/" + filename
	# 			img = tfHelper.image_to_array(path2, convertColor)
	# 			X_train.append(img)
	# 	return (np.array(X_train))


	# @staticmethod
	# def get_dataset_with_once_folder(name, path, convertColor):
	# 	X_train = None
	# 	Y_train = None

	# 	# for foldername in os.listdir(path):
	# 	# print "Load folder: " + path
	# 	for filename in os.listdir(path):
	# 		path2 = path + filename
	# 		img = tfHelper.image_to_array(path2, convertColor)
	# 		X_train = tfHelper.add_img_to_dataset(X_train, img)
	# 		Y_train = tfHelper.add_output_to_dataset(Y_train, 10, int(name))
	# 	return (X_train, Y_train)

	###########################################################################
	################################### ELSE ##################################
	###########################################################################

	@staticmethod
	def get_all_allpout(path):
		Y_train = []

		for foldername in os.listdir(path):
			if foldername[0] != '.':
				# print("Load folder: " + foldername)
				for filename in os.listdir(path + foldername):
					if filename[0] != '.':
						Y_train.append(foldername)

		return np.unique(Y_train)

	@staticmethod
	def to_categorical_string(array, allOutput):
		if allOutput is not None:
			cat = pd.Series(array).astype(pd.api.types.CategoricalDtype(categories=allOutput))
			array = pd.get_dummies(cat)
		else:
			array = pd.get_dummies(array)
		return array

	@staticmethod
	def log_level_decrease():
		os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Remove warning CPU SSE4 etc.

	@staticmethod
	def numpy_show_entire_array(px):
		lnbreak = (px + 1) * 4
		np.set_printoptions(threshold=np.nan)
		np.set_printoptions(linewidth=lnbreak)
		# np.set_printoptions(threshold='nan', linewidth=lnbreak)

	@staticmethod
	def count_elem_in_folder(path):
		nb = 0
		for foldername in os.listdir(path):
			nb = nb + 1
		return (nb)
