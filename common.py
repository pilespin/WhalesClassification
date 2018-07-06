
import numpy as np
import os

allOutput = None
num_classes = 4251
batchSize = 5000
epochs = 1
imgWidth = 50
imgHeight = 50
imgChanel = 3	
convertColor = 'RGB'

def normalize(x_train):
    x_train = x_train.reshape(x_train.shape[0], imgWidth, imgHeight, imgChanel)
    x_train = x_train.astype('float32')
    x_train /= 255
    return x_train

def mkdir_p(path):
	if not os.path.exists(path):
		os.mkdir(path)

def count_elem_in_folder(path):
	return len(os.listdir(path))
