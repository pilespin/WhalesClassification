# from Test import Test
from Train import Train
from tfHelper import tfHelper

import model as m
import common

import os

common.mkdir_p("models")

folderPath = './classed_train/'

for cur,subfolder in enumerate(os.listdir(folderPath)):
	if subfolder[0] != '.':

		# te = Test()
		tr = Train(subfolder)

		modelPath = "models/model" + '_' + subfolder

		if os.path.exists(modelPath + ".h5"):
			model = tfHelper.load_model(modelPath)
		else:
			model = m.model()

		# print(model.summary())

		# while True:
			# te.test(model)
		model = tr.train(model)
		tfHelper.save_model(model, modelPath)
		# exit(0)
