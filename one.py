# from Test import Test
from Train import Train
from tfHelper import tfHelper

import model

import os


# te = Test()
tr = Train()

if os.path.exists("model.h5"):
	model = tfHelper.load_model("model")
else:
	model = model.model()

while True:
	# te.test(model)
	model = tr.train(model)
