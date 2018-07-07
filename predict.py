
import numpy as np
from tfHelper import tfHelper
import data

tfHelper.log_level_decrease()
# tfHelper.numpy_show_entire_array(28)
# np.set_printoptions(linewidth=200)


print ("Load data ...")
# _, X_id, label = data.load_data_predict()
# x_train, y_train, le = tfHelper.get_dataset_with_folder('classed_train/')
# X_pred, X_id = tfHelper.get_dataset_with_one_folder('new_test/')

X_pred = data.normalize(X_pred)
# print (X_id)
# exit(0)

model = tfHelper.load_model("model")
# model = tfHelper.load_model("model")

######################### Predict #########################
predictions = model.predict(X_pred)

print(predictions)
# exit (0)

AllPrediction = []
for i in predictions:
	# indexMax = np.argmax(i)
	indexMax = np.argwhere(i>0.5)
	# print(i)
	AllPrediction.append(indexMax)

# print (AllPrediction)
AllPrediction_converted = le.inverse_transform(AllPrediction)
# print(a)


with open("output_img", "w+") as file:
	# Head
	file.write("Image,Id\n")

	for line, id in zip(AllPrediction_converted, X_id):
		file.write(str(id) + "," + str(line) + "\n")
