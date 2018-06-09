
import pandas as pd
import numpy as np

# from sklearn import preprocessing
import sklearn.preprocessing as skp

imgWidth = 32

def normalize(array):
	array = array.reshape(array.shape[0], imgWidth, imgWidth, 1)
	array = array.astype('float32')
	array /= 255
	return array

def load_data_train():

	train_df, test_df = load_data()


	X_train_df = train_df.drop(['species', 'id'], axis=1)
	Y_train_df = train_df['species']

	# print(X_train_df[:2].to_string())
	# exit(0)

	############################## Scale ##############################
	# X_train = preprocessing.MinMaxScaler().fit_transform(X_train.values)

	X_train = X_train_df.values
	Y_train = Y_train_df.values

	############################## Split ##############################
	split = int(len(X_train) * 0.1)

	X_test = X_train[:split]
	X_train = X_train[split:]

	# print (Y_train)

	le = skp.LabelEncoder().fit(Y_train)
	Y_train = le.transform(Y_train)
	# print (Y_train)
	Y_train = pd.get_dummies(Y_train)
	# print (Y_train)

	Y_test = Y_train[:split]
	Y_train = Y_train[split:]


	return (X_train, Y_train), (X_test, Y_test)

def load_data_predict():

	train_df, test_df = load_data()

	idStart = 1
	X_id = range(idStart, len(test_df) + idStart)
	out = np.unique(train_df['species'])
	label = ['id']
	for i in out:
		label.append(i)
	X_test_df = test_df.drop(['species', 'id'], axis=1)

	############################## Scale ##############################
	# X_train = preprocessing.MinMaxScaler().fit_transform(test_df.values)

	############################## Split ##############################

	X_test = X_test_df.values

	return (X_test, test_df['id'], label)


def load_data():

	train_df = pd.read_csv('datasets/train.csv')
	test_df = pd.read_csv('datasets/test.csv')

	train_objs_num = len(train_df)
	datasetAll = pd.concat(objs=[train_df, test_df], axis=0)

	train = datasetAll[:train_objs_num]
	test = datasetAll[train_objs_num:]

	return(train, test)
