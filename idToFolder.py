
import pandas as pd
import os
from shutil import copy


path_img = 'new/'
path_new_dataset = 'classed/'
path_dataset = 'datasets/train.csv'


def selectColumnById(df, id_str, id_nb, columnInRow):
	row = df.loc[df[id_str] == id_nb]
	if (len(row) <= 0):
		return 'None'
	return row[columnInRow].to_string(index=False)

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)


df = pd.read_csv(path_dataset)
mkdir(path_new_dataset)

output = []
for current in os.listdir(path_img):
	if current[0] != '.':
		id, _ = os.path.splitext(current)
		# print(current)
		species = selectColumnById(df, 'Image', current, 'Id')
		folder = path_new_dataset + species + '/'

		move_from = path_img + current
		move_to = folder + current

		print ("id: " + str(current) + ', ' + species)

		mkdir(folder)
		copy(move_from, move_to)
		