
import numpy as np
import math
import cv2
import os

path_new_dataset = 'new_test/'
path_dataset = 'datasets/test/'

# path_new_dataset = 'new_test/'
# path_dataset = 'datasets/test/'

IMAGE_WIDTH = 50

# def rotate(img, angle):
# 	# tmp = swirl(img, rotation=90)

# 	num_rows, num_cols = img.shape[:2]
# 	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
# 	img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
# 	return img_rotation

def rotate(mat, angle): # I don't know what it do
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def resize(img, width_final):

	height, width = img.shape[:2]
	ratio_width = width_final / width

	ROTATED = None

	if height > width:
		ROTATED = True
		img = rotate(img, 90)
		height, width = img.shape[:2]
		ratio_width = width_final / width

	new = cv2.resize(img, None, fx=ratio_width, fy=ratio_width)

	height, width, depth = new.shape[:3]
	miss = width_final - height
	half_miss = int(miss/2)

	colorFill = new[0][0]

	ar = []
	for i in range(half_miss):
		miss -= 1
		for j in range(width_final):
			for k in colorFill:
				ar.append(k)

	for i in new.ravel():
		ar.append(i)

	for i in range(miss):
		for j in range(width_final):
			for k in colorFill:
				ar.append(k)

	new = np.array(ar).reshape(width_final,width_final,depth)

	if ROTATED is True:
		cv2.imwrite('tmp.jpg', new)
		new = cv2.imread('tmp.jpg')
		# new = cv2.imread('tmp.jpg', cv2.IMREAD_GRAYSCALE)
		new = rotate(new, -90)

	return new


# img = cv2.imread("datasets/images/1001.jpg", cv2.IMREAD_GRAYSCALE)
# if (img is None):
# 	print("Image not read")

# new = resize(img, 100)

# cv2.imwrite('new.jpg', new)
# exit(0)
# im = Image.open('newimg.jpg')
# im.rotate(45).show()

# IMAGE_WIDTH = 100

if not os.path.exists(path_new_dataset):
	os.mkdir(path_new_dataset)

output = []
for current in os.listdir(path_dataset):
	if current[0] != '.':
		# print ("Load: " + path_dataset + current)
		img = cv2.imread(path_dataset + current)
		# img = cv2.imread(path_dataset + current, cv2.IMREAD_GRAYSCALE)
		if (img is None):
			print("Image not read: " + current)	
		new = resize(img, IMAGE_WIDTH)
		cv2.imwrite(path_new_dataset + current, new)
		print ("Converted: " + path_dataset + current)


# cv2.imshow("Original image", img)
# cv2.imshow("new image", new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
