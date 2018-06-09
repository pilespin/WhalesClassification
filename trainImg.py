
import tensorflow as tf
import numpy as np
# np.set_printoptions(linewidth=200)

from tfHelper import tfHelper
import data

k = tf.keras

tfHelper.log_level_decrease()

batch_size = 64
num_classes = 10
epochs = 100
imgWidth = 32

print ("Load data ...")
# (x_train, y_train), (x_test, y_test) = data.load_data_train()
(x_train, y_train) = tfHelper.get_dataset_with_folder('classed/', 'L')

# print (x_train)
# print (y_train)

# X_pred, X_id, label = data.load_data_predict()

split = int(len(x_train) * 0.00001)
x_test = x_train[:split]
x_train = x_train[split:]
y_test = y_train[:split]
y_train = y_train[split:]

input_size = len(x_train[0])
num_classes = 4250

print(str(num_classes) + ' classes')
print(str(input_size) + ' features')
print(str(len(x_train)) + ' lines')

print(x_train.shape, 'train samples')

x_train = data.normalize(x_train)
x_test = data.normalize(x_test)
print(x_train.shape, 'xtrain')
print(y_train.shape, 'ytrain')



model = tfHelper.load_model("model")

# model = k.models.Sequential()
# model.add(k.layers.Dense(300, input_dim=input_size, activation='tanh'))
# model.add(k.layers.Dense(200, activation='tanh'))
# model.add(k.layers.Dense(150, activation='tanh'))
# model.add(k.layers.Dense(num_classes, activation='softmax'))

# model = k.models.Sequential()
# model.add(k.layers.Conv2D(16, (5, 5), activation='relu',
# 				 input_shape = (imgWidth, imgWidth, 1)))
# model.add(k.layers.MaxPool2D(strides=(2,2)))

# model.add(k.layers.Flatten())
# model.add(k.layers.Dense(1000, activation='relu'))
# model.add(k.layers.Dense(num_classes, activation='softmax'))

# model = k.models.Sequential()
# model.add(k.layers.Conv2D(16, (3, 3), activation='relu',
#                  input_shape = (imgWidth, imgWidth, 1)))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Flatten())
# model.add(k.layers.Dense(1024, activation='relu'))
# model.add(k.layers.Dropout(0.2))
# model.add(k.layers.Dense(1024, activation='relu'))
# model.add(k.layers.Dropout(0.2))
# model.add(k.layers.Dense(100, activation='relu'))
# model.add(k.layers.Dropout(0.2))
# model.add(k.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy'
			, optimizer=k.optimizers.Adam(lr=0.001, decay=0.001)
			, metrics=['accuracy'])


learning_rate_reduction = k.callbacks.ReduceLROnPlateau(monitor='val_loss', 
														patience=1, 
														verbose=1, 
														factor=0.5, 
														min_lr=1e-09)

tensorBoard = k.callbacks.TensorBoard()

# model.compile(loss='categorical_crossentropy',
# 			  optimizer=opt,
# 			  metrics=['accuracy'])

datagen = k.preprocessing.image.ImageDataGenerator( rotation_range=180,
													width_shift_range=0.5,
													height_shift_range=0.5,
													shear_range=0.5,
													zoom_range=0.8,
													horizontal_flip=True,
													fill_mode='nearest')
datagen.fit(x_train)

# model.fit(x_train, y_train,
# for i in range(epochs):

len = x_train.shape[0]
step = 200
for i in range(epochs):
	for i in range(0, len, step):
		# model.train_on_batch(self, x_train, y_train)
		print ("Batch " + str(i) + "-" + str(i+step))
		# print("Batch " + str(i) + '/' + str(len))
		model.fit_generator(datagen.flow(x_train[i:i+step], y_train[i:i+step],
		# model.fit_generator(datagen.flow(x_train, y_train,
				batch_size=batch_size),
				epochs=1,
				# validation_data=(x_train, y_train),
				steps_per_epoch=500,
				validation_data=(x_test, y_test),
				shuffle=True,
				verbose=1,
				callbacks=[tensorBoard]
				)

		tfHelper.save_model(model, "model")
