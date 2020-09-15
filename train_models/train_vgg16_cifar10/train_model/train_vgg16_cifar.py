import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10


def train_model(train_data, test_data):
	epochs = 60
	batch_size = 250


	model = Sequential()

	model.add(InputLayer(input_shape=(32,32,3)))
	# Block 1
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
	model.add(Dropout(0.25))

	# Block 2
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
	model.add(Dropout(0.25))

	# Block 3
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
	model.add(Dropout(0.25))

	# Block 4
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
	model.add(Dropout(0.25))

	# Block 5
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
	model.add(Dropout(0.25))

	# Classification Block
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
		          optimizer=opt,
		          metrics=['accuracy'])

	# Fit the model
	model.fit(train_data,
	                epochs=epochs,
	                validation_data=test_data,
	                workers=4)


	return model

