from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def load_imagenet():

	datapath = "/mnt/dataset/"

	train_data = tf.keras.preprocessing.image_dataset_from_directory(
		datapath + "train/",
		labels="inferred",
		label_mode="categorical",
		image_size=(224, 224),
		batch_size=64,
	)

	test_data = tf.keras.preprocessing.image_dataset_from_directory(
		datapath + "val/",
		labels="inferred",
		label_mode="categorical",
		image_size=(224, 224),
		batch_size=64,
	)

	return train_data, test_data
	
	
def load_cifar10():

	datapath = "/mnt/dataset/"
	
	train_data = tf.keras.preprocessing.image_dataset_from_directory(
					datapath + "train/",
					labels="inferred",
					label_mode="categorical",
					image_size=(32, 32),
					batch_size=64,
				)
					
	test_data = tf.keras.preprocessing.image_dataset_from_directory(
					datapath + "test/",
					labels="inferred",
					label_mode="categorical",
					image_size=(32, 32),
					batch_size=64,
				)

	# preprocess data
	train_data = train_data.map(lambda x, y: ((x / 127.5 - 1), y))
	test_data = test_data.map(lambda x, y: ((x / 127.5 - 1), y))
	
	return train_data, test_data


def get_dataset(name):

	if name == 'imagenet':
		#load data
		x_train, y_train, x_test, y_test = load_imagenet()
	elif name == 'cifar10':
		#load data
		x_train, y_train, x_test, y_test = load_cifar10()

	return x_train, y_train, x_test, y_test
