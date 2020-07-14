# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import pandas as pd


# data
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")

# Normalize inputs
x_train = x_train / 127.5 -1

#Train-Test Split
X_dev, X_val, Y_dev, Y_val = train_test_split(x_train, y_train, test_size=0.03, shuffle=True, random_state=2019)
T_dev = pd.get_dummies(Y_dev).to_numpy()
T_val = pd.get_dummies(Y_val).to_numpy()

#Reshape the input
X_dev = X_dev.reshape(X_dev.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

# LeNet 5 model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# build model
model.build()
print(model.summary())

# optimizer
adam = Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                patience=3,
                                verbose=1,
                                factor=0.2,
                                min_lr=1e-6)

# Data Augmentation
datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1)
datagen.fit(X_dev)

# train model
model.fit_generator(datagen.flow(X_dev, T_dev, batch_size=100), steps_per_epoch=int(len(X_dev)/100),
                    epochs=30, validation_data=(X_val, T_val), callbacks=[reduce_lr])


score = model.evaluate(X_val, T_val, batch_size=32)
print(score)

x_test = x_test / 127.5 -1

x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
test_score = model.evaluate(x_test, pd.get_dummies(y_test), batch_size=32)
print(test_score)

model.save('../../models/lenet_mnist/model_fixed.h5')

