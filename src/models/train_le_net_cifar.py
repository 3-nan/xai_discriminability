import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

epochs = 100
batch_size = 250

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Train-Test Split
# X_dev, X_val, Y_dev, Y_val = train_test_split(x_train, y_train, test_size=0.03, shuffle=True, random_state=2019)
# T_dev = pd.get_dummies(Y_dev).to_numpy()
# T_val = pd.get_dummies(Y_val).to_numpy()

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalize inputs
x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1

# Fit the model
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4)

score = model.evaluate(x_train, y_train, batch_size=32)
print(score)

#x_test = x_test / 127.5 - 1

# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
# test_score = model.evaluate(x_test, pd.get_dummies(y_test), batch_size=32)
test_score = model.evaluate(x_test, y_test, batch_size=32)
print(test_score)

model.save('../../models/lenet_cifar10/model_old.h5')

