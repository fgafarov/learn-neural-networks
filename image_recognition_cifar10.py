from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os

batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_tr, y_tr), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_tr.shape)
print(x_tr.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_tr = keras.utils.to_categorical(y_tr, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

convNetModel = Sequential()
convNetModel.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_tr.shape[1:]))
convNetModel.add(Activation('relu'))
convNetModel.add(Conv2D(32, (3, 3)))
convNetModel.add(Activation('relu'))
convNetModel.add(MaxPooling2D(pool_size=(2, 2)))
convNetModel.add(Dropout(0.25))

convNetModel.add(Conv2D(64, (3, 3), padding='same'))
convNetModel.add(Activation('relu'))
convNetModel.add(Conv2D(64, (3, 3)))
convNetModel.add(Activation('relu'))
convNetModel.add(MaxPooling2D(pool_size=(2, 2)))
convNetModel.add(Dropout(0.25))

convNetModel.add(Flatten())
convNetModel.add(Dense(512))
convNetModel.add(Activation('relu'))
convNetModel.add(Dropout(0.5))
convNetModel.add(Dense(num_classes))
convNetModel.add(Activation('softmax'))


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# Let's train the model using RMSprop
convNetModel.compile(loss='categorical_crossentropy',optimizer=opt,  metrics=['accuracy'])
x_tr = x_tr.astype('float32')
x_test = x_test.astype('float32')
x_tr /= 255
x_test /= 255
convNetModel.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),   shuffle=True)
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_ph = os.path.join(save_dir, model_name)
convNetModel.save(model_ph)
print('Saved trained model at %s ' % model_ph)

# Score trained model.
scores = convNetModel.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])