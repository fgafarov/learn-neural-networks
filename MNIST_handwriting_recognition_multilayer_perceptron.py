import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

(X_tr, y_tr), (X_tst, y_tst) = mnist.load_data()
npix = X_tr.shape[1] * X_tr.shape[2]
X_tr = X_tr.reshape(X_tr.shape[0], npix).astype('float32')
X_tst = X_tst.reshape(X_tst.shape[0], npix).astype('float32')

# normalize inputs from 0-255 to 0-1
X_tr = X_tr / 255
X_tst = X_tst / 255

# one hot encode outputs
y_tr = np_utils.to_categorical(y_tr)
y_tst = np_utils.to_categorical(y_tst)
num_classes = y_tst.shape[1]

def create_model():
	# create model
	m = Sequential()
	m.add(Dense(npix,input_dim=npix, kernel_initializer='normal', activation='relu'))
	m.add(Dense(num_classes,kernel_initializer='normal', activation='softmax'))
	# Compile model
	m.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	return m
# build the model
model_perc = create_model()
# Fit the model
model_perc.fit(X_tr, y_tr, validation_data=(X_tst, y_tst), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model_perc.evaluate(X_tst, y_tst, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
