import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def prepare_data(dataset, look_back=1):
	dX, dY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dX.append(a)
		dY.append(dataset[i + look_back, 0])
	return numpy.array(dX), numpy.array(dY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
df = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
ds = df.values
ds = ds.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
ds = scaler.fit_transform(ds)
# split into train and test sets
tr_size = int(len(ds) * 0.67)
tst_size = len(ds) - tr_size
tr, tst = ds[0:tr_size,:], ds[tr_size:len(ds),:]
# reshape into X=t and Y=t+1
look_back = 1
trX, trY = prepare_data(tr, look_back)
tstX, tstY = prepare_data(tst, look_back)
# reshape input to be [samples, time steps, features]
trX = numpy.reshape(trX, (trX.shape[0], 1, trX.shape[1]))
tstX = numpy.reshape(tstX, (tstX.shape[0], 1, tstX.shape[1]))
# create and fit the LSTM network
modelPred = Sequential()
modelPred.add(LSTM(4, input_shape=(1, look_back)))
modelPred.add(Dense(1))
modelPred.compile(loss='mean_squared_error', optimizer='adam')
modelPred.fit(trX, trY, epochs=100, batch_size=1, verbose=2)
# make predictions
trPredict = modelPred.predict(trX)
tstPredict = modelPred.predict(tstX)
# invert predictions
trPredict = scaler.inverse_transform(trPredict)
trY = scaler.inverse_transform([trY])
tstPredict = scaler.inverse_transform(tstPredict)
tstY = scaler.inverse_transform([tstY])
# calculate root mean squared error
trScore = math.sqrt(mean_squared_error(trY[0], trPredict[:,0]))
print('Train Score: %.2f RMSE' % (trScore))
tstScore = math.sqrt(mean_squared_error(tstY[0], tstPredict[:,0]))
print('Test Score: %.2f RMSE' % (tstScore))
# shift train predictions for plotting
trPredictPlot = numpy.empty_like(ds)
trPredictPlot[:, :] = numpy.nan
trPredictPlot[look_back:len(trPredict)+look_back, :] = trPredict
# shift test predictions for plotting
tstPredictPlot = numpy.empty_like(ds)
tstPredictPlot[:, :] = numpy.nan
tstPredictPlot[len(trPredict)+(look_back*2)+1:len(ds)-1, :] = tstPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(ds))
plt.plot(trPredictPlot)
plt.plot(tstPredictPlot)
plt.show()

#import pandas
#import matplotlib.pyplot as plt
#dataset = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#plt.plot(dataset)
#plt.show()