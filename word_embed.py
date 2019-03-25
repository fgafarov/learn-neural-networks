from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
lbls = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vs = 50
enc_docs = [one_hot(d, vs) for d in docs]
print(enc_docs)
# pad documents to a max length of 4 words
max_length = 4
p_docs = pad_sequences(enc_docs, maxlen=max_length, padding='post')
print(p_docs)
# define the model
modelEmb = Sequential()
modelEmb.add(Embedding(vs, 8, input_length=max_length))
modelEmb.add(Flatten())
modelEmb.add(Dense(1, activation='sigmoid'))
# compile the model
modelEmb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(modelEmb.summary())
# fit the model
modelEmb.fit(p_docs, lbls, epochs=150, verbose=0)
# evaluate the model
loss, accuracy = modelEmb.evaluate(p_docs, lbls, verbose=2)
print('Accuracy: %f' % (accuracy*100))