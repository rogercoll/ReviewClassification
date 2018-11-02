import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class NN_Movie_Review(object):
	def __init__(self):
		self.accuracy = 0
		self.loss = 1

	def decode_review(self,text):
		return ' '.join([self.reverse_word_index.get(i, '?') for i in text])

	def encode_review(self,text):
		text = text.split(' ')
		mylist = []
		mylist.append(1)
		for word in text:
			aux = self.word_index.get(word,2)
			if aux > self.vocab_size:
				aux = 2
			mylist.append(aux)
		print(mylist)
		return mylist
	
	def test(self,text):
		messagearray = self.encode_review(text)
		messagearray = [messagearray]
		messagearray = keras.preprocessing.sequence.pad_sequences(messagearray,
														value=self.word_index["<PAD>"],
														padding='post',
														maxlen=256)
		prediction = self.model.predict(messagearray)
		print("Predictions goes from 0(bad review) to 1(good review)")
		if np.round(prediction[0]) == 1:
			print("Your review is a good comment because the final result was:")
		else:
			print("Your review is a bad comment because the final result was:")
		print(prediction[0])


	def train(self,n):
		#n means the maximum value that the words can have in the test and train data
		imdb = keras.datasets.imdb
		(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=n)
		self.word_index = imdb.get_word_index()
		self.word_index = {k:(v+3) for k,v in self.word_index.items()} 
		self.word_index["<PAD>"] = 0
		self.word_index["<START>"] = 1
		self.word_index["<UNK>"] = 2  # unknown
		self.word_index["<UNUSED>"] = 3
		self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])
		#Fem un padding de les dades pqe totes tinguin la mateixa llargada
		train_data = keras.preprocessing.sequence.pad_sequences(train_data,
																value=self.word_index["<PAD>"],
																padding='post',
																maxlen=256)
		test_data = keras.preprocessing.sequence.pad_sequences(test_data,
																value=self.word_index["<PAD>"],
																padding='post',
																maxlen=256)
		# input shape is the vocabulary count used for the movie reviews (10,000 words)
		self.vocab_size = 25000

		self.model = keras.Sequential()
		self.model.add(keras.layers.Embedding(self.vocab_size, 16))
		self.model.add(keras.layers.GlobalAveragePooling1D())
		self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
		self.model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

		self.model.compile(optimizer=tf.train.AdamOptimizer(),
					loss='binary_crossentropy',
					metrics=['accuracy'])

		#El primer va del 0 fins al 10000 i el segon del 10000 fins al final
		x_val = train_data[:10000]
		partial_x_train = train_data[10000:]

		y_val = train_labels[:10000]
		partial_y_train = train_labels[10000:]

		#Validation data, Data on which to evaluate the loss and any modal metrics at the end of each epoch
		#Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
		self.epochs = input("N of epochs(rec:40) --> ")
		self.history = self.model.fit(partial_x_train,
							partial_y_train,
							epochs=int(self.epochs),
							batch_size=512,
							validation_data=(x_val, y_val),
							verbose=1)
		print("")
		print("Evaluating some test data...")
		results = self.model.evaluate(test_data, test_labels)
		print("(Loss function, Accuracy)")
		print(results)
		self.loss = results[0]
		self.accuracy = results[1]