import os
import math
import numpy as np
import pandas as pd
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras import losses
from sklearn import metrics
import keras

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath) 

	def MRE(self, y_true, y_pred):
		R_1 = []
		R_2 = []
		R_3 = []
		mean = np.mean(y_true)
		mean_pre = np.mean(y_pred)
		for i in range(len(y_true)):
			R_1.append((y_true[i] - mean) * (y_pred[i] - mean_pre))
			R_2.append((y_true[i] - mean) * (y_pred[i] - mean))
			R_3.append((y_pred[i] - mean_pre) * (y_pred[i] - mean_pre))
		r = np.sum(R_1) / (np.sqrt(np.sum(R_2)) * np.sqrt(np.sum(R_3)))
		return r

	def R(self, y_true, y_pred):
		a = keras.backend.sum((y_true - keras.backend.mean(y_true)) * (y_pred - keras.backend.mean(y_pred)))
		b = keras.backend.sum((y_true - keras.backend.mean(y_true)) * (y_true - keras.backend.mean(y_true)))
		c = keras.backend.sum((y_pred - keras.backend.mean(y_pred)) * (y_pred - keras.backend.mean(y_pred)))
		r = a / (keras.backend.sqrt(b * c))
		return r

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=[self.R])
		print('[Model] Model Compiled')
		timer.stop()
		
	def train(self, x, y, x_val, y_val, epochs, batch_size,  save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=10),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
			ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
			CSVLogger('F:/VScode/Pylearn/Test_1/figures/loss.csv', separator=',', append=False),
			TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		]
		self.history = self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			validation_data=(x_val, y_val),
			#validation_split=validation_split,
			callbacks=callbacks
		)
		self.model.save(save_fname)
		filepath_save = 'saved_models/F10.7_LSTM_100.h5'
		self.model.save(filepath_save)
		print('[Model] Training Completed. Model saved as %s' % filepath_save)
		self.model.summary()
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		#print('[Model] Predicting Point-by-Point...')	
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return np.array(predicted)

	def evaluate(self, y_nor, predictions_nor):
		y_nor = np.array(y_nor)
		predictions_nor = np.array(predictions_nor)
		y_nor = np.reshape(y_nor, (y_nor.shape[1], 1))
		predictions_nor = np.reshape(predictions_nor, (predictions_nor.shape[1], 1))
		#evaluate the accuracy of model
		loss = self.history.history['loss']
		val_loss = self.history.history['val_loss']
		#acc = self.history.history['acc']
		#val_acc = self.history.history['val_acc']
		#data_evaluate = np.vstack((loss, val_loss, acc, val_acc))
		data_evaluate = np.vstack((loss, val_loss))
		data_evaluate = pd.DataFrame(data_evaluate)
		data_evaluate.to_csv('figures/data_evaluate.csv')
		fig_loss = plt.figure(facecolor='white')
		plt.plot(self.history.history['loss'], label='train_loss')
		plt.plot(self.history.history['val_loss'], label='test_loss')
		plt.legend()
		plt.xlabel('Epochs', fontsize = 12)
		plt.ylabel('Loss', fontsize = 12)
		plt.title('Loss')
		plt.show()
		fig_loss.savefig('figures/fig_loss.png')
		#pyplot.ion()
		#pyplot.pause(5)
		#pyplot.close()
		"""
		#acc
		fig_acc = plt.figure(facecolor='white')
		plt.plot(self.history.history['acc'], label='train_acc')
		plt.plot(self.history.history['val_acc'], label='test_acc')
		plt.legend(['train', 'test'], loc='upper left')
		plt.xlabel('Epochs', fontsize = 12)
		plt.ylabel('Acc', fontsize = 12)
		plt.title('Acc')
		plt.show()
		fig_acc.savefig('figures/fig_mre.png')
		print('[File] Evaluate file saved.')
		"""
		error = []
		error_mean_nor = []
		error_mean_pre = []
		Squarderror = []
		Abserror = []
		Mrerror = []
		R_1 = []
		R_2 = []
		R_3 = []
		mean = np.mean(y_nor)
		mean_pre = np.mean(predictions_nor)
		for i in range(len(y_nor)):
			error.append(y_nor[i] - predictions_nor[i])
			error_mean_pre.append(predictions_nor[i] - mean)
			error_mean_nor.append(y_nor[i] - mean)
			Mrerror.append(abs(y_nor[i] - predictions_nor[i]) / y_nor[i])
			R_1.append((y_nor[i] - mean) * (predictions_nor[i] - mean_pre))
			R_2.append((y_nor[i] - mean) * (y_nor[i] - mean))
			R_3.append((predictions_nor[i] - mean_pre) * (predictions_nor[i] - mean_pre))
		for val in error:
			Squarderror.append(val * val)
			Abserror.append(abs(val))
		mse = np.sum(Squarderror) / len(y_nor)
		rmse = np.sqrt(mse)
		mae = np.sum(Abserror) / len(y_nor)
		mre = np.sum(Mrerror) / len(y_nor)
		r = np.sum(R_1) / (np.sqrt(np.sum(R_2)) * np.sqrt(np.sum(R_3)))
		#r_2 = np.sum()
		return mse, rmse, mae, mre, r

		
