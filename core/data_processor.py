import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
from math import sqrt
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename, header=0, index_col=0)
        #dataframe = dataframe[:28813]
        #print(dataframe.head(5))
        """
        i_split = 28800
        j_split = 20160
        s_split = 28700
        """
        i_split = 18000
        j_split = 15000
        s_split = 17973
        self.dataset = dataframe.values
        res = [1, 2, 3]
        self.dataset = self.dataset[:18054]
        #self.dataset[:, 1] = - np.log10((self.dataset[:, 1]))
        self.data_train_before = self.dataset[:i_split]
        self.data_test_before = self.dataset[s_split:]
        #self.dataset[:, -1] = - np.log((self.dataset[:, -1]))
        """
        #plot
        plt.figure(1)
        ax1 = plt.subplot(311)
        plt.plot(self.dataset[:, 0])
        ax2 = plt.subplot(312)
        plt.plot(self.dataset[:, 1])
        ax3 = plt.subplot(313)
        plt.plot(self.dataset[:, 2])
        ax1.set_title('Kp')
        ax2.set_title('Ap')
        ax3.set_title('F0.7')
        plt.legend()
        plt.show()
        """
        #check for NAN
        for i in range(self.dataset.shape[0]):
            for j in range(self.dataset.shape[1]):
                if self.dataset[i, j] == 'NaN' and i != 0 and i != (len(self.dataset) + 1):
                    self.dataset[i, j] = 0
        self.data_train = self.dataset[:i_split]
        self.data_test  = self.dataset[s_split:]

        #train normalise
        self.minmaxscaler_train = sklearn.preprocessing.StandardScaler().fit(self.data_train)
        self.data_train = self.minmaxscaler_train.transform(self.data_train)

        self.data_val = self.data_train[j_split:]
        self.data_train = self.data_train[:j_split]

        #test_1 normalise
        self.minmaxscaler = sklearn.preprocessing.StandardScaler().fit(self.data_test)
        self.data_test = self.minmaxscaler.transform(self.data_test)
        #self.data_test = self.normalise_std(self.data_test)
        
        #len
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_val = len(self.data_val)
        #self.data_eve = self.dataset[i_split:, -1]
        self.len_train_windows = None

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
	        cols.append(df.shift(i))
	        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
	        cols.append(df.shift(-i))
	        if i == 0:
		        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
	        else:
		        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	    # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def get_test_data(self, seq_len, normalise, out_dim):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_test - seq_len + 1 - out_dim + 1):
            x, y = self.next_window(self.data_test, i, seq_len, out_dim)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_train_data(self, seq_len, normalise, out_dim):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len + 1 - out_dim + 1):
            x, y = self.next_window(self.data_train, i, seq_len, out_dim)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_validation_data(self, seq_len, normalise, out_dim):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_val - seq_len + 1 - out_dim + 1):
            x, y = self.next_window(self.data_val, i, seq_len, out_dim)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise, out_dim):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self.next_window(self.data_train, i, seq_len, out_dim)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)
    
    def next_window(self, data, i, seq_len, out_dim):
        """
        Get the next window for data train val and test
        """
        window = data[i:i + seq_len]
        window_y = data[i: (i + seq_len + out_dim - 1), -1]
        window_y = np.reshape(window_y, (window_y.shape[0]))
        x = window[:-1]
        y = []
        for j in range(out_dim):
            y.append(window_y[seq_len - 1 + j])
        x = np.reshape(x, (x.shape[0], 3))
        return x, y
    
    def normalise_std(self, data):
        data = (data - np.mean(data)) / np.std(data)
        return data
    
    def de_normalise_std(self, data):
        #de_data =  data * (self.minmaxscaler.data_max_[2] - self.minmaxscaler.data_min_[2]) + self.minmaxscaler.data_min_[2]
        de_data =  data * self.minmaxscaler.scale_[-1] + self.minmaxscaler.mean_[-1]
        return de_data

    def get_split_data(self, x_test, pre_tem, i, seq_len):
        x_split = x_test[i]
        print(len(pre_tem))
        if i == 1 :
            #print('ori = ', x_split[26, -1])
            #print('new = ', pre_tem[0])
            x_split[26, -1] = pre_tem[0]
        elif i < seq_len - 1:
            #print('ori = ', x_split[(seq_len - 1 - i):27, -1])
            #print('new = ', pre_tem[0:i])
            x_split[(seq_len - 1 - i):27, -1] = pre_tem[0:i]
        else:
            #print('ori = ', x_split[:, -1])
            #print('new = ', pre_tem[(i - seq_len + 1):i])
            x_split[:, -1] = pre_tem[(i - seq_len + 1):i]
        return x_split

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
