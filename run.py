import os
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
from sklearn.preprocessing import StandardScaler

def plot_results(predicted_data, true_data):
    predicted_data = np.array(predicted_data)
    true_data = np.array(true_data)
    predicted_data = np.reshape(predicted_data, (predicted_data.shape[1], 1))
    true_data = np.reshape(true_data, (true_data.shape[1], 1))
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.title('Prediction result')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    #plt.ion()
    #plt.pause(5)
    #plt.close()
    plt.show()
    fig.savefig('F:/VScode/Pylearn/Test_1/figures/fig_predict.png')


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('F:/VScode/Pylearn/Test_1/config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        out_dim=configs['data']['output_dim']
    )
    
    #get test
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        out_dim=configs['data']['output_dim']
    )

    #get validation
    x_val, y_val = data.get_validation_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        out_dim=configs['data']['output_dim']
    )
    
	# in-memory training
    model.train(
		x,
		y,
        x_val,
        y_val,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
        #validation_split = configs['training']['validation_split'],
		save_dir = configs['model']['save_dir']
	)
    
    predictions_all = []
    y_test_all = []
    mse = []
    rmse = []
    mre = []
    mae = []
    r = []
    predicitons_tem = []
    print('[Model] Predicting Point-by-Point...')
    # Predicting Point-by-Point
    for i in range(x_test.shape[0]):
        y_split = y_test[i]
        #y_split.reshape(y_split.shape[0], 1)
        y_split = np.reshape(y_split, (y_split.shape[0], 1))
        y_nor_split = data.de_normalise_std(y_split)
        y_nor_split = np.reshape(y_nor_split, (y_nor_split.shape[0]))
        #predict
        """
        if i == 0:
            x_split = x_test[i]
        else:
            x_split = data.get_split_data(x_test, predicitons_tem, i, seq_len=configs['data']['sequence_length'])
        """
        x_split = x_test[i]
        x_split = np.reshape(x_split, (1, x_split.shape[0], x_split.shape[1]))
        predictions = model.predict_point_by_point(x_split)
        predicitons_tem.append(predictions)
        predictions = data.de_normalise_std(predictions)
        predictions = np.reshape(predictions, (predictions.shape[0]))     
        #y_nor_split = np.e ** ( - y_nor_split)
        #predictions = np.e ** ( - predictions)
        y_test_all.append(y_nor_split)
        predictions_all.append(predictions)
    plot_results(predictions_all, y_test_all)
    mse_s, rmse_s, mae_s, mre_s, r_s = model.evaluate(y_test_all, predictions_all)
    rmse.append(rmse_s)
    mae.append(mae_s)
    mre.append(mre_s)
    r.append(r_s)
    mse.append(mse_s)
    print('精度指标为:')
    print('[Evaluste] MSE = ', mse_s)
    print('[Evaluate] RMSE = ', rmse_s)
    print('[Evaluate] MAE = ', mae_s)
    print('[Evaluate] MRE = ', mre_s)
    print('[Evaluate] R = ', r_s)
    #save
    Metrics = np.vstack((rmse, mre, r))
    data_evaluate = pd.DataFrame(Metrics)
    data_evaluate.to_csv('F:/VScode/Pylearn/Test_1/figures/Metrics.csv')
    np.savetxt('F:/VScode/Pylearn/Test_1/figures/y_test.csv', y_test_all, delimiter=',')
    np.savetxt('F:/VScode/Pylearn/Test_1/figures/predictions.csv', predictions_all, delimiter=',')
    print('[File] File Saved.')

if __name__ == '__main__':
    main()