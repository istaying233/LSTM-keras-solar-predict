import numpy as np
import pandas as pd
import math
import os
from sklearn import metrics
import tensorflow as tf
from tensorflow import placeholder

data = []
data_txt_18 = np.array(np.loadtxt('F:/VScode/Pylearn/Test_1/data/CHAMP/CHAMP_Density_04_018_v1.txt'))
c = [0, 15]
data_txt_18 = data_txt_18[:, c]
a1 = 1.32e-12
a2 = 2.53e-12
a3 = (a2 - a1) / 72
for j in range(72):
    ins = [j * 60 + 57840, a1 + j * a3]
    data_txt_18 = np.insert(data_txt_18, 5784 + j, ins, axis=0)
for i in range(len(data_txt_18[:, 0])):
    if data_txt_18[i, 0] % 60 == 0:
        data.append(data_txt_18[i, 0])
print(len(data))

txtpath = 'F:/VScode/Pylearn/Test_1/data/CHAMP/'
txttype = 'txt'
txtlists = os.listdir(txtpath)
data_20 = []
for txt in txtlists:
    data_txt = np.array(np.loadtxt(txtpath + txt))
    c = [0, 15]
    data_txt = data_txt[:, c]
    data = []

    if txt == 'CHAMP_Density_04_018_v1.txt':
        a1 = 1.32e-12
        a2 = 2.53e-12
        a3 = (a2 - a1) / 72
        for j in range(72):
            ins = [j * 60 + 57840, a1 + j * a3]
            data_txt = np.insert(data_txt, 5785 + j, ins, axis=0)
    else:
        for i in range(1440):
            if data_txt[i * 6, 0] != i * 60:
                a = (data_txt[i * 6 + 1, 1] + data_txt[i * 6 - 1, 1]) / 2
                ins = [60 * i, a]
                data_txt = np.insert(data_txt, i * 6, ins, axis=0)    
    for k in range(len(data_txt[:, 0])):
        if data_txt[k, 0] % 60 == 0:
            data.append(data_txt[k, 1])
    data_20.append(data)
    print(' length ', txt, len(data))
data_20 = np.array(data_20)
data_all = data_20.reshape(len(data_20[:, 0]) * len(data_20[0, :]), 1)
np.savetxt('F:/VScode/Pylearn/Test_1/data/data_all.csv', data_all, delimiter=',')
a = 1