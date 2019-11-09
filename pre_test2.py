import numpy as np
import pandas as pd
All = pd.read_csv('F:/VScode/Pylearn/Test_1/data/APF10.7.csv', index_col=0)
den = pd.read_csv('F:/VScode/Pylearn/Test_1/data/CHAMP0423.csv')
All = All.values
den = den.values
Kp = [0, 1, 2, 3, 4, 5, 6, 7]
Ap = [9, 10, 11, 12, 13, 14, 15, 16]
Kp_avg = [8]
Ap_avg = [17]
F107 = [-3]
Kp = All[:, Kp]
Ap = All[:, Ap]
Kp_avg = All[:, Kp_avg] / 8
Ap_avg = All[:, Ap_avg]
F107 = All[:, F107]
Kp = np.reshape(Kp, (Kp.shape[0] * Kp.shape[1], 1))
Ap = np.reshape(Ap, (Ap.shape[0] * Ap.shape[1], 1))
Kp_all = []
Ap_all = []
F107_aLL = []
for i in range(184):
    for j in range(180):
        Kp_all.append(Kp[i])
        Ap_all.append(Ap[i])
for h in range(23):
    for k in range(1440):
        F107_aLL.append(F107[h])
res = np.hstack((Kp_all, Ap_all, F107_aLL, den)) 
np.savetxt('F:/VScode/Pylearn/Test_1/data/CHAMPkpapdensity.csv',res, delimiter=',')       
B = 1
