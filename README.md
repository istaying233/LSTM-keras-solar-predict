# LSTM Keras solar predict

Predict solar with LSTM using keras. 

This project is based on https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction. 

## **Add:**

#### something is added for solar predicting , including:

Evaluate function

Model improvement

Normalise and De-normalise

Metrics and Calbacks



## Result:

### predict 1 day :

just means the accuracy of predicting 1 day

![](https://github.com/istaying233/LSTM-keras-solar-predict/blob/master/eve/F10.7/fig_predict.png)

![](https://github.com/istaying233/LSTM-keras-solar-predict/blob/master/eve/F10.7/fig_loss.png)

### predict 54 days:

#### low solar(from 2007.1.11 70-90 sfu)

![](https://github.com/istaying233/LSTM-keras-solar-predict/blob/master/eve/low/l_2/fig_predict.png)

R = 0.75

MRE = 0.035

#### mid solar(from 2012.08.17 100-150 sfu)

![](https://github.com/istaying233/LSTM-keras-solar-predict/blob/master/eve/mid/fig_predict.png)

R = 0.78

MRE = 0.088

#### high solar(from 1997.07.14 100-150 sfu)

![](https://github.com/istaying233/LSTM-keras-solar-predict/blob/master/eve/high/fig_predict.png)

R = 0.76

MRE = 0.135

THANKS TO MR.ZHANG'S HELP





