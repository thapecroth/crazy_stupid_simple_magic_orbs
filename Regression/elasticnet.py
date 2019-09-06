import pandas as pd
from pandas_datareader import data as pdr
import sklearn 
import keras
import tensorflow as tf
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

yf.pdr_override()
data = pdr.get_data_yahoo("SPY", start="2000-01-01", end="2019-09-5")
dff = data.diff()
y = dff['Adj Close'].shift(periods=-1)
x = dff
x2 = dff.shift(periods=1)
x3 = dff.shift(periods=2)
x = x.fillna(0)
x2 = x2.fillna(0)
x3 = x3.fillna(0)
y = y.fillna(0)
x = pd.concat([x, x2,x3], axis=1)
x = x.fillna(0)
x = x[2:len(x)]
y = y[2:len(y)]
Train = 0.8
end_train = int(len(x)*Train)
x_train = x[0:end_train]
y_train = y[0:end_train]
x_test = x[end_train:len(x)-1]
y_test = y[end_train:len(y)-1]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)


from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(x_train, y_train)  
print(regr.coef_) 
print(regr.intercept_) 


pred = regr.predict(x_test)
plt.plot(y_test,label='Y_test',color="chocolate")
plt.plot(pred,label='prediction',color="green")
plt.title("Comparing residuel return")
plt.legend()
plt.show()
print("r2 score:",r2_score(y_test,pred))
print("median abs err",median_absolute_error(y_test,pred))