import pandas as pd
from pandas_datareader import data as pdr
import sklearn 
import keras
import tensorflow as tf
import numpy as np
import yfinance as yf
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
from hyperopt import fmin, tpe, Trials,hp
import numpy as np
from hyperopt.pyll import scope
import time as time
param_hyperopt= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5,100,5)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 1500, 5)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time
from sklearn.metrics import r2_score

def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    
    def f_to_min1(hps, X, y, ncv=5):
        model = f_clf1(hps)
        cv_res = cross_val_score(model, X, y, cv=StratifiedKFold(ncv, random_state=SEED), 
                                 scoring='roc_auc', n_jobs=-1)
        return -cv_resx.mean()

    def objective_function(params):
        clf = lgb.LGBMRegressor(**params)
        score = cross_val_score(clf, X_train, y_train, cv=5,scoring ="r2").mean()
        return {'loss': -score, 'status': STATUS_OK}

    def objective_function_mean_err(params):
        clf = lgb.LGBMRegressor(**params)
        score = cross_val_score(clf, X_train, y_train, cv=5,scoring ="neg_mean_squared_error").mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(1))
    loss = [x['result']['loss'] for x in trials.trials]
    
    best_param_values = [x for x in best_param.values()]
    
    if best_param_values[0] == 0:
        boosting_type = 'gbdt'
    else:
        boosting_type= 'dart'
    
    clf_best = lgb.LGBMRegressor(learning_rate=best_param_values[2],
                                  num_leaves=int(best_param_values[5]),
                                  max_depth=int(best_param_values[3]),
                                  n_estimators=int(best_param_values[4]),
                                  boosting_type=boosting_type,
                                  colsample_bytree=best_param_values[1],
                                  reg_lambda=best_param_values[6],
                                 )
                                  
    clf_best.fit(X_train, y_train)
    
    
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Test Score: ", clf_best.score(X_test, y_test))
    print("Parameter combinations evaluated: ", num_eval)
    
    return clf_best,trials
regr, trails = hyperopt(param_hyperopt,x_train,y_train,x_test,y_test,300)
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
pred = regr.predict(x_test)
plt.plot(y_test,label='Y_test',color="chocolate")
plt.plot(pred,label='prediction',color="green")
plt.title("Comparing residuel return")
plt.legend()
plt.show()
print("r2 score:",r2_score(y_test,pred))
print("median abs err",median_absolute_error(y_test,pred))