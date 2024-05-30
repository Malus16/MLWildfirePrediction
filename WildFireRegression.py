# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:19:02 2024

@author: mnok
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
import sklearn.preprocessing as preprocessing
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, RocCurveDisplay, f1_score

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
all_data = pd.read_csv('Tabulated grid data.csv').iloc[:,1:]

#%%
# train = train.sample(frac=0.1, random_state=112)
only_burn_true = all_data.loc[all_data['burned_area'] > 0]
#%%
some_noburn = all_data.loc[all_data['burned_area'] == 0].sample(frac=0.05, random_state=112)
#%%
train = pd.concat([only_burn_true, some_noburn], ignore_index=True)
#%%
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
train_scaled = train.drop('burned_area', axis=1).drop('fraction_of_burnable_area', axis=1)
columns = train_scaled.columns
scaler = preprocessing.StandardScaler()
train_scaled = scaler.fit_transform(train_scaled)
train_scaled = pd.DataFrame(data=train_scaled, columns=columns)


X = train_scaled
y = train['burned_area']
# y = np.log1p(train['burned_area'])

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=111)

#%%
# REGRESSION
clf = xgb.XGBRegressor(n_estimators=150, max_depth=25, max_bin=50, learning_rate=0.05, n_jobs=4, tree_method="hist", early_stopping_rounds=2, objective='reg:squarederror')
clf.fit(X_train,y_train, eval_set=[(X_test, y_test)], verbose=1)

print(mean_squared_error(y_test, clf.predict(X_test)))
# print(mean_absolute_error(y_test, clf.predict(X_test)))
# r2_score(y_test, clf.predict(X_test))
clf.best_score


#%%
from scipy.stats import randint, uniform
# specify parameters and distributions to sample from
parameters_RandomSearch = {'max_depth': randint(5,12), 
                           'n_estimators': randint(30,70),
                           # 'n_estimators': randint(80,50),
                           'learning_rate': uniform(0.08,0.17)}

RandomSearch = RandomizedSearchCV(clf, 
                                  param_distributions=parameters_RandomSearch,
                                  n_iter=5, 
                                  cv=3, 
                                  return_train_score=True,
                                  random_state=111,
                                 )

RandomSearch.fit(X_train, y_train, eval_set=[(X_test, y_test)])

#%%
RandomSearch_results = pd.DataFrame(RandomSearch.cv_results_)

clf_RandomSearch = RandomSearch.best_estimator_
#%%
y_test.hist(bins=20)
# pd.Series(clf.predict(X_test)).hist(bins=20)
# y_test.hist(bins='auto', log=True)
# pd.Series(clf.predict(X_test)).hist(bins='auto', log=True)