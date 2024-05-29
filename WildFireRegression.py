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

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
train = pd.read_csv('Tabulated grid data.csv').iloc[:,1:]

#%%
train = train.sample(frac=0.1, random_state=112)
#%%
train_scaled = train.drop('burned_area', axis=1).drop('fraction_of_burnable_area', axis=1)
columns = train_scaled.columns
scaler = preprocessing.StandardScaler()
train_scaled = scaler.fit_transform(train_scaled)
train_scaled = pd.DataFrame(data=train_scaled, columns=columns)

X = train_scaled
y = train['burned_area']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=111)

#%%
clf = xgb.XGBRegressor(n_estimators=120, max_depth=7, max_bin=100, learning_rate=0.2, tree_method="hist", early_stopping_rounds=2, objective='reg:absoluteerror')
clf.fit(X_train,y_train, eval_set=[(X_test, y_test)], verbose=0)

print(mean_squared_error(y_test, clf.predict(X_test)))
# print(mean_absolute_error(y_test, clf.predict(X_test)))
# r2_score(y_test, clf.predict(X_test))
clf.best_score
