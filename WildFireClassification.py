# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:23:55 2024

@author: mnok
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
import sklearn.preprocessing as preprocessing
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, RocCurveDisplay, f1_score, DetCurveDisplay
from sklearn.inspection import permutation_importance
import shap

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
all_data = pd.read_csv('Tabulated grid data.csv').iloc[:,1:]

#%%
# train = train.sample(frac=0.1, random_state=112)
only_burn_true = all_data.loc[all_data['burned_area'] > 0]
only_burn_true.loc[:,'burned_area'] = 1.0
#%%
some_noburn = all_data.loc[all_data['burned_area'] == 0].sample(frac=0.05, random_state=112)
#%%
train = pd.concat([only_burn_true, some_noburn], ignore_index=True)
#%%
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
clf = xgb.XGBClassifier(n_estimators=150, max_depth=25, max_bin=100, learning_rate=0.1, tree_method="hist", early_stopping_rounds=2, objective='binary:logistic')
clf.fit(X_train,y_train, eval_set=[(X_test, y_test)], verbose=1);

print(accuracy_score(y_test, clf.predict(X_test)))
clf.best_score
# accuracy_score(y_train, clf.predict(X_train))
log_loss(y_test, clf.predict(X_test))
#%%
# confusion_matrix(y_test, clf.predict(X_test))
fig, ax = plt.subplots(figsize=[5,5])
RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)
# f1_score(y_test, clf.predict(X_test))
# DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)

#%%
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)