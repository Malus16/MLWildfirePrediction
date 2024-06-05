# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:23:55 2024

@author: mnok
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
import sklearn.preprocessing as preprocessing
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, RocCurveDisplay, f1_score, DetCurveDisplay
from sklearn.inspection import permutation_importance
import shap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
all_data = pd.read_csv('Tabulated grid data w dew wo corr.csv').drop('sst', axis=1)

#%%
# columns_to_drop = ['u100', 'v100', 'u10n', 'v10n', 'stl1', 'stl2', 'strdc', 'ttrc', 'ssrdc', 'tisr', 'ssrd', 'slhf', 'crr', 'ilspf', 
#                    'alnid', 'ishf', 'stl2', 'stl3', 'tsr', 'tsrc', 'tisr', 'strd', 'aluvd', 'swvl2']
# coords = ['time', 'latitude', 'longitude']
coords = ['time']
# all_data = all_data[list(set(all_data.columns) - set(columns_to_drop))].sort_index(axis=1)

#%%
the_month = '2019-06-01'
all_but_one_month = all_data[all_data['time'] != the_month]
#%%
# train = train.sample(frac=0.1, random_state=112)
only_burn_true = all_but_one_month.loc[all_but_one_month['burned_area'] > 0]
only_burn_true.loc[:,'burned_area'] = 1.0
#%%
some_noburn = all_but_one_month.loc[all_data['burned_area'] == 0].sample(frac=0.15, random_state=112)
#%%
train = pd.concat([only_burn_true, some_noburn], ignore_index=True)
#%%
# check = train.isna().max()
train = train.dropna()
#%%
train_scaled = train[list(set(train.columns) - set(coords))].drop('burned_area', axis=1).sort_index(axis=1)#.drop('fraction_of_burnable_area', axis=1)
# columns = train_scaled.columns
# scaler = preprocessing.StandardScaler()
# train_scaled = scaler.fit_transform(train_scaled)
# train_scaled = pd.DataFrame(data=train_scaled, columns=columns)


X = train_scaled
y = train['burned_area']
# y = np.log1p(train['burned_area'])

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=111)
#%%
X_train_nolatlon, X_test_nolatlon, y_train_nolatlon, y_test_nolatlon = train_test_split(X, 
                                                                                        y, 
                                                                                        test_size=0.20, 
                                                                                        random_state=111)

#%%
one_month = all_data[all_data['time'] == the_month]
one_month.loc[:,'burned_area'] = 1.0
# one_month['burned_area'] = one_month['burned_area'].where(one_month['burned_area'] == 0, other=1)
one_month_X = one_month[list(set(one_month.columns) - set(coords))].drop(['burned_area'], axis=1).sort_index(axis=1)
# one_month_X = scaler.transform(one_month_X)
# one_month_X = pd.DataFrame(data=one_month_X, columns=columns)
#%%
# w_latlon
pos_weight = 1
clf = xgb.XGBClassifier(n_estimators=150, max_depth=30, max_bin=100, learning_rate=0.05, tree_method="hist", 
                        scale_pos_weight=pos_weight, early_stopping_rounds=2, objective='binary:logistic')
clf.fit(X_train,y_train, eval_set=[(X_test, y_test)], verbose=1)

print(accuracy_score(y_test, clf.predict(X_test)))
clf.best_score
# accuracy_score(y_train, clf.predict(X_train))
log_loss(y_test, clf.predict(X_test))

#%%
# no_latlon
pos_weight = 1
clf_nolatlon = xgb.XGBClassifier(n_estimators=150, max_depth=30, max_bin=100, learning_rate=0.05, tree_method="hist", 
                        scale_pos_weight=pos_weight, early_stopping_rounds=2, objective='binary:logistic')
clf_nolatlon.fit(X_train,y_train, eval_set=[(X_test, y_test)], verbose=1)

print(accuracy_score(y_test, clf_nolatlon.predict(X_test)))
clf.best_score
# accuracy_score(y_train, clf.predict(X_train))
log_loss(y_test, clf_nolatlon.predict(X_test))
#%%
# confusion_matrix(y_test, clf.predict(X_test))
fig, ax = plt.subplots(figsize=[5,5])
# RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)
# f1_score(y_test, clf.predict(X_test))
# DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, **{'label': 'With lat-lon'})
# DetCurveDisplay.from_estimator(clf_nolatlon, X_test_nolatlon, y_test_nolatlon, ax=ax, **{'label': 'Without lat-lon'})
RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)#, **{'label': 'With lat-lon'})
RocCurveDisplay.from_estimator(clf_nolatlon, X_test_nolatlon, y_test_nolatlon, ax=ax)#, **{'label': 'Without lat-lon'})

#%%
permutation_result = permutation_importance(clf, X_test[:10000], y_test[:10000])
best_vars_pd2 = pd.DataFrame(data=permutation_result.importances_mean, index=X_test.columns).sort_values(by=0, ascending=False)

#%%
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test[:500], check_additivity=False)

shap.plots.bar(shap_values)

#%%
# pred = clf.predict_proba(one_month_X)[:,1]
pred = clf.predict(one_month_X)
# corr = X_train.corr()
# corr = corr.abs()
# upper_corr = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
# highly_correlated = upper_corr[upper_corr > 0.95]
# columns_to_drop = ['u100', 'v100', 'u10n', 'v10n', 'stl1', 'stl2', 'strdc', 'ttrc', 'ssrdc', 'tisr', 'ssrd', 'slhf', 'crr', 'ilspf', 
#                    'alnid', 'ishf', 'stl2', 'stl3', 'tsr', 'tsrc', 'tisr', 'strd', 'aluvd', 'swvl2']
#%%
unique_lats = np.linspace(15,90,301)
unique_lons = np.linspace(-170,-45,501)

def create_grid(grid,pred=None):
# Create an empty grid array with the size based on unique values
    grid_array = np.zeros((301, 501))
    
    # Loop through the dataframe and populate the grid array
    i = 0
    for index, row in grid.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        if type(pred) == np.ndarray:
            data_value = pred[i]
        else:
            data_value = row['burned_area']
        
        # Find the corresponding indices in the grid array for this lat/lon
        lat_index = np.where(unique_lats == lat)[0][0]
        lon_index = np.where(unique_lons == lon)[0][0]
        
        # Assign the data value to the corresponding gridpoint
        grid_array[lat_index, lon_index] = data_value
        i += 1
    
    return grid_array

grid_array = create_grid(one_month, pred=pred)

# grid_array_wlatlon = grid_array
#%%
fig = plt.figure(layout='tight')
fig.set_figwidth(10)
cmap = colors.LinearSegmentedColormap.from_list("", ["white", "darkred"], N=2)
crs = ccrs.Miller(central_longitude=-110)#, standard_parallels=(30,55))
ax = plt.axes(projection=crs)
lakes_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')
# rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_north_america', '10m')

ax.set_extent([-170, -45, 15, 70], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(lakes_50m)
# ax.add_feature(rivers_50m)
ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, dms=True, x_inline=False, y_inline=False)
c = plt.contourf(unique_lons, unique_lats, grid_array, transform=ccrs.PlateCarree(), cmap=cmap)
# c = plt.contourf(unique_lons, unique_lats, grid_array, transform=ccrs.PlateCarree(), cmap=cmap, levels=np.linspace(0,1,11))
# cb = plt.colorbar(c, location='right', shrink=1)
plt.title(f'Burned area: 2019-06 model prediction pos_weight={pos_weight}')
# plt.title(f'Burned area: 2018-06 model prediction pos_weight={pos_weight}')
# plt.title('Burned area: 2018-06 reanalysis data')

#%%
import pickle
with open("classifier_nolatlon.pkl", "wb") as f:
    pickle.dump(clf_nolatlon, f)

#%%
with open("classifier_nolatlon.pkl", "rb") as f:
    loaded_object = pickle.load(f)