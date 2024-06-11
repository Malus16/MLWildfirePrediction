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

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import sklearn.preprocessing as preprocessing
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, RocCurveDisplay, f1_score, DetCurveDisplay, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import shap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle

from tensorflow import keras
#%%
# all_data = pd.read_csv('Tabulated grid data w dew wo corr.csv').drop('sst', axis=1)
all_data_3by3 = pd.read_parquet('D:/FINAL PROJECT/Tabulated grid data w dew wo corr 3by3kernelavg.parquet').drop('sst', axis=1)
#%%
all_data = pd.read_parquet('D:/FINAL PROJECT/Tabulated grid data w dew wo corr.parquet').drop('sst', axis=1)
#%%
test_data=all_data[:1000]
cols = list(test_data.filter(like='lat')) + list(test_data.filter(like='lon'))
test_data['left_neighbor'] = test_data.groupby('latitude')['longitude'].shift(1)
#%%
# with open("E:/classifier_nolatlon_3by3kernelavg_NN14x14x14.pkl", "rb") as f:
with open("classifier_nolatlon_alldata_posweight1_25var_1000000points.pkl", "rb") as f:
    clf_25var_posweight1 = pickle.load(f)
#%%
# columns_to_drop = ['u100', 'v100', 'u10n', 'v10n', 'stl1', 'stl2', 'strdc', 'ttrc', 'ssrdc', 'tisr', 'ssrd', 'slhf', 'crr', 'ilspf', 
#                    'alnid', 'ishf', 'stl2', 'stl3', 'tsr', 'tsrc', 'tisr', 'strd', 'aluvd', 'swvl2']
coords = ['time', 'latitude', 'longitude']
# coords = ['time']
# all_data = all_data[list(set(all_data.columns) - set(columns_to_drop))].sort_index(axis=1)

#%%
exclude_months = ['2019-06-01', '2018-06-01']
dates_df = pd.to_datetime(exclude_months)
# all_but_one_month = all_data[all_data['time'] != the_month]
# all_but_one_month = all_data.loc[~all_data['time'].isin(the_month)]
all_but_exluded_months = all_data[~all_data.index.get_level_values('time').isin(dates_df)]
#%%
# train = train.sample(frac=0.1, random_state=112)
only_burn_true = all_but_exluded_months.loc[all_but_exluded_months['burned_area'] > 0]
# only_burn_true.loc[:,'burned_area'] = 1.0
#%%
some_noburn = all_but_exluded_months.loc[all_data['burned_area'] == 0]#.sample(frac=0.4)
#%%
train = pd.concat([only_burn_true, some_noburn], ignore_index=True)
#%%
# check = train.isna().max()
train = all_but_exluded_months.dropna()
#%%
# train_scaled = train[list(set(train.columns) - set(coords))].drop('burned_area', axis=1).sort_index(axis=1)#.drop('fraction_of_burnable_area', axis=1)
train_scaled = train.drop('burned_area', axis=1)
# columns = train_scaled.columns
# scaler = preprocessing.StandardScaler()
# train_scaled = scaler.fit_transform(train_scaled)
# train_scaled = pd.DataFrame(data=train_scaled, columns=columns)


X = train_scaled
y = train['burned_area']
weights = np.log1p(y)
y = y.where(y == 0, 1)
# y = np.log1p(train['burned_area'])
#%%

X_train, X_test, weight_train, weight_test = train_test_split(X, 
                                                    weights, 
                                                    test_size=0.20, 
                                                    random_state=111)

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
best_vars_perm = pd.read_csv('permutation_ranking.csv', header=None, index_col=False)
train_new_perm25 = train_scaled[best_vars_perm[0].values[0:25]]
X_train25, X_test25, y_train25, y_test25 = train_test_split(train_new_perm25, 
                                                    y,
                                                    test_size=0.20, 
                                                    random_state=111)


#%%
# one_month = all_data.loc[all_data['time'].isin(the_month)]#all_data[all_data['time'] == the_month]
excluded_months = all_data[all_data.index.get_level_values('time').isin(dates_df)]

# one_month.loc[:,'burned_area'] = 1.0
excluded_months['burned_area'] = excluded_months['burned_area'].where(excluded_months['burned_area'] == 0, other=1)
# exluded_months_X = exluded_months[list(set(exluded_months.columns) - set(coords))].drop(['burned_area'], axis=1).sort_index(axis=1)
excluded_months_X = excluded_months.drop(['burned_area'], axis=1)
excluded_months_X = excluded_months_X[best_vars_perm[0].values[0:25]]
# excluded_months_X = scaler.transform(excluded_months_X)
# excluded_months_X = pd.DataFrame(data=excluded_months_X, columns=columns)
#%%
weight_train_scaled = weight_train+(weight_train-9)*4
weight_train_scaled = weight_train_scaled.where(weight_train_scaled > 0, 1)

#%%
from sklearn.utils.class_weight import compute_class_weight

# Get unique class labels and their counts
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
#%%

# w_latlon
pos_weight = 20
clf = xgb.XGBClassifier(n_estimators=150, max_depth=30, max_bin=100, learning_rate=0.05, tree_method="hist", 
                        scale_pos_weight=pos_weight, early_stopping_rounds=2, objective='binary:logistic')
# clf.fit(X_train25[:1000000],y_train25[:1000000], eval_set=[(X_test25[:1000000], y_test25[:1000000])], verbose=1)#, sample_weight = weight_train_scaled)
clf.fit(X_train25,y_train25, eval_set=[(X_test25, y_test25)], verbose=1)#, sample_weight = weight_train_scaled)

print(accuracy_score(y_test25, clf.predict(X_test25)))
# clf.best_score
log_loss(y_test25, clf.predict(X_test25))

 #%%
clf_nolatlon = xgb.Booster.load_model(fname="model_nolatlon_alldata_burnweight.json")

#%%
clf_nn = MLPClassifier(solver='sgd', alpha=1e-5, early_stopping=True, learning_rate_init=0.1, learning_rate='adaptive',
                    hidden_layer_sizes=(14,14,14)).fit(X_train, y_train)

print(accuracy_score(y_test, clf_nn.predict(X_test)))
# clf_nn.loss_curve_[-1]
log_loss(y_test, clf_nn.predict(X_test))

#%%

model = keras.Sequential([
  keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with 16 neurons and ReLU activation
  keras.layers.Dense(14, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)
model.fit(X_train, y_train, epochs=10, batch_size=32,  class_weight = {0: 1, 1: 15.0})


#%%
permutation_result = permutation_importance(clf, X_test25[:10000], y_test25[:10000])
best_vars_perm = pd.DataFrame(data=permutation_result.importances_mean, index=X_test25.columns).sort_values(by=0, ascending=False)
#%%
# best_vars_perm.to_csv('permutation_ranking.csv')
best_vars_perm = pd.read_csv('permutation_ranking.csv', header=None, index_col=False)
#%%
# best_vars_perm = best_vars_perm.set_index('0')['1']
best_vars_perm.columns = ['features', 'values']
best_vars_perm = best_vars_perm.set_index('features')['values']

best_vars_perm.iloc[:15].plot(kind='bar', color='darkolivegreen')
plt.xticks(rotation=45)
plt.title('Permutation importance')
#%%
explainer = shap.TreeExplainer(clf_nolatlon)
shap_values = explainer(X_test[:500], check_additivity=False)

shap.plots.bar(shap_values)
#%%
best_vars_shap = pd.DataFrame(data=np.abs(shap_values.values).mean(0), index=X_test.columns).sort_values(by=0, ascending=False)
#%%
# train_new_perm = train_scaled[best_vars_perm[0].index[0:10]]
train_new_perm10 = train_scaled[best_vars_perm[0].values[0:10]]
train_new_perm25 = train_scaled[best_vars_perm[0].values[0:25]]

# train_new_shap = train_scaled[best_vars_shap[0].index[0:25]]

X_train, X_test10, y_train, y_test10 = train_test_split(train_new_perm10, 
                                                    y,
                                                    test_size=0.20, 
                                                    random_state=111)

X_train25, X_test25, y_train25, y_test25 = train_test_split(train_new_perm25, 
                                                    y,
                                                    test_size=0.20, 
                                                    random_state=111)

#%%
with open("classifier_nolatlon_alldata_posweight20_10var_500000points.pkl", "rb") as f:
    clf_10var = pickle.load(f)

with open("classifier_nolatlon_alldata_posweight20_25var_500000points.pkl", "rb") as f:
    clf_25var = pickle.load(f)

with open("classifier_nolatlon_alldata_posweight20_50var_500000points.pkl", "rb") as f:
    clf_50var = pickle.load(f)
    
#%%

# confusion_matrix(y_test, clf.predict(X_test))
fig, ax = plt.subplots(figsize=[5,5])
# RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax)
# f1_score(y_test, clf.predict(X_test))
# DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, **{'label': 'With lat-lon'})
# DetCurveDisplay.from_estimator(clf_nolatlon, X_test_nolatlon, y_test_nolatlon, ax=ax, **{'label': 'Without lat-lon'})
RocCurveDisplay.from_estimator(clf_10var, X_test10, y_test, ax=ax, **{'label': '10 variables (AUC = 0.91)', 'color':'firebrick'})#, **{'label': 'With lat-lon'})
RocCurveDisplay.from_estimator(clf_25var, X_test25, y_test, ax=ax, **{'label': '25 variables (AUC = 0.93)', 'color':'darkolivegreen'})
RocCurveDisplay.from_estimator(clf_50var, X_test, y_test, ax=ax, **{'label': '50 variables (AUC = 0.93)', 'color':'steelblue'})
# plt.legend(loc='best')
plt.title('XGBoost ROC-curves')
# RocCurveDisplay.from_estimator(clf_nolatlon, X_test_nolatlon, y_test_nolatlon, ax=ax)#, **{'label': 'Without lat-lon'})

#%%
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def DecisionTree_CrossValidation(max_depth, min_samples_leaf, learning_rate, num_estimators, data, targets):
    """Decision Tree cross validation.
       Fits a Decision Tree with the given paramaters to the target 
       given data, calculated a CV accuracy score and returns the mean.
       The goal is to find combinations of max_depth, min_samples_leaf 
       that maximize the accuracy
    """
    
    estimator = xgb.XGBClassifier(max_depth=max_depth,       
                                      learning_rate=learning_rate,
                                      n_estimators=num_estimators,
                                      min_samples_leaf=min_samples_leaf
)
    
    cval = cross_val_score(estimator, data, targets, scoring='accuracy', cv=5)
    
    return cval.mean()

def optimize_DecisionTree(data, targets, pars, n_iter=5):
    """Apply Bayesian Optimization to Decision Tree parameters."""
    
    def crossval_wrapper(max_depth, min_samples_leaf, learning_rate, num_estimators):
        """Wrapper of Decision Tree cross validation. 
           Notice how we ensure max_depth, min_samples_leaf 
           are casted to integer before we pass them along.
        """
        return DecisionTree_CrossValidation(max_depth=int(max_depth), 
                                            min_samples_leaf=int(min_samples_leaf),
                                            learning_rate=learning_rate,
                                            num_estimators=int(num_estimators),
                                            data=data, 
                                            targets=targets)

    optimizer = BayesianOptimization(f=crossval_wrapper, 
                                     pbounds=pars, 
                                     random_state=42, 
                                     verbose=2)
    optimizer.maximize(init_points=4, n_iter=n_iter)

    return optimizer
#%%
# one_month_X = one_month_X[best_vars_perm[0].index[0:10]]
pred = clf.predict_proba(excluded_months_X)[:,1]
# pred = clf_nn.predict(excluded_months_X.dropna())
# corr = X_train.corr()
# corr = corr.abs()
# upper_corr = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
# highly_correlated = upper_corr[upper_corr > 0.95]
#%%
unique_lats = np.linspace(15,90,301)
unique_lons = np.linspace(-170,-45,501)

def create_grid(grid,pred=None):
# Create an empty grid array with the size based on unique values
    grid_array = np.zeros((301, 501))
    
    # Loop through the dataframe and populate the grid array
    i = 0
    for index, row in grid.iterrows():
        # lat = row['latitude']
        # lon = row['longitude']
        lat = index[1]
        lon = index[0]
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

grid_array = create_grid(excluded_months, pred=pred)

grid_array_data = create_grid(excluded_months)
#%%
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.colors as colors
unique_lats = np.linspace(15,90,301)
unique_lons = np.linspace(-170,-45,501)

fig = plt.figure(layout='tight')
fig.set_figwidth(10)
cmap_bin = colors.LinearSegmentedColormap.from_list("", ["white", "darkred"], N=2)
cmap_diff = colors.LinearSegmentedColormap.from_list("", ["blue", "White", "red"], N=9)
cmap_cont = colors.LinearSegmentedColormap.from_list("", ["white", "darkred"], N=10)
crs = ccrs.Miller(central_longitude=-110)#, standard_parallels=(30,55))
ax = plt.axes(projection=crs)
lakes_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m')
# rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_north_america', '10m')

ax.set_extent([-170, -45, 15, 70], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(lakes_50m)
# ax.add_feature(rivers_50m)
ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, dms=True, x_inline=False, y_inline=False)
c = plt.contourf(unique_lons, unique_lats, grid_array_data, transform=ccrs.PlateCarree(), cmap=cmap_bin) # Binary plotting
# c = plt.contourf(unique_lons, unique_lats, grid_array, transform=ccrs.PlateCarree(), cmap=cmap_cont, levels=np.linspace(0,1,11)) # Continous plotting
# cb = plt.colorbar(c, location='right', shrink=1) # Use with continuous

c = plt.contourf(unique_lons, unique_lats, grid_array-grid_array_data, transform=ccrs.PlateCarree(), cmap=cmap_diff, levels=np.linspace(-1,1,10)) # Continous plotting
cb = plt.colorbar(c, location='right', shrink=1) # Use with continuous
# plt.title(f'Burned area: 2019-06 XGBoost model prediction pos_weight=20')
# plt.title(f'Burned area: 2018-06 model prediction pos_weight={pos_weight}')
# plt.title('Burned area: 2019-06 dataset values')
plt.title('Burned area: 2019-06 model-data difference map')


#%%
cm = confusion_matrix(y_test, clf.predict(X_test25), labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm)*100,
                              display_labels=clf.classes_)
#%%
disp.plot()
plt.title('Confusion matrix: No weighting [%]')
#%%

with open("classifier_nolatlon_alldata_posweight20_25var_allpoints.pkl", "wb") as f:
    pickle.dump(clf, f)

#%%
with open("classifier_nolatlon.pkl", "rb") as f:
    loaded_object = pickle.load(f)
    
#%%
wow = grid_array.astype(np.bool_)
#%%
from PIL import Image

img = Image.fromarray(wow[0:80,200:300])

img.save('gridPIL.png')

#%%
shuffled = train_scaled.sample(frac=0.05, random_state=111)
labels = train['burned_area'].sample(frac=0.05, random_state=111)
#%%
from sklearn.manifold import TSNE
TSNE_data = TSNE(perplexity=50, early_exaggeration=20,random_state=112).fit_transform(shuffled)

#%%
import umap
umap_data = umap.UMAP(n_neighbors=50).fit_transform(shuffled)

#%%
def dim_reduce_plot(data, method_name):
    indices = np.where(labels==1)[0]
    plt.scatter(data[:,0][~indices], data[:,1][~indices], marker='.', color='darkgreen', alpha=1, label='No burn')
    plt.scatter(data[:,0][indices], data[:,1][indices], marker='.', color='orangered', alpha=0.5, label='Burn')
    plt.title(f'{method_name} on balanced tabulated data')
    plt.legend()

#%%
dim_reduce_plot(umap_data, 'UMAP')

#%%
tablenum = 7
plt.table(colLabels=all_data.iloc[:tablenum,:tablenum].columns, cellText=all_data.iloc[:tablenum,:tablenum].values, loc='center')
plt.axis('off')

#%%
xgb.plot_tree(clf, num_trees=1)