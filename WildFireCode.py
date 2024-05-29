import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

#%%
data = xr.open_dataset('merge_fixed.nc')
#%%
firedata = xr.open_dataset('fire_NA.nc')
#%%
one_gridpoint = xr.concat([da[:36,120,0] for _, da in data.data_vars.items()], dim='variable')


one_gridpoint = one_gridpoint.to_numpy().T

#%%
# Split the dataset up into parts so that the command doesn't overload memory
gridpoints = xr.concat([da[:,:,:] for _, da in data.data_vars.items()], dim='variable')

#%%
# If the data overloads memory, split into four
# gridpoints2 = xr.concat([da[:,:,125:250] for _, da in data.data_vars.items()], dim='variable')
# gridpoints3 = xr.concat([da[:,:,250:375] for _, da in data.data_vars.items()], dim='variable')
# gridpoints4 = xr.concat([da[:,:,375:] for _, da in data.data_vars.items()], dim='variable')

# gridpointlist = [gridpoints,gridpoints2,gridpoints3,gridpoints4]
#%%
# Initial pandas dataframe that is appended to later. Initialize on land gridpoint
all_grid_points = pd.DataFrame(data=one_gridpoint, columns=data.data_vars)
# This is the loop creating the tabulated data by appending to the dataframe
lats = len(data['latitude'])
lons = len(data['longitude'])
sst = data['sst'][0,:,:]
burned_area = data['burned_area'][0,:,:]
var_names = data.data_vars#[3:]
arrlist = [0]*125 + [1]*125 + [2]*125 + [3]*126
gridpointlist = []
cnt = 0
for i in range(lats):
    for j in range(lons):
        arr = arrlist[j]
        if np.isnan(sst[i,j]) == False or np.isnan(burned_area[i,j]) == True:
            continue
        if i == 120 and j == 0:
            continue
        gridpoint = gridpoints[:,:,i,j].T#gridpointlist[arr][:,:,i,j - arr*125].T
        gridpoint = pd.DataFrame(data=gridpoint.to_numpy(), columns=var_names)
        cnt += len(gridpoint)
        gridpointlist.append(gridpoint)
    if i % 5 == 0:
        print(f'lat {i}')
        print(f'total gridpoints: {cnt}')

all_grid_points = pd.concat(gridpointlist, ignore_index=True)
#%%
all_grid_points = all_grid_points.drop('sst', axis=1)
#%%
all_grid_points.to_csv('Tabulated grid data.csv')


#%%

