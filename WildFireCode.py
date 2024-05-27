import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

#%%
data = xr.open_dataset('ERA5_new.nc')
#%%
firedata = xr.open_dataset('fire_NA.nc')
#%%
one_gridpoint = xr.concat([da[:,120,0] for _, da in data.data_vars.items()], dim='variable')


one_gridpoint = one_gridpoint.to_numpy().T


all_grid_points = pd.DataFrame(data=one_gridpoint, columns=list(data.variables)[3:])


#%%
gridpoints = xr.concat([da[:,:,:125] for _, da in data.data_vars.items()], dim='variable')

#%%
gridpoints2 = xr.concat([da[:,:,125:250] for _, da in data.data_vars.items()], dim='variable')
gridpoints3 = xr.concat([da[:,:,250:375] for _, da in data.data_vars.items()], dim='variable')
gridpoints4 = xr.concat([da[:,:,375:] for _, da in data.data_vars.items()], dim='variable')

#%%
gridpointlist = [gridpoints,gridpoints2,gridpoints3,gridpoints4]
#%%

lats = len(data['latitude'])
lons = len(data['longitude'])
sst = data['sst'][0,:,:]
var_names = list(data.variables)[3:]
for i in range(lats):
    for j in range(50):
        if np.isnan(sst[i,j]) == False:
            continue
        if i == 120 and j == 0:
            continue
        gridpoint = gridpoints[:,:,i,j].T
        # gridpoint = xr.concat([da[:,i,j] for _, da in data.data_vars.items()], dim='variable').T
        gridpoint = pd.DataFrame(data=gridpoint.to_numpy(), columns=var_names)
        all_grid_points = pd.concat([all_grid_points,gridpoint])



#%%
gridpoints2 = xr.concat([da[:,46,45] for _, da in data.data_vars.items()], dim='variable').T


gridpoints2 = pd.DataFrame(data=gridpoints2.to_numpy(), columns=list(data.variables)[3:])

#%%
wow2 = pd.concat([wow,gridpoints2])
# wow2 = wow.append(gridpoints2, ignore_index=True)


#%%
damn = xr.concat([gridpoints, gridpoints2], dim=gridpoints.dims)