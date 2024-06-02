import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
#%%
data2 = xr.open_dataset('merge_fixed.nc')
#%%
burned_area = xr.open_dataset('merge_fixed.nc')['burned_area']
data = xr.open_dataset('ERA5.nc')
data = data.isel(time=slice(None,-36))
burned_area.coords['time'] = data['t2m'].coords['time']
data['burned_area'] = burned_area
#%%
for var_name, da in data.data_vars.items():
    if da.dtype == "float64":
        data[var_name] = da.astype("float32")

#%%
data = data.sel(time=xr.DataArray([f'20{y:02d}-{m:02d}-01' for y in range(10,20) for m in range(4,11)], dims=['time']))
#%%
# data.to_netcdf("ERA5_wBurned_float32.nc", format="NETCDF4", encoding={"zlib": True})
#%%
all_gridpoints = data.to_dataframe()

#%%
# all_gridpoints = all_gridpoints.dropna(subset=['burned_area'])
all_gridpoints = all_gridpoints[all_gridpoints['sst'].isna()]
#%%
all_gridpoints.to_csv('Tabulated grid data extra var.csv')
#%%
wow_mod = wow_mod.dropna(dim='latitude', subset=['burned_area'], how='all')

#%%
for var in data.data_vars:
    print(f'{var}: {data[var].long_name}')

#%%
# Split the dataset up into parts so that the command doesn't overload memory
gridpoints = [da[:-36,:,:125] for _, da in data.data_vars.items()]
gridpoints.append(burned_area[:,:,:125])
# gridpoints = xr.concat([da[:-36,:,:125] for _, da in data.data_vars.items()], dim='variable')

# gridpoints = xr.concat([x for x in gridpoints], dim='variable')

#%%
# If the data overloads memory, split into four
# gridpoints2 = xr.concat([da[:,:,125:250] for _, da in data.data_vars.items()], dim='variable')
# gridpoints3 = xr.concat([da[:,:,250:375] for _, da in data.data_vars.items()], dim='variable')
# gridpoints4 = xr.concat([da[:,:,375:] for _, da in data.data_vars.items()], dim='variable')

# gridpointlist = [gridpoints,gridpoints2,gridpoints3,gridpoints4]
#%%
# This is the loop creating the tabulated data by appending to the dataframe
lats = len(data['latitude'])
lons = len(data['longitude'])
sst = data['sst'][0,:,:]
burned_area = data['burned_area'][0,:,:]
var_names = data.data_vars#[3:]
var_names.append('burned_area')
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

