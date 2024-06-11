import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
#%%
dew_veg = xr.open_dataset('E:/ERA5_2mdew_veg.nc')
#%%
merge_fixed = xr.open_dataset('E:/merge_fixed.nc')
#%%

data = xr.open_dataset('E:/ERA5.nc')
data = data.isel(time=slice(None,-36))
# data['fraction_of_burnable_area'] = fraction_burnable

columns_to_drop = ['u100', 'v100', 'u10n', 'v10n', 'stl1', 'stl2', 'strdc', 'ttrc', 'ssrdc', 'tisr', 'ssrd', 'slhf', 'crr', 'ilspf', 
                   'alnid', 'ishf', 'stl2', 'stl3', 'tsr', 'tsrc', 'tisr', 'strd', 'aluvd', 'swvl2'] # Based on correlation analysis
data = data.drop_vars(columns_to_drop)
#%%
dew_veg.coords['time'] = data['t2m'].coords['time']
for var_name, da in dew_veg.data_vars.items():
    data[var_name] = dew_veg[var_name]

#%%
data = data.sel(time=xr.DataArray([f'20{y:02d}-{m:02d}-01' for y in range(1,20) for m in range(4,11)], dims=['time']))
#%%
sst = data['sst']
data = data.drop_vars('sst')
#%%
data = data.pad({'latitude': (1,1), 'longitude': (1,1)}, mode='reflect')
#%%
data = data.rolling({'latitude': 3, 'longitude': 3}, center=True).mean()
#%%
# test4 = test3[0,:10,:10].to_numpy()
data = data.isel(longitude=slice(1,-1), latitude=slice(1,-1))

#%%
# test5 = ndimage.generic_filter(data['t2m'], np.nanmean, footprint=np.ones((1,3,3)), mode='reflect')
burned_area = merge_fixed['burned_area']
fraction_burnable = merge_fixed['fraction_of_burnable_area']

burned_area.coords['time'] = xr.open_dataset('E:/ERA5.nc').isel(time=slice(None,-36))['t2m'].coords['time']
# fraction_burnable.coords['time'] = data['t2m'].coords['time']

burned_area = burned_area.sel(time=[f'20{y:02d}-{m:02d}-01' for y in range(1,20) for m in range(4,11)])
#%%
data['burned_area'] = burned_area
# data['sst'] = sst
#%%
for var_name, da in data.data_vars.items():
    if da.dtype == "float64":
        data[var_name] = da.astype("float32")

#%%
data_dict = {}
for x in data.data_vars:
    data_dict[x] = {'dtype':'float32', 'zlib':True}

#%%
encoding = {"zlib": True}
data.to_netcdf("E:/ERA5_wBurned&dewveg_apr-oct_float32.nc", format="NETCDF4", engine="netcdf4", encoding=data_dict)
#%%
all_gridpoints = data.to_dataframe()

#%%
# all_gridpoints = all_gridpoints.dropna(subset=['burned_area'])
all_gridpoints = all_gridpoints[all_gridpoints['sst'].isna()]
#%%
# all_gridpoints.to_csv('Tabulated grid data w dew wo corr 3by3kernelavg.csv')
all_gridpoints.to_parquet('F:/FINAL PROJECT/Tabulated grid data w dew wo corr 3by3kernelavg.parquet')
#%%
for var in dew_veg.data_vars:
    print(f'{var}: {data[var].long_name}')
