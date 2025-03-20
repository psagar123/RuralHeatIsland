
# import general packages
import xarray as xr
import dask
import dask.array as da
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, vectorize
# import packages needed for obtaining google cloud data
import pandas as pd
import fsspec

# import Liljgren functions for calculating cosine zenith angle
from coszenith import coszda, cosza
# import functions for calculating WBGT
from WBGT import WBGT_Liljegren, WBGT_GCM
from WBGT import Tg_GCM, Tnwb_GCM

# import other required packages
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy.ndimage import shift


# Read NetCDF files for lat and lon
lat_dataset = nc.Dataset('/expanse/lustre/projects/sdu137/s1parajuli/wrf_output/apr_oct_20_default/1km_apr_oct_sensor_select_new_irr.nc')
long_dataset = nc.Dataset('/expanse/lustre/projects/sdu137/s1parajuli/wrf_output/apr_oct_20_default/1km_apr_oct_sensor_select_new_irr.nc')
lat = lat_dataset['XLAT'][:]
lon = long_dataset['XLONG'][:]
print(lat.shape)
print(lon.shape)


# Define a function to shift the data and match PST local time (-8 hours shift)
def shift_to_pst(data_array):
    return shift(data_array, shift=(-8, 0, 0), cval=np.nan)

# Load the NetCDF main file
dataset_path = '/expanse/lustre/projects/sdu137/s1parajuli/wrf_output/apr_oct_20_default/1km_apr_oct_sensor_select_new_irr.nc'
ds = xr.open_dataset(dataset_path)

coszen_20_aug_sensor = shift_to_pst(ds['coszen'].values)
swdown_20_aug_sensor = shift_to_pst(ds['swdown'].values)
glw_20_aug_sensor = shift_to_pst(ds['glw'].values)
swddir_20_aug_sensor = shift_to_pst(ds['swddir'].values)
swupb_20_aug_sensor = shift_to_pst(ds['swupb'].values)
lwupb_20_aug_sensor = shift_to_pst(ds['lwupb'].values)
T2_20_aug_sensor = shift_to_pst(ds['t2'].values)
u_20_aug_sensor = shift_to_pst(ds['u10'].values)
v_20_aug_sensor = shift_to_pst(ds['v10'].values)
psfc_20_aug_sensor = shift_to_pst(ds['psfc'].values)

ds.close()

# Load relative humidity and dew point temperature data from second netcdf file
rh_td_dataset_path = '/expanse/lustre/projects/sdu137/s1parajuli/wrf_output/apr_oct_20_default/1km_apr_oct_sensor_select_new_rh2m_irr.nc'
rh_td_ds = xr.open_dataset(rh_td_dataset_path)

rh_20_aug_sensor = shift_to_pst(rh_td_ds['rh2m'].values)
td_20_aug_sensor = shift_to_pst(rh_td_ds['td2m'].values)
rh_td_ds.close()

rh_20_aug_sensor = np.clip(rh_20_aug_sensor, 0, 100)

# Calculate wind speed magnitude
ws_20_aug_sensor = np.sqrt(u_20_aug_sensor**2 + v_20_aug_sensor**2); del u_20_aug_sensor, v_20_aug_sensor;

print(ws_20_aug_sensor.shape)


#import required Liljegren's functions

from WBGT import Tg_Liljegren, Tnwb_Liljegren

coszen_20_aug_sensor[coszen_20_aug_sensor <= 0] = -0.5; 


psfc = psfc_20_aug_sensor; 
czda = coszen_20_aug_sensor; # try using this first directly. If it doesn't work, calculate czda as done by Kong 
va = ws_20_aug_sensor;
rh = rh_20_aug_sensor; 
t2_k = T2_20_aug_sensor;


swdnb = swdown_20_aug_sensor;
swupb = swupb_20_aug_sensor; 
lwdnb = glw_20_aug_sensor; 
lwupb = lwupb_20_aug_sensor; 

f = swddir_20_aug_sensor/swdown_20_aug_sensor; # the ratio of the direct horizontal solar irradiance to the total horizontal solar irradiance, fdir

f = np.where(coszen_20_aug_sensor <= np.cos(89.5 / 180 * np.pi), 0, f)
f = np.where(f > 0.9, 0.9, f)
f = np.where(f < 0, 0, f)
f = np.where(swdown_20_aug_sensor <= 0, 0, f)

print(np.nanmin(f.flatten()))
print(np.nanmax(f.flatten()))


# calculate Liljegren WBGT with full treatment of radiations
wbgt_gcm_k=xr.apply_ufunc(WBGT_GCM, t2_k, rh, psfc, va, swdnb, swupb, lwdnb, lwupb, f, czda, False, output_dtypes=[float])

import thermofeel
from thermofeel import kelvin_to_fahrenheit; 
wbgt_f = kelvin_to_fahrenheit(wbgt_gcm_k);

print('min wbgt_f', np.nanmin(wbgt_f.flatten()))
print('max wbgt_f', np.nanmax(wbgt_f.flatten()))

wbgt_f +=5.4; # increase WBGT with clothing correction factor to get wbgt-eff

print(wbgt_f.shape)
print(np.nanmax(wbgt_f))
print(np.nanmin(wbgt_f))

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import MultipleLocator

# Load the county and lake shapefiles
counties = gpd.read_file("ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
salton = gpd.read_file("Salton_geog/Salton_geog.shp")






# extract data corresponding to workshifts for WBGT

wbgt_f_diurnal = np.reshape(wbgt_f, (24, 214, 171, 162), order='F');
wbgt_f_day = wbgt_f_diurnal[6:14, :, :, :]; del wbgt_f_diurnal; 

# extract wbgt_f for IV
wbgt_f_iv = wbgt_f[:, 20:70, 106:140]; del wbgt_f; 
wbgt_f_iv_diurnal = np.reshape(wbgt_f_iv, (24, 214, 50, 34), order='F');
wbgt_f_iv_day = wbgt_f_iv_diurnal[6:14, :, :, :]; del wbgt_f_iv_diurnal;


# review the data range
print('wbgt_min', np.nanmin(wbgt_f_iv.flatten()))
print('wbgt_max', np.nanmax(wbgt_f_iv.flatten()))
print(wbgt_f_iv.shape)





# Exceedance for entire study area using workshift

# first light_RAL for all data and IV both

#all data

non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_light_RAL_day = np.sum((wbgt_f_day >= 84.2) & (wbgt_f_day < 86.8), axis=(0, 1))
wbgt_f_hour_exceed_yellow_light_RAL_restm_perhour_day = wbgt_f_hour_exceed_yellow_light_RAL_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_light_RAL_day = np.sum((wbgt_f_day >= 86.8) & (wbgt_f_day < 88.3), axis=(0, 1))
wbgt_f_hour_exceed_orange_light_RAL_restm_perhour_day = wbgt_f_hour_exceed_orange_light_RAL_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_light_RAL_day = np.sum((wbgt_f_day >= 88.3) & (wbgt_f_day < 89.8), axis=(0, 1))
wbgt_f_hour_exceed_red_light_RAL_restm_perhour_day = wbgt_f_hour_exceed_red_light_RAL_day*45/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_light_RAL_day = np.sum(wbgt_f_day >= 89.8, axis=(0, 1))
wbgt_f_hour_exceed_pink_light_RAL_restm_perhour_day = wbgt_f_hour_exceed_pink_light_RAL_day*60/non_nan_counts_wbgt_day  # 45 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day_light_RAL = wbgt_f_hour_exceed_yellow_light_RAL_restm_perhour_day + wbgt_f_hour_exceed_orange_light_RAL_restm_perhour_day + wbgt_f_hour_exceed_red_light_RAL_restm_perhour_day + wbgt_f_hour_exceed_pink_light_RAL_restm_perhour_day;

#IV data

non_nan_counts_wbgt_iv_day = np.sum(~np.isnan(wbgt_f_iv_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_light_RAL_iv_day = np.sum((wbgt_f_iv_day >= 84.2) & (wbgt_f_iv_day < 86.8), axis=(0, 1))
wbgt_f_hour_exceed_yellow_light_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_yellow_light_RAL_iv_day*15/non_nan_counts_wbgt_iv_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_light_RAL_iv_day = np.sum((wbgt_f_iv_day >= 86.8) & (wbgt_f_iv_day < 88.3), axis=(0, 1))
wbgt_f_hour_exceed_orange_light_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_orange_light_RAL_iv_day*30/non_nan_counts_wbgt_iv_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_light_RAL_iv_day = np.sum((wbgt_f_iv_day >= 88.3) & (wbgt_f_iv_day < 89.8), axis=(0, 1))
wbgt_f_hour_exceed_red_light_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_red_light_RAL_iv_day*45/non_nan_counts_wbgt_iv_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_light_RAL_iv_day = np.sum(wbgt_f_iv_day >= 89.8, axis=(0, 1))
wbgt_f_hour_exceed_pink_light_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_pink_light_RAL_iv_day*60/non_nan_counts_wbgt_iv_day  # 45 minutes every hour

# total rest minutes from all above to be exported
wbgt_f_hour_exceed_total_restm_perhour_iv_day_light_RAL = wbgt_f_hour_exceed_yellow_light_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_orange_light_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_red_light_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_pink_light_RAL_restm_perhour_iv_day;


# second medium_RAL for all data and IV both

# all data
non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_medium_RAL_day = np.sum((wbgt_f_day >= 78.8) & (wbgt_f_day < 81.4), axis=(0, 1))
wbgt_f_hour_exceed_yellow_medium_RAL_restm_perhour_day = wbgt_f_hour_exceed_yellow_medium_RAL_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_medium_RAL_day = np.sum((wbgt_f_day >= 81.4) & (wbgt_f_day < 83.7), axis=(0, 1))
wbgt_f_hour_exceed_orange_medium_RAL_restm_perhour_day = wbgt_f_hour_exceed_orange_medium_RAL_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_medium_RAL_day = np.sum((wbgt_f_day >= 83.7) & (wbgt_f_day < 85.8), axis=(0, 1))
wbgt_f_hour_exceed_red_medium_RAL_restm_perhour_day = wbgt_f_hour_exceed_red_medium_RAL_day*45/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_medium_RAL_day = np.sum(wbgt_f_day >= 85.8, axis=(0, 1))
wbgt_f_hour_exceed_pink_medium_RAL_restm_perhour_day = wbgt_f_hour_exceed_pink_medium_RAL_day*60/non_nan_counts_wbgt_day  # 40 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day_medium_RAL = wbgt_f_hour_exceed_yellow_medium_RAL_restm_perhour_day + wbgt_f_hour_exceed_orange_medium_RAL_restm_perhour_day + wbgt_f_hour_exceed_red_medium_RAL_restm_perhour_day + wbgt_f_hour_exceed_pink_medium_RAL_restm_perhour_day;

# IV data
non_nan_counts_wbgt_iv_day = np.sum(~np.isnan(wbgt_f_iv_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_medium_RAL_iv_day = np.sum((wbgt_f_iv_day >= 78.8) & (wbgt_f_iv_day < 81.4), axis=(0, 1))
wbgt_f_hour_exceed_yellow_medium_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_yellow_medium_RAL_iv_day*15/non_nan_counts_wbgt_iv_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_medium_RAL_iv_day = np.sum((wbgt_f_iv_day >= 81.4) & (wbgt_f_iv_day < 83.7), axis=(0, 1))
wbgt_f_hour_exceed_orange_medium_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_orange_medium_RAL_iv_day*30/non_nan_counts_wbgt_iv_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_medium_RAL_iv_day = np.sum((wbgt_f_iv_day >= 83.7) & (wbgt_f_iv_day < 85.8), axis=(0, 1))
wbgt_f_hour_exceed_red_medium_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_red_medium_RAL_iv_day*45/non_nan_counts_wbgt_iv_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_medium_RAL_iv_day = np.sum(wbgt_f_iv_day >= 85.8, axis=(0, 1))
wbgt_f_hour_exceed_pink_medium_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_pink_medium_RAL_iv_day*60/non_nan_counts_wbgt_iv_day  # 40 minutes every hour

# total rest minutes from all to be exported
wbgt_f_hour_exceed_total_restm_perhour_iv_day_medium_RAL = wbgt_f_hour_exceed_yellow_medium_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_orange_medium_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_red_medium_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_pink_medium_RAL_restm_perhour_iv_day;


# third heavy_RAL for all data and IV both

non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))

#all data
#yellow
wbgt_f_hour_exceed_yellow_heavy_RAL_day = np.sum((wbgt_f_day >= 74.6) & (wbgt_f_day < 77.7), axis=(0, 1))
wbgt_f_hour_exceed_yellow_heavy_RAL_restm_perhour_day = wbgt_f_hour_exceed_yellow_heavy_RAL_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_heavy_RAL_day = np.sum((wbgt_f_day >= 77.7) & (wbgt_f_day < 80.4), axis=(0, 1))
wbgt_f_hour_exceed_orange_heavy_RAL_restm_perhour_day = wbgt_f_hour_exceed_orange_heavy_RAL_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_heavy_RAL_day = np.sum((wbgt_f_day >= 80.4) & (wbgt_f_day < 83.0), axis=(0, 1))
wbgt_f_hour_exceed_red_heavy_RAL_restm_perhour_day = wbgt_f_hour_exceed_red_heavy_RAL_day*45/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_heavy_RAL_day = np.sum(wbgt_f_day >= 83.0, axis=(0, 1))
wbgt_f_hour_exceed_pink_heavy_RAL_restm_perhour_day = wbgt_f_hour_exceed_pink_heavy_RAL_day*60/non_nan_counts_wbgt_day  # 40 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day_heavy_RAL = wbgt_f_hour_exceed_yellow_heavy_RAL_restm_perhour_day + wbgt_f_hour_exceed_orange_heavy_RAL_restm_perhour_day + wbgt_f_hour_exceed_red_heavy_RAL_restm_perhour_day + wbgt_f_hour_exceed_pink_heavy_RAL_restm_perhour_day;


#IV data
non_nan_counts_wbgt_iv_day = np.sum(~np.isnan(wbgt_f_iv_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_heavy_RAL_iv_day = np.sum((wbgt_f_iv_day >= 74.6) & (wbgt_f_iv_day < 77.7), axis=(0, 1))
wbgt_f_hour_exceed_yellow_heavy_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_yellow_heavy_RAL_iv_day*15/non_nan_counts_wbgt_iv_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_heavy_RAL_iv_day = np.sum((wbgt_f_iv_day >= 77.7) & (wbgt_f_iv_day < 80.4), axis=(0, 1))
wbgt_f_hour_exceed_orange_heavy_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_orange_heavy_RAL_iv_day*30/non_nan_counts_wbgt_iv_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_heavy_RAL_iv_day = np.sum((wbgt_f_iv_day >= 80.4) & (wbgt_f_iv_day < 83.0), axis=(0, 1))
wbgt_f_hour_exceed_red_heavy_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_red_heavy_RAL_iv_day*45/non_nan_counts_wbgt_iv_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_heavy_RAL_iv_day = np.sum(wbgt_f_iv_day >= 83.0, axis=(0, 1))
wbgt_f_hour_exceed_pink_heavy_RAL_restm_perhour_iv_day = wbgt_f_hour_exceed_pink_heavy_RAL_iv_day*60/non_nan_counts_wbgt_iv_day  # 40 minutes every hour

# total rest minutes from all above to be exported
wbgt_f_hour_exceed_total_restm_perhour_iv_day_heavy_RAL = wbgt_f_hour_exceed_yellow_heavy_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_orange_heavy_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_red_heavy_RAL_restm_perhour_iv_day + wbgt_f_hour_exceed_pink_heavy_RAL_restm_perhour_iv_day;



# Now REL


# first light_REL for all data and IV both

# all data
non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_light_REL_day = np.sum((wbgt_f_day >= 87.6) & (wbgt_f_day < 89.1), axis=(0, 1))
wbgt_f_hour_exceed_yellow_light_REL_restm_perhour_day = wbgt_f_hour_exceed_yellow_light_REL_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_light_REL_day = np.sum((wbgt_f_day >= 89.1) & (wbgt_f_day < 91.1), axis=(0, 1))
wbgt_f_hour_exceed_orange_light_REL_restm_perhour_day = wbgt_f_hour_exceed_orange_light_REL_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_light_REL_day = np.sum((wbgt_f_day >= 91.1) & (wbgt_f_day < 92.4), axis=(0, 1))
wbgt_f_hour_exceed_red_light_REL_restm_perhour_day = wbgt_f_hour_exceed_red_light_REL_day*45/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_light_REL_day = np.sum(wbgt_f_day >= 92.4, axis=(0, 1))
wbgt_f_hour_exceed_pink_light_REL_restm_perhour_day = wbgt_f_hour_exceed_pink_light_REL_day*60/non_nan_counts_wbgt_day  # 40 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day_light_REL = wbgt_f_hour_exceed_yellow_light_REL_restm_perhour_day + wbgt_f_hour_exceed_orange_light_REL_restm_perhour_day + wbgt_f_hour_exceed_red_light_REL_restm_perhour_day + wbgt_f_hour_exceed_pink_light_REL_restm_perhour_day;

# iv data
non_nan_counts_wbgt_iv_day = np.sum(~np.isnan(wbgt_f_iv_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_light_REL_iv_day = np.sum((wbgt_f_iv_day >= 87.6) & (wbgt_f_iv_day < 89.1), axis=(0, 1))
wbgt_f_hour_exceed_yellow_light_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_yellow_light_REL_iv_day*15/non_nan_counts_wbgt_iv_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_light_REL_iv_day = np.sum((wbgt_f_iv_day >= 89.1) & (wbgt_f_iv_day < 91.1), axis=(0, 1))
wbgt_f_hour_exceed_orange_light_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_orange_light_REL_iv_day*30/non_nan_counts_wbgt_iv_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_light_REL_iv_day = np.sum((wbgt_f_iv_day >= 91.1) & (wbgt_f_iv_day < 92.4), axis=(0, 1))
wbgt_f_hour_exceed_red_light_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_red_light_REL_iv_day*45/non_nan_counts_wbgt_iv_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_light_REL_iv_day = np.sum(wbgt_f_iv_day >= 92.4, axis=(0, 1))
wbgt_f_hour_exceed_pink_light_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_pink_light_REL_iv_day*60/non_nan_counts_wbgt_iv_day  # 40 minutes every hour

# total rest minutes from all above to be exported
wbgt_f_hour_exceed_total_restm_perhour_iv_day_light_REL = wbgt_f_hour_exceed_yellow_light_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_orange_light_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_red_light_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_pink_light_REL_restm_perhour_iv_day;



# second medium_REL for all data and IV both

#all data
non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_medium_REL_day = np.sum((wbgt_f_day >= 83.1) & (wbgt_f_day < 84.9), axis=(0, 1))
wbgt_f_hour_exceed_yellow_medium_REL_restm_perhour_day = wbgt_f_hour_exceed_yellow_medium_REL_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_medium_REL_day = np.sum((wbgt_f_day >= 84.9) & (wbgt_f_day < 86.7), axis=(0, 1))
wbgt_f_hour_exceed_orange_medium_REL_restm_perhour_day = wbgt_f_hour_exceed_orange_medium_REL_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_medium_REL_day = np.sum((wbgt_f_day >= 86.7) & (wbgt_f_day < 88.7), axis=(0, 1))
wbgt_f_hour_exceed_red_medium_REL_restm_perhour_day = wbgt_f_hour_exceed_red_medium_REL_day*45/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_medium_REL_day = np.sum(wbgt_f_day >= 88.7, axis=(0, 1))
wbgt_f_hour_exceed_pink_medium_REL_restm_perhour_day = wbgt_f_hour_exceed_pink_medium_REL_day*60/non_nan_counts_wbgt_day  # 40 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day_medium_REL = wbgt_f_hour_exceed_yellow_medium_REL_restm_perhour_day + wbgt_f_hour_exceed_orange_medium_REL_restm_perhour_day + wbgt_f_hour_exceed_red_medium_REL_restm_perhour_day + wbgt_f_hour_exceed_pink_medium_REL_restm_perhour_day;


#IV data
non_nan_counts_wbgt_iv_day = np.sum(~np.isnan(wbgt_f_iv_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_medium_REL_iv_day = np.sum((wbgt_f_iv_day >= 83.1) & (wbgt_f_iv_day < 84.9), axis=(0, 1))
wbgt_f_hour_exceed_yellow_medium_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_yellow_medium_REL_iv_day*15/non_nan_counts_wbgt_iv_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_medium_REL_iv_day = np.sum((wbgt_f_iv_day >= 84.9) & (wbgt_f_iv_day < 86.7), axis=(0, 1))
wbgt_f_hour_exceed_orange_medium_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_orange_medium_REL_iv_day*30/non_nan_counts_wbgt_iv_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_medium_REL_iv_day = np.sum((wbgt_f_iv_day >= 86.7) & (wbgt_f_iv_day < 88.7), axis=(0, 1))
wbgt_f_hour_exceed_red_medium_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_red_medium_REL_iv_day*45/non_nan_counts_wbgt_iv_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_medium_REL_iv_day = np.sum(wbgt_f_iv_day >= 88.7, axis=(0, 1))
wbgt_f_hour_exceed_pink_medium_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_pink_medium_REL_iv_day*60/non_nan_counts_wbgt_iv_day  # 40 minutes every hour

# total rest minutes from all above to be exported
wbgt_f_hour_exceed_total_restm_perhour_iv_day_medium_REL = wbgt_f_hour_exceed_yellow_medium_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_orange_medium_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_red_medium_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_pink_medium_REL_restm_perhour_iv_day;



# third heavy_REL for all data and IV both
#all data
non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_heavy_REL_day = np.sum((wbgt_f_day >= 79.9) & (wbgt_f_day < 82.0), axis=(0, 1))
wbgt_f_hour_exceed_yellow_heavy_REL_restm_perhour_day = wbgt_f_hour_exceed_yellow_heavy_REL_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_heavy_REL_day = np.sum((wbgt_f_day >= 82.0) & (wbgt_f_day < 83.8), axis=(0, 1))
wbgt_f_hour_exceed_orange_heavy_REL_restm_perhour_day = wbgt_f_hour_exceed_orange_heavy_REL_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_heavy_REL_day = np.sum((wbgt_f_day >= 83.8) & (wbgt_f_day < 86.4), axis=(0, 1))
wbgt_f_hour_exceed_red_heavy_REL_restm_perhour_day = wbgt_f_hour_exceed_red_heavy_REL_day*45/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_heavy_REL_day = np.sum(wbgt_f_day >= 86.4, axis=(0, 1))
wbgt_f_hour_exceed_pink_heavy_REL_restm_perhour_day = wbgt_f_hour_exceed_pink_heavy_REL_day*60/non_nan_counts_wbgt_day  # 40 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day_heavy_REL = wbgt_f_hour_exceed_yellow_heavy_REL_restm_perhour_day + wbgt_f_hour_exceed_orange_heavy_REL_restm_perhour_day + wbgt_f_hour_exceed_red_heavy_REL_restm_perhour_day + wbgt_f_hour_exceed_pink_heavy_REL_restm_perhour_day;

#IV data
non_nan_counts_wbgt_iv_day = np.sum(~np.isnan(wbgt_f_iv_day), axis=(0, 1))

#yellow
wbgt_f_hour_exceed_yellow_heavy_REL_iv_day = np.sum((wbgt_f_iv_day >= 79.9) & (wbgt_f_iv_day < 82.0), axis=(0, 1))
wbgt_f_hour_exceed_yellow_heavy_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_yellow_heavy_REL_iv_day*15/non_nan_counts_wbgt_iv_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_heavy_REL_iv_day = np.sum((wbgt_f_iv_day >= 82.0) & (wbgt_f_iv_day < 83.8), axis=(0, 1))
wbgt_f_hour_exceed_orange_heavy_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_orange_heavy_REL_iv_day*30/non_nan_counts_wbgt_iv_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_heavy_REL_iv_day = np.sum((wbgt_f_iv_day >= 83.8) & (wbgt_f_iv_day < 86.4), axis=(0, 1))
wbgt_f_hour_exceed_red_heavy_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_red_heavy_REL_iv_day*45/non_nan_counts_wbgt_iv_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_heavy_REL_iv_day = np.sum(wbgt_f_iv_day >= 86.4, axis=(0, 1))
wbgt_f_hour_exceed_pink_heavy_REL_restm_perhour_iv_day = wbgt_f_hour_exceed_pink_heavy_REL_iv_day*60/non_nan_counts_wbgt_iv_day  # 40 minutes every hour

# total rest minutes from all above to be exported
wbgt_f_hour_exceed_total_restm_perhour_iv_day_heavy_REL = wbgt_f_hour_exceed_yellow_heavy_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_orange_heavy_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_red_heavy_REL_restm_perhour_iv_day + wbgt_f_hour_exceed_pink_heavy_REL_restm_perhour_iv_day;


import numpy as np
import pandas as pd

iv_day_light_RAL = np.column_stack(wbgt_f_hour_exceed_total_restm_perhour_iv_day_light_RAL.flatten())
df_iv_day_light_RAL = pd.DataFrame(iv_day_light_RAL)
df_iv_day_light_RAL.to_csv('iv_day_light_RAL.csv', index=False)

iv_day_medium_RAL = np.column_stack(wbgt_f_hour_exceed_total_restm_perhour_iv_day_medium_RAL.flatten())
df_iv_day_medium_RAL = pd.DataFrame(iv_day_medium_RAL)
df_iv_day_medium_RAL.to_csv('iv_day_medium_RAL.csv', index=False)

iv_day_heavy_RAL = np.column_stack(wbgt_f_hour_exceed_total_restm_perhour_iv_day_heavy_RAL.flatten())
df_iv_day_heavy_RAL = pd.DataFrame(iv_day_heavy_RAL)
df_iv_day_heavy_RAL.to_csv('iv_day_heavy_RAL.csv', index=False)


iv_day_light_REL = np.column_stack(wbgt_f_hour_exceed_total_restm_perhour_iv_day_light_REL.flatten())
df_iv_day_light_REL = pd.DataFrame(iv_day_light_REL)
df_iv_day_light_REL.to_csv('iv_day_light_REL.csv', index=False)

iv_day_medium_REL = np.column_stack(wbgt_f_hour_exceed_total_restm_perhour_iv_day_medium_REL.flatten())
df_iv_day_medium_REL = pd.DataFrame(iv_day_medium_REL)
df_iv_day_medium_REL.to_csv('iv_day_medium_REL.csv', index=False)

iv_day_heavy_REL = np.column_stack(wbgt_f_hour_exceed_total_restm_perhour_iv_day_heavy_REL.flatten())
df_iv_day_heavy_REL = pd.DataFrame(iv_day_heavy_REL)
df_iv_day_heavy_REL.to_csv('iv_day_heavy_REL.csv', index=False)




# spatial plot of NOISH thresholds for unacclimatized and acclimatized workers for different work loads

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.rcParams.update({'font.size': 12})

levels = np.arange(0, 51, 10)  # used for colorbar tick ylabel

fig, ((ax_1, ax_2), (ax_3, ax_4), (ax_5, ax_6)) = plt.subplots(
    3, 2, figsize=(14, 20),
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1]}
)
plt.subplots_adjust(wspace=0.1, hspace=0.15)  # Adjust the width space between subplots

# First subplot - grid labels on the left and top side only
ax_1.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_1 = ax_1.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day_light_RAL, cmap='YlOrRd', vmin=0, vmax=50)
counties.plot(ax=ax_1, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_1, linewidth=1.5, color='black', facecolor='skyblue')
gl_1 = ax_1.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_1.right_labels = False
gl_1.bottom_labels = False
gl_1.top_labels = False
ax_1.set_title(r'(a) $RAL_{\mathrm{light}}$')
ax_1.text(0.5, 0.9, '90th IV = 21.3 minutes', ha='center', va='top', transform=ax_1.transAxes, fontsize=10)

# Second subplot - grid labels on the top only
ax_2.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_2 = ax_2.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day_light_REL, cmap='YlOrRd', vmin=0, vmax=50)
counties.plot(ax=ax_2, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_2, linewidth=1.5, color='black', facecolor='skyblue')
gl_2 = ax_2.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_2.left_labels = False
gl_2.right_labels = False
gl_2.bottom_labels = False
gl_2.top_labels = False
ax_2.text(0.5, 0.9, '90th IV = 14.4 minutes', ha='center', va='top', transform=ax_2.transAxes, fontsize=10)

ax_2.set_title(r'(b) $REL_{\mathrm{light}}$')

# Third subplot - grid labels on the left only
ax_3.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_3 = ax_3.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day_medium_RAL, cmap='YlOrRd', vmin=0, vmax=50)
counties.plot(ax=ax_3, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_3, linewidth=1.5, color='black', facecolor='skyblue')
gl_3 = ax_3.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_3.top_labels = False
gl_3.right_labels = False
gl_3.bottom_labels = False
ax_3.set_title(r'(c) $RAL_{\mathrm{medium}}$', fontsize=14)
ax_3.text(0.5, 0.9, '90th IV = 32.9 minutes', ha='center', va='top', transform=ax_3.transAxes, fontsize=10)

# Fourth subplot - no grid labels
ax_4.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_4 = ax_4.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day_medium_REL, cmap='YlOrRd', vmin=0, vmax=50)
counties.plot(ax=ax_4, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_4, linewidth=1.5, color='black', facecolor='skyblue')
gl_4 = ax_4.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_4.left_labels = False
gl_4.right_labels = False
gl_4.top_labels = False
gl_4.bottom_labels = False
ax_4.set_title(r'(d) $REL_{\mathrm{medium}}$', fontsize=14)
ax_4.text(0.5, 0.9, '90th IV = 24.9 minutes', ha='center', va='top', transform=ax_4.transAxes, fontsize=10)

# Fifth subplot - grid labels on the left and bottom side only
ax_5.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_5 = ax_5.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day_heavy_RAL, cmap='YlOrRd', vmin=0, vmax=50)
counties.plot(ax=ax_5, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_5, linewidth=1.5, color='black', facecolor='skyblue')
gl_5 = ax_5.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_5.top_labels = False
gl_5.right_labels = False
ax_5.set_title(r'(e) $RAL_{\mathrm{heavy}}$')
ax_5.text(0.5, 0.9, '90th IV = 39.8 minutes', ha='center', va='top', transform=ax_5.transAxes, fontsize=10)

# Sixth subplot - grid labels on the bottom side only
ax_6.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_6 = ax_6.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day_heavy_REL, cmap='YlOrRd', vmin=0, vmax=50)
counties.plot(ax=ax_6, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_6, linewidth=1.5, color='black', facecolor='skyblue')
gl_6 = ax_6.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_6.left_labels = False
gl_6.right_labels = False
gl_6.top_labels = False
ax_6.set_title(r'(f) $REL_{\mathrm{heavy}}$')
ax_6.text(0.5, 0.9, '90th IV = 31.6 minutes', ha='center', va='top', transform=ax_6.transAxes, fontsize=10)


# Add a single colorbar for all subplots
cbar = fig.colorbar(mesh_1, ax=[ax_1, ax_2, ax_3, ax_4, ax_5, ax_6], orientation='vertical', shrink=0.5, pad=0.06, ticks=levels)
cbar.set_label('Rest-minutes per hour', fontsize=16, labelpad=10)  # Change 14 to your desired label font size


plt.savefig('spatial_rest_minutes_NIOSH.png', bbox_inches='tight', dpi=300)
