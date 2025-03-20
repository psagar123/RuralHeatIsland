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
rh_td_ds.close()

rh_20_aug_sensor = np.clip(rh_20_aug_sensor, 0, 100)

# Calculate wind speed magnitude
ws_20_aug_sensor = np.sqrt(u_20_aug_sensor**2 + v_20_aug_sensor**2); del u_20_aug_sensor, v_20_aug_sensor;

print(ws_20_aug_sensor.shape)


#import required Liljegren's functions

from WBGT import Tg_Liljegren, Tnwb_Liljegren
import thermofeel; 

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
wbgt_f[wbgt_f < 0] = np.nan;
print('min wbgt_f', np.nanmin(wbgt_f.flatten()))
print('max wbgt_f', np.nanmax(wbgt_f.flatten()))

t2_f = kelvin_to_fahrenheit(t2_k);


import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import MultipleLocator

# Load the county and lake shapefiles
counties = gpd.read_file("ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
salton = gpd.read_file("Salton_geog/Salton_geog.shp")


# Calculate Heat Index

# extract t2 and rh to calculate HI for IV

t2_f_iv = t2_f[:, 20:70, 106:140]

rh_pct = rh_20_aug_sensor;
rh_pct_iv = rh_pct[:, 20:70, 106:140];


# calculate HI using metpy, using the NWS HI equation https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

# MetPy: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.heat_index.html

from metpy.calc import heat_index
from metpy.units import units



# extract data corresponding to workshifts for DBT, WBGT, and HI

# first DBT
t2_f_diurnal = np.reshape(t2_f, (24, 214, 171, 162), order='F');
t2_f_day = t2_f_diurnal[6:14, :, :, :]; # 6am to 2pm

# second WBGT
wbgt_f_diurnal = np.reshape(wbgt_f, (24, 214, 171, 162), order='F');
wbgt_f_day = wbgt_f_diurnal[6:14, :, :, :];


# Calculate Heat Index

# calculate HI using metpy, using the NWS HI equation https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

# MetPy: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.heat_index.html


# third HI
HI_f_all = heat_index(t2_f*units.degF, rh_pct*units.percent) # heat index in F entire domain
HI_f_all_diurnal = np.reshape(HI_f_all.magnitude, (24, 214, 171, 162), order='F');
HI_f_all_day = HI_f_all_diurnal[6:14, :, :, :];


# calculate for IV region

t2_f_iv = t2_f[:, 20:70, 106:140]
t2_f_iv_diurnal = np.reshape(t2_f_iv, (24, 214, 50, 34), order='F');
t2_f_iv_day = t2_f_iv_diurnal[6:14, :, :, :];

rh_pct_iv = rh_pct[:, 20:70, 106:140];


HI_f_iv = heat_index(t2_f_iv*units.degF, rh_pct_iv*units.percent) # heat index in F
# HI_f_iv.magnitude[HI_f_iv.magnitude < 20] = np.nan


print('HI IV min:', np.nanmin(HI_f_iv.magnitude.flatten()))
print('HI IV max:', np.nanmax(HI_f_iv.magnitude.flatten()))

#HI_f_iv_clean = HI_f_iv.magnitude[np.isfinite(HI_f_iv.magnitude)]



HI_f_iv_diurnal = np.reshape(HI_f_iv.magnitude, (24, 214, 50, 34), order='F');
HI_f_iv_day = HI_f_iv_diurnal[6:14, :, :, :];

# extract wbgt_f for IV
wbgt_f_iv = wbgt_f[:, 20:70, 106:140];
wbgt_f_iv_diurnal = np.reshape(wbgt_f_iv, (24, 214, 50, 34), order='F');
wbgt_f_iv_day = wbgt_f_iv_diurnal[6:14, :, :, :]; del wbgt_f_iv_diurnal;


# extract wbgt_f for IV
wbgt_f_iv = wbgt_f[:, 20:70, 106:140];

# review the data range
print('wbgt_min', np.nanmin(wbgt_f_iv.flatten()))
print('wbgt_max', np.nanmax(wbgt_f_iv.flatten()))

print('t2_min', np.nanmin(t2_f_iv.flatten()))
print('t2_max', np.nanmax(t2_f_iv.flatten()))

print('HI min', np.nanmin(HI_f_iv.magnitude.flatten()))
print('HI max', np.nanmax(HI_f_iv.magnitude.flatten()))

print(wbgt_f_iv.shape)


# plot histograms for data for data within IV bounding box for all data as well as work shift data

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
plt.subplots_adjust(wspace=0.3, hspace=0.42)  # Adjust the width space between subplots

common_range = (20, 125)

axes[0,0].hist(wbgt_f_iv.flatten(), bins=20, histtype='step', linewidth=1.5, color='grey')
axes[0,0].set_title(r'$WBGT_{\mathrm{all\_hours}}$', fontsize=12)
axes[0,0].set_ylim(0, 1.5E6)
axes[0,0].set_xlabel('(\u00b0F)', fontsize=12)
axes[0,0].axvspan(80, 85, color='#ffff00')
axes[0,0].axvspan(85, 88, color='#ff9900')
axes[0,0].axvspan(88, 90, color='#ff0000')
axes[0,0].axvspan(90, 140.00, color='#ff00ff')
axes[0,0].set_xlim(20, 125)

# Correct the axis indexing
axes[0,1].hist(t2_f_iv.flatten(), bins=20, histtype='step', linewidth=1.5, color='grey')
axes[0,1].set_title(r'$DBT_{\mathrm{all\_hours}}$', fontsize=12)
axes[0,1].set_ylim(0, 1.5E6)
axes[0,1].set_xlabel('(\u00b0F)', fontsize=12)
axes[0,1].axvline(x=95, color='yellow', linestyle='-', linewidth=1.5)
axes[0,1].axvspan(95, 140.00, color='#ffff00')
axes[0,1].set_xlim(20, 125)


axes[0,2].hist(HI_f_iv.flatten(), bins=20, histtype='step', linewidth=1.5, color='grey')
axes[0,2].set_title(r'$HI_{\mathrm{all\_hours}}$', fontsize=12)
axes[0,2].set_ylim(0, 1.5E6)
axes[0,2].set_xlabel('(\u00b0F)', fontsize=12)
axes[0,2].axvline(x=80, color='yellow', linestyle='-', linewidth=1.5)
axes[0,2].axvspan(80, 140.00, color='#ffff00')
axes[0,2].set_xlim(20, 125)


# First subplot
axes[1,0].hist(wbgt_f_iv_day.flatten(), bins=20, histtype='step', linewidth=1.5, color='grey')
axes[1,0].set_title(r'$WBGT_{\mathrm{work\_hours}}$', fontsize=12)
axes[1,0].set_ylim(0, 1.5E6)
axes[1,0].set_xlabel('(\u00b0F)', fontsize=12)
axes[1,0].axvspan(80, 85, color='#ffff00')
axes[1,0].axvspan(85, 88, color='#ff9900')
axes[1,0].axvspan(88, 90, color='#ff0000')
axes[1,0].axvspan(90, 140.00, color='#ff00ff')
axes[1,0].set_xlim(20, 125)

# Second subplot
axes[1,1].hist(t2_f_iv_day.flatten(), bins=20, histtype='step', linewidth=1.5, color='grey')
axes[1,1].set_title(r'$DBT_{\mathrm{work\_hours}}$', fontsize=12)
axes[1,1].set_ylim(0, 1.5E6)
axes[1,1].set_xlabel('(\u00b0F)', fontsize=12)
axes[1,1].axvline(x=95, color='yellow', linestyle='-', linewidth=1.5)
axes[1,1].axvspan(95, 140.00, color='#ffff00')
axes[1,1].set_xlim(20, 125)

# Third subplot
axes[1,2].hist(HI_f_iv_day.flatten(), bins=20, histtype='step', linewidth=1.5, color='grey')
axes[1,2].set_title(r'$HI_{\mathrm{work\_hours}}$', fontsize=12)
axes[1,2].set_ylim(0, 1.5E6)
axes[1,2].set_xlabel('(\u00b0F)', fontsize=12)
axes[1,2].axvline(x=80, color='yellow', linestyle='-', linewidth=1.5)
axes[1,2].axvspan(80, 140.00, color='#ffff00')
axes[1,2].set_xlim(20, 125)



plt.savefig('histogram_hi_dbt_wbgt.png', bbox_inches='tight', dpi=400)


# calculate exceedance grid point hours per day for time series plot

import numpy as np

wbgt_f_iv_diurnal = np.reshape(wbgt_f_iv, (24, 214, 50, 34), order='F');
gt_80_wbgt_iv_all = np.sum(wbgt_f_iv_diurnal > 80, axis=(0, 2, 3))

HI_f_iv_diurnal = np.reshape(HI_f_iv.magnitude, (24, 214, 50, 34), order='F');
gt_80_HI_iv_all = np.sum(HI_f_iv_diurnal > 80, axis=(0, 2, 3))

t2_f_iv_diurnal = np.reshape(t2_f_iv, (24, 214, 50, 34), order='F');
gt_95_t2_iv_all = np.sum(t2_f_iv_diurnal > 95, axis=(0, 2, 3))



# Calculate number of hours exceeding a given threshold using all data
#note only along time dimension calculations are done

# using WBGT
non_nan_counts_wbgt = np.sum(~np.isnan(wbgt_f), axis=0)


#yellow
wbgt_f_hour_exceed_yellow = np.sum((wbgt_f >= 80) & (wbgt_f < 85), axis=0);
wbgt_f_hour_exceed_yellow_restm_perhour = wbgt_f_hour_exceed_yellow*15/non_nan_counts_wbgt  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange = np.sum((wbgt_f >= 85) & (wbgt_f < 88), axis=0)
wbgt_f_hour_exceed_orange_restm_perhour = wbgt_f_hour_exceed_orange*30/non_nan_counts_wbgt  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red = np.sum((wbgt_f >= 88) & (wbgt_f < 90), axis=0)
wbgt_f_hour_exceed_red_restm_perhour = wbgt_f_hour_exceed_red*40/non_nan_counts_wbgt  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink = np.sum(wbgt_f >= 90, axis=0); del wbgt_f;
wbgt_f_hour_exceed_pink_restm_perhour = wbgt_f_hour_exceed_pink*45/non_nan_counts_wbgt  # 45 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour = wbgt_f_hour_exceed_yellow_restm_perhour + wbgt_f_hour_exceed_orange_restm_perhour + wbgt_f_hour_exceed_red_restm_perhour + wbgt_f_hour_exceed_pink_restm_perhour;

# using DBT
non_nan_counts_t2 = np.sum(~np.isnan(t2_f), axis=0)
t2_f_hour_exceed_95 = np.sum(t2_f > 95, axis=0); del t2_f;
t2_f_hour_exceed_95_restm_perhour = t2_f_hour_exceed_95*5/non_nan_counts_t2  # 5 minutes every hour

# using HI
non_nan_counts_HI = np.sum(~np.isnan(HI_f_all.magnitude), axis=0)
HI_f_hour_exceed_80 = np.sum(HI_f_all.magnitude > 80, axis=0); del HI_f_all;
HI_f_hour_exceed_80_restm_perhour = HI_f_hour_exceed_80*7.5/non_nan_counts_HI  # 7.5 minutes every hour


# average time-series over the IV region
t2_f_iv_av = np.nanmean(np.nanmean(t2_f_iv, axis=2), axis=1)
wbgt_f_iv_av = np.nanmean(np.nanmean(wbgt_f_iv, axis=2), axis=1)
HI_f_iv_av = np.nanmean(np.nanmean(HI_f_iv.magnitude, axis=2), axis=1)


# calculate exceedance percentages for WBGT, HI, and DBT for IV data all hours


# WBGT
#yellow
gt_yellow_wbgt_all = np.sum((wbgt_f_iv.flatten() >= 80) & (wbgt_f_iv.flatten() < 85))
total_wbgt_all = np.count_nonzero(~np.isnan(wbgt_f_iv.flatten()))
percentage_wbgt_all_yellow = (gt_yellow_wbgt_all / total_wbgt_all) * 100 # percentage samples gt 80
#orange
gt_orange_wbgt_all = np.sum((wbgt_f_iv.flatten() >= 85) & (wbgt_f_iv.flatten() < 88))
percentage_wbgt_all_orange = (gt_orange_wbgt_all / total_wbgt_all) * 100 # percentage
#red
gt_red_wbgt_all = np.sum((wbgt_f_iv.flatten() >= 88) & (wbgt_f_iv.flatten() < 90))
percentage_wbgt_all_red = (gt_red_wbgt_all / total_wbgt_all) * 100 # percentage
# pink
gt_pink_wbgt_all = np.sum(wbgt_f_iv.flatten() >= 90)
percentage_wbgt_all_pink = (gt_pink_wbgt_all / total_wbgt_all) * 100 # percentage samples gt 80

percentage_wbgt_all = (percentage_wbgt_all_yellow + percentage_wbgt_all_orange + percentage_wbgt_all_red + percentage_wbgt_all_pink);

# DBT
gt_95_t2_all = np.sum(t2_f_iv.flatten() > 95)
total_t2_all = np.count_nonzero(~np.isnan(t2_f_iv.flatten()))
percentage_t2_all = (gt_95_t2_all / total_t2_all) * 100

# HI

gt_80_HI_all = np.sum(HI_f_iv.magnitude.flatten() > 80)
total_HI_all = np.count_nonzero(~np.isnan(HI_f_iv.magnitude.flatten()))
percentage_HI_all = (gt_80_HI_all / total_HI_all) * 100

print('% WBGT exceedance IV all hours', percentage_wbgt_all_yellow)
print('% WBGT exceedance IV all hours', percentage_wbgt_all_orange)
print('% WBGT exceedance IV all hours', percentage_wbgt_all_red)
print('% WBGT exceedance IV all hours', percentage_wbgt_all_pink)
print('% WBGT exceedance total WBGT IV all hours', percentage_wbgt_all)

print('% DBT exceedance IV all hours', percentage_t2_all)
print('% HI exceedance IV all hours', percentage_HI_all)


# calculate exceedance percentages for WBGT, HI, and DBT for IV data day only

import numpy as np

# WBGT
#yellow
gt_yellow_wbgt = np.sum((wbgt_f_iv_day.flatten() >= 80) & (wbgt_f_iv_day.flatten() < 85))
total_wbgt = np.count_nonzero(~np.isnan(wbgt_f_iv_day.flatten()))
percentage_wbgt_yellow = (gt_yellow_wbgt / total_wbgt) * 100 # percentage samples gt 80
#orange
gt_orange_wbgt = np.sum((wbgt_f_iv_day.flatten() >= 85) & (wbgt_f_iv_day.flatten() < 88))
percentage_wbgt_orange = (gt_orange_wbgt / total_wbgt) * 100 # percentage
#red
gt_red_wbgt = np.sum((wbgt_f_iv_day.flatten() >= 88) & (wbgt_f_iv_day.flatten() < 90))
percentage_wbgt_red = (gt_red_wbgt / total_wbgt) * 100 # percentage
# pink
gt_pink_wbgt = np.sum(wbgt_f_iv_day.flatten() >= 90)
percentage_wbgt_pink = (gt_pink_wbgt / total_wbgt) * 100 # percentage samples gt 80

percentage_wbgt = (percentage_wbgt_yellow + percentage_wbgt_orange + percentage_wbgt_red + percentage_wbgt_pink);

# DBT
gt_95_t2 = np.sum(t2_f_iv_day.flatten() > 95)
total_t2 = np.count_nonzero(~np.isnan(t2_f_iv_day.flatten()))
percentage_t2 = (gt_95_t2 / total_t2) * 100

# HI
gt_80_HI = np.sum(HI_f_iv_day.flatten() > 80)
total_HI = np.count_nonzero(~np.isnan(HI_f_iv_day.flatten()))
percentage_HI = (gt_80_HI / total_HI) * 100

print('% WBGT exceedance IV 6am-2pm', percentage_wbgt_yellow)
print('% WBGT exceedance IV 6am-2pm', percentage_wbgt_orange)
print('% WBGT exceedance IV 6am-2pm', percentage_wbgt_red)
print('% WBGT exceedance IV 6am-2pm', percentage_wbgt_pink)
print('% WBGT exceedance total WBGT IV 6am-2pm', percentage_wbgt)

print('% DBT exceedance IV 6am-2pm', percentage_t2)
print('% HI exceedance IV 6am-2pm', percentage_HI)






# Calculate number of hours exceeding a given threshold using workshift data for spatial plot


# using WBGT
non_nan_counts_wbgt_day = np.sum(~np.isnan(wbgt_f_day), axis=(0, 1))


#yellow
wbgt_f_hour_exceed_yellow_day = np.sum((wbgt_f_day >= 80) & (wbgt_f_day < 85), axis=(0, 1))
wbgt_f_hour_exceed_yellow_restm_perhour_day = wbgt_f_hour_exceed_yellow_day*15/non_nan_counts_wbgt_day  # 15 minutes every hour
#orange
wbgt_f_hour_exceed_orange_day = np.sum((wbgt_f_day >= 85) & (wbgt_f_day < 88), axis=(0, 1))
wbgt_f_hour_exceed_orange_restm_perhour_day = wbgt_f_hour_exceed_orange_day*30/non_nan_counts_wbgt_day  # 30 minutes every hour
#red
wbgt_f_hour_exceed_red_day = np.sum((wbgt_f_day >= 88) & (wbgt_f_day < 90), axis=(0, 1))
wbgt_f_hour_exceed_red_restm_perhour_day = wbgt_f_hour_exceed_red_day*40/non_nan_counts_wbgt_day  # 40 minutes every hour
#pink
wbgt_f_hour_exceed_pink_day = np.sum(wbgt_f_day >= 90, axis=(0, 1))
wbgt_f_hour_exceed_pink_restm_perhour_day = wbgt_f_hour_exceed_pink_day*45/non_nan_counts_wbgt_day  # 45 minutes every hour

# total rest minutes from all above to be plotted
wbgt_f_hour_exceed_total_restm_perhour_day = wbgt_f_hour_exceed_yellow_restm_perhour_day + wbgt_f_hour_exceed_orange_restm_perhour_day + wbgt_f_hour_exceed_red_restm_perhour_day + wbgt_f_hour_exceed_pink_restm_perhour_day;

# using DBT
non_nan_counts_t2_day = np.sum(~np.isnan(t2_f_day), axis=(0, 1))
t2_f_hour_exceed_95_day = np.sum(t2_f_day > 95, axis=(0, 1))
t2_f_hour_exceed_95_restm_perhour_day = t2_f_hour_exceed_95_day*5/non_nan_counts_t2_day  # 5 minutes every hour

# using HI
non_nan_counts_HI_day = np.sum(~np.isnan(HI_f_all_day), axis=(0, 1))
HI_f_hour_exceed_80_day = np.sum(HI_f_all_day > 80, axis=(0, 1))
HI_f_hour_exceed_80_restm_perhour_day = HI_f_hour_exceed_80_day*7.5/non_nan_counts_HI_day  # 7.5 minutes every hour


non_nan_counts_wbgt_day

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.rcParams.update({'font.size': 12})

levels = np.arange(0, 16, 3)  # used for colorbar tick ylabel

fig, ((ax_1, ax_2), (ax_3, ax_4), (ax_5, ax_6)) = plt.subplots(
    3, 2, figsize=(14, 20),
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1]}
)
plt.subplots_adjust(wspace=0.1, hspace=0.15)  # Adjust the width space between subplots

# First subplot - grid labels on the left and top side only
ax_1.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_1 = ax_1.pcolormesh(lon, lat, wbgt_f_hour_exceed_yellow_restm_perhour, cmap='YlOrRd', vmin=0, vmax=15)
counties.plot(ax=ax_1, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_1, linewidth=1.5, color='black', facecolor='skyblue')
gl_1 = ax_1.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_1.right_labels = False
gl_1.bottom_labels = False
gl_1.top_labels = False

ax_1.set_title(r'(a) $WBGT_{\mathrm{all\_hours}}$')

# Second subplot - grid labels on the top only
ax_2.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_2 = ax_2.pcolormesh(lon, lat, wbgt_f_hour_exceed_total_restm_perhour_day, cmap='YlOrRd', vmin=0, vmax=15)
counties.plot(ax=ax_2, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_2, linewidth=1.5, color='black', facecolor='skyblue')
gl_2 = ax_2.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_2.left_labels = False
gl_2.right_labels = False
gl_2.bottom_labels = False
gl_2.top_labels = False

ax_2.set_title(r'(b) $WBGT_{\mathrm{work\_hours}}$')

# Third subplot - grid labels on the left only
ax_3.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_3 = ax_3.pcolormesh(lon, lat, t2_f_hour_exceed_95_restm_perhour, cmap='YlOrRd', vmin=0, vmax=15)
counties.plot(ax=ax_3, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_3, linewidth=1.5, color='black', facecolor='skyblue')
gl_3 = ax_3.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_3.top_labels = False
gl_3.right_labels = False
gl_3.bottom_labels = False
ax_3.set_title(r'(c) $DBT_{\mathrm{all\_hours}}$', fontsize=14)

# Fourth subplot - no grid labels
ax_4.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_4 = ax_4.pcolormesh(lon, lat, t2_f_hour_exceed_95_restm_perhour_day, cmap='YlOrRd', vmin=0, vmax=15)
counties.plot(ax=ax_4, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_4, linewidth=1.5, color='black', facecolor='skyblue')
gl_4 = ax_4.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_4.left_labels = False
gl_4.right_labels = False
gl_4.top_labels = False
gl_4.bottom_labels = False
ax_4.set_title(r'(d) $DBT_{\mathrm{work\_hours}}$', fontsize=14)

# Fifth subplot - grid labels on the left and bottom side only
ax_5.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_5 = ax_5.pcolormesh(lon, lat, HI_f_hour_exceed_80_restm_perhour, cmap='YlOrRd', vmin=0, vmax=15)
counties.plot(ax=ax_5, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_5, linewidth=1.5, color='black', facecolor='skyblue')
gl_5 = ax_5.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_5.top_labels = False
gl_5.right_labels = False

ax_5.set_title(r'(e) $HI_{\mathrm{all\_hours}}$')

# Sixth subplot - grid labels on the bottom side only
ax_6.set_extent([-116.7, -115.1, 32.55, 34.05])
mesh_6 = ax_6.pcolormesh(lon, lat, HI_f_hour_exceed_80_restm_perhour_day, cmap='YlOrRd', vmin=0, vmax=15)
counties.plot(ax=ax_6, edgecolor='k', linewidth=0.5, facecolor='none')
salton.plot(ax=ax_6, linewidth=1.5, color='black', facecolor='skyblue')
gl_6 = ax_6.gridlines(xlocs=np.arange(-116.7, -115.1, 0.4), ylocs=np.arange(32.55, 34.05, 0.4), linewidth=0.5, draw_labels=True)
gl_6.left_labels = False
gl_6.right_labels = False
gl_6.top_labels = False

ax_6.set_title(r'(f) $HI_{\mathrm{work\_hours}}$')

# Add a single colorbar for all subplots
cbar = fig.colorbar(mesh_1, ax=[ax_1, ax_2, ax_3, ax_4, ax_5, ax_6], orientation='vertical', shrink=0.5, pad=0.06, ticks=levels)
cbar.set_label('Rest-minutes per hour', fontsize=16, labelpad=10)  # Change 14 to your desired label font size


plt.savefig('spatial_rest_minutes_hi_dbt_wbgt.png', bbox_inches='tight', dpi=300)


np.nanmean(HI_f_hour_exceed_80_restm_perhour.flatten())

# plot timeseries

t2_f_iv_av_diurnal = np.reshape(t2_f_iv_av, (24, 214), order='F');
t2_f_iv_av_diurnal_max = np.max(t2_f_iv_av_diurnal, axis=0);

wbgt_f_iv_av_diurnal = np.reshape(wbgt_f_iv_av, (24, 214), order='F');
wbgt_f_iv_av_diurnal_max = np.max(wbgt_f_iv_av_diurnal, axis=0);

HI_f_iv_av_diurnal = np.reshape(HI_f_iv_av, (24, 214), order='F');
HI_f_iv_av_diurnal_max = np.max(HI_f_iv_av_diurnal, axis=0);



# plot comparision figures for all stations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})

#start_date = '2020-04-01'
#end_date = '2020-10-31'

num_steps = 214
start_date = np.datetime64('2020-04-01')
end_date = np.datetime64('2020-10-31')
step_size = np.timedelta64(1, 'D')

dates = np.arange(start_date, end_date + step_size, step_size)


fig, ax1 = plt.subplots(1, 1, figsize=(16, 4))
plt.subplots_adjust(hspace=0.5)


ax1.plot(dates[:], np.ma.masked_invalid(t2_f_iv_av_diurnal_max), label='DBT', marker='', linestyle='-', color='blue')
ax1.plot(dates[:], np.ma.masked_invalid(wbgt_f_iv_av_diurnal_max), label='WBGT', marker='', linestyle='-', color='red')
ax1.plot(dates[:], np.ma.masked_invalid(HI_f_iv_av_diurnal_max), label='HI', marker='', linestyle='-', color='#8B008B')

ax1.axhline(y=95, color='blue', linestyle='--', label='DBT threshold (95\u00b0F)', linewidth=0.8)
ax1.axhline(y=80, color='red', linestyle='--', label='WBGT/HI threshold (80\u00b0F)', linewidth=0.8)

ax1.set_title(r'Daily maximum heat indices averaged within IV crop fields')
ax1.set_xlabel('Date')
ax1.set_xlim([start_date, end_date])
ax1.tick_params(axis='x', labelrotation=0)
ax1.set_ylabel('(\u00b0F)')
#ax1.tick_params(axis='both', labelsize=8)
ax1.grid(True)
ax1.legend(loc='lower center', fontsize=8)

plt.savefig('timeseries_hi_dbt_wbgt.png', bbox_inches='tight', dpi=400)


# plot no. of exceeding hours per day

# plot timeseries

# plot comparision figures for all stations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

# Select rows between April 1 and April 30, 2020
start_date = '2020-04-01'
end_date = '2020-10-31'

num_steps = 214
start_date = np.datetime64('2020-04-01')
end_date = np.datetime64('2020-10-31')
step_size = np.timedelta64(1, 'D')

dates = np.arange(start_date, end_date + step_size, step_size)


fig, ax1 = plt.subplots(1, 1, figsize=(16, 4))
plt.subplots_adjust(hspace=0.5)


ax1.plot(dates[:], gt_95_t2_iv_all, label='DBT > 95\u00b0F', marker='', linestyle='-', color='blue')
ax1.plot(dates[:], gt_80_wbgt_iv_all, label='WBGT >80 \u00b0F', marker='', linestyle='-', color='red')
ax1.plot(dates[:], gt_80_HI_iv_all, label='HI > 80\u00b0F', marker='', linestyle='-', color='#8B008B')

ax1.set_title('Total no. of grid point-hours exceeding heat index thresholds within IV Crop fields')
ax1.set_xlabel('Date')
ax1.set_xlim([start_date, end_date])
ax1.tick_params(axis='x', labelrotation=0)
ax1.set_ylabel('Grid point-hours')
ax1.grid(True)
ax1.legend(loc='upper left', fontsize=12)

plt.savefig('timeseries_exceeding_hours_daily_iv.png', bbox_inches='tight', dpi=400)


# plot humidity time series for humidity average daily maximum

# plot timeseries
rh_iv_av = np.nanmean(np.nanmean(rh_pct_iv, axis=2), axis=1)
rh_iv_av_diurnal = np.reshape(rh_iv_av, (24, 214), order='F');
rh_iv_av_diurnal_max = np.max(rh_iv_av_diurnal, axis=0);


# plot no. of exceeding hours per day

# plot timeseries

# plot comparision figures for all stations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})

num_steps = 214
start_date = np.datetime64('2020-04-01')
end_date = np.datetime64('2020-10-31')
step_size = np.timedelta64(1, 'D')

dates = np.arange(start_date, end_date + step_size, step_size)


fig, ax1 = plt.subplots(1, 1, figsize=(16, 4))
plt.subplots_adjust(hspace=0.5)


ax1.plot(dates[:], np.ma.masked_invalid(rh_iv_av_diurnal_max), marker='', linestyle='-', color='blue')

ax1.set_title('Daily maximum relative humidity averaged within IV crop fields')
ax1.set_xlabel('Date')
ax1.set_xlim([start_date, end_date])
ax1.tick_params(axis='x', labelrotation=0)
ax1.set_ylabel('RH (%)')
ax1.grid(True)
#ax1.legend(loc='lower center', fontsize=12)

plt.savefig('timeseries_rh_daily_iv_rh.png', bbox_inches='tight', dpi=400)





# End the timer
end_time = time.time()

# Calculate the total time taken in minutes
execution_time_minutes = (end_time - start_time) / 60

# Display the time taken
print(f"Total execution time: {execution_time_minutes:.2f} minutes")
