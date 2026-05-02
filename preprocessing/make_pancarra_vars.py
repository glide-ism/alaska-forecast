import xarray as xr
import pyproj
from projection_dictionary import crs

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LightSource
from glare import PanCarraBase 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain-path", type=str, default=None)
args = parser.parse_args()
DOMAIN_PATH = args.domain_path

OUTPUT_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_climate.nc'

GEOM_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_dem.nc'

YEAR = 2012
PRECIP_PATH = f'../common_data/climate/pancarra/{YEAR}/precip/precip.nc'
T2M_PATH = f'../common_data/climate/pancarra/{YEAR}/t2m/t2m.nc'
OROG_PATH = '../common_data/climate/pancarra/topo/topo.grib'

# Load target grid
dem = xr.load_dataset(GEOM_PATH)

# Load pancarra base
panc = PanCarraBase(PRECIP_PATH, T2M_PATH, OROG_PATH) 

# Regrid temperature and precipitation fields
_, t2m_fields, precip_fields = panc.regrid_carra2_fields(dem, crs, method='linear',t2m_lapse_rate=0.003)

# Create t2m data array
t2m_da = xr.DataArray(
        t2m_fields,
        dims=['t','y','x'],
        coords={
            't':np.arange(0,12,dtype=np.float32)/12.,
            'y': dem.y,
            'x': dem.x,
        },
        attrs = {
            "units": "Deg C",
            "long_name": "Monthly average temperatures derived from pan-arctic CARRA2",
        }
    )

# Create precip data array
precip_da = xr.DataArray(precip_fields/917*365,
        dims = ['t','y','x'],
        coords = {"t":np.arange(0,12,dtype=np.float32)/12,
                  "y": dem.y,
                  "x": dem.x},
        attrs = {"units": "m ice equivalent / yr",
                 "long_name": "Precipitation rate derived from pan-arctic CARRA2 at monthly time steps"}
        )

out_ds = dem.copy()

out_ds["monthly_t2m"] = t2m_da
out_ds["monthly_precip"] = precip_da

del out_ds['elevation']
del out_ds['domain_mask']
del out_ds['rgi_mask']
del out_ds['topography']
del out_ds['bathymetry']
del out_ds['bathymetry_mask']

out_ds.to_netcdf(OUTPUT_PATH)

