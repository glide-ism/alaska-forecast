import numpy as np
import xarray as xr
import pyproj
from scipy.interpolate import RegularGridInterpolator

GEOM_PATH = 'model_inputs/gridded_dem.nc'
VX_PATH = '../../../common_data/velocity/ITS_LIVE_velocity_120m_RGI01A_0000_V02.1_vx.tif'
VY_PATH = '../../../common_data/velocity/ITS_LIVE_velocity_120m_RGI01A_0000_V02.1_vy.tif'

OUTPUT_PATH = 'model_inputs/gridded_velocity.nc'

EPS = 10.0

# Get project grid and crs
ds_dem = xr.load_dataset(GEOM_PATH)
crs_dem = pyproj.CRS(ds_dem.spatial_ref.crs_wkt)

# Load ITSLive
ds_vx = xr.load_dataarray(VX_PATH)
ds_vy = xr.load_dataarray(VY_PATH)
crs_vx = pyproj.CRS(ds_vx.mapping.crs_wkt)

# Define DEM grid
X,Y = np.meshgrid(ds_dem.x,ds_dem.y)

# Project to ITSLive grid
X_,Y_ = pyproj.transform(crs_dem,crs_vx,X,Y)

# Build ITSLive interpolants
interpolator_vx = RegularGridInterpolator((ds_vx.y,ds_vx.x),ds_vx.values.squeeze())
interpolator_vy = RegularGridInterpolator((ds_vx.y,ds_vx.x),ds_vy.values.squeeze())

# Interpolate dem grid points
vx_pts = interpolator_vx((Y_,X_))
vy_pts = interpolator_vy((Y_,X_))

# Finite difference displacement
X_plus = X_ + EPS*vx_pts
Y_plus = Y_ + EPS*vy_pts

X_minus = X_ - EPS*vx_pts
Y_minus = Y_ - EPS*vy_pts

# Transform displaced points back to project grid
X0_plus,Y0_plus = pyproj.transform(crs_vx,crs_dem,X_plus,Y_plus)
X0_minus,Y0_minus = pyproj.transform(crs_vx,crs_dem,X_minus,Y_minus)

# Calculate velocities
vx = (X0_plus - X0_minus)/(2*EPS)
vy = (Y0_plus - Y0_minus)/(2*EPS)

# Create gridded velocity xarray
ds_dem['vx'] = (('y','x'),vx.astype('float32'))
ds_dem['vy'] = (('y','x'),vy.astype('float32'))

del ds_dem['elevation']
del ds_dem['domain_mask']
del ds_dem['rgi_mask']

ds_dem.to_netcdf(OUTPUT_PATH)




