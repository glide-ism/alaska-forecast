import xarray as xr

GEOM_PATH = 'model_inputs/gridded_dem.nc'
VELOCITY_PATH = 'model_inputs/gridded_velocity.nc'
SOLAR_PATH = 'model_inputs/gridded_insolation.nc'
CLIMATE_PATH = 'model_inputs/gridded_climate.nc'

OUTPUT_PATH = 'model_inputs/GLIDE_inputs.nc'

geom = xr.load_dataset(GEOM_PATH)
velo = xr.load_dataset(VELOCITY_PATH)
solar = xr.load_dataset(SOLAR_PATH)
clim = xr.load_dataset(CLIMATE_PATH)

merged = xr.merge([geom,velo,solar,clim])

merged.to_netcdf(OUTPUT_PATH)







