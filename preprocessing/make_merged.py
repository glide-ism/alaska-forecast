import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("--domain-path", type=str, default=None)
args = parser.parse_args()
DOMAIN_PATH = args.domain_path

GEOM_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_dem.nc'
VELOCITY_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_velocity.nc'
SOLAR_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_insolation.nc'
CLIMATE_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_climate.nc'

OUTPUT_PATH = f'{DOMAIN_PATH}/model_inputs/GLIDE_inputs.nc'

geom = xr.load_dataset(GEOM_PATH)
velo = xr.load_dataset(VELOCITY_PATH)
solar = xr.load_dataset(SOLAR_PATH)
clim = xr.load_dataset(CLIMATE_PATH)

merged = xr.merge([geom,velo,solar,clim])

merged.to_netcdf(OUTPUT_PATH)







