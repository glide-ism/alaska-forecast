import numpy as np
import xarray as xr
import pyproj
import polars as pl
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import interp1d
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain-path", type=str, default=None)
args = parser.parse_args()
DOMAIN_PATH = args.domain_path

OUTPUT_PATH = f'{DOMAIN_PATH}/model_inputs/flightlines.gpkg'

GEOM_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_dem.nc'
DIRECTORY_PATHS = [Path('../common_data/flightlines/iruaf'),Path('../common_data/flightlines/irares')]
RESOLUTION = 90. # sampling distance in meters along flightlines

# Get project spatial reference
ds_dem = xr.load_dataset(GEOM_PATH)
crs_dem = pyproj.CRS(ds_dem.spatial_ref.crs_wkt)
proj = pyproj.Proj(crs_dem)

xs = []
ys = []
beds = []
srfs = []
for directory_path in DIRECTORY_PATHS:
    for file_path in directory_path.iterdir():
        if file_path.is_file() and '.csv' in file_path.name:
            data = pl.read_csv(file_path,skip_rows=13)
            x,y = proj(data['lon_deg_e'].to_numpy(),data['lat_deg_n'].to_numpy())
            bed = data['bed_height_m'].cast(pl.Float64).fill_null(np.nan).to_numpy()
            srf = data['surface_height_m'].cast(pl.Float64).fill_null(np.nan).to_numpy()
            
            delta_x,delta_y = np.diff(x),np.diff(y)
            delta_r = np.hstack(([0],np.sqrt(delta_x**2 + delta_y**2) + 1e-3))
            r = np.cumsum(delta_r)
            x_interp = interp1d(r,x)(np.arange(r.min(),r.max(),RESOLUTION))
            y_interp = interp1d(r,y)(np.arange(r.min(),r.max(),RESOLUTION))
            bed_interp = interp1d(r,bed)(np.arange(r.min(),r.max(),RESOLUTION))
            srf_interp = interp1d(r,srf)(np.arange(r.min(),r.max(),RESOLUTION))
            xs.append(x_interp)
            ys.append(y_interp)
            srfs.append(srf_interp)
            beds.append(bed_interp)

xs = np.hstack(xs)
ys = np.hstack(ys)
beds = np.hstack(beds)
srfs = np.hstack(srfs)

x_min,x_max = ds_dem.x.values.min(),ds_dem.x.values.max()
y_min,y_max = ds_dem.y.values.min(),ds_dem.y.values.max()

bbox_mask = ((xs > x_min) 
        & (x_max > xs)
        & (ys > y_min) 
        & (y_max > ys) 
        & ~np.isnan(beds) 
        & ~np.isnan(srfs))

df = pl.DataFrame({
    "x": xs[bbox_mask],
    "y": ys[bbox_mask],
    "surface": srfs[bbox_mask],
    "bed": beds[bbox_mask],
    "thk": (srfs - beds)[bbox_mask]
})

# Convert to pandas (required bridge)
pdf = df.to_pandas()

# Build geometry
geometry = [Point(xy) for xy in zip(pdf.x, pdf.y)]

gdf = gpd.GeoDataFrame(
    pdf,
    geometry=geometry,
    crs=crs_dem.to_wkt()
)

gdf.to_file(OUTPUT_PATH, driver="GPKG")




