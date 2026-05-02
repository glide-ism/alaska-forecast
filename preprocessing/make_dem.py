import xarray as xr
import rioxarray
import numpy as np
import shapely
import geopandas

from bmi_topography import Topography
from projection_dictionary import crs

import math
import os
from pathlib import Path
from urllib.parse import urlencode

import requests
import math

import argparse

def largest_valid_rectangle(valid):
    rows, cols = valid.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)  # area, y_start, x_start, y_end, x_end

    for i in range(rows):
        heights = np.where(valid[i], heights + 1, 0)

        # Largest rectangle in this histogram row
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                x_start = 0 if not stack else stack[-1] + 1
                width = j - x_start
                area = height * width
                if area > best[0]:
                    best = (area, i - height + 1, x_start, i, x_start + width - 1)
            stack.append(j)

    _, y_start, x_start, y_end, x_end = best
    return y_start, y_end, x_start, x_end


def export_dem(
    xmin,
    ymin,
    xmax,
    ymax,
    out_file,
    resolution_deg=None,
    width=None,
    height=None,
    bbox_sr=4326,
    image_sr=4326,
    interpolation="RSP_BilinearInterpolation",
    fmt="tiff",
):
    """
    Download a DEM subset from NOAA DEM Global Mosaic using ArcGIS exportImage.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : float
        Bounding box in degrees lon/lat (EPSG:4326 by default).
    out_file : str or Path
        Output GeoTIFF path.
    resolution_deg : float, optional
        Output pixel size in degrees. If supplied, width/height are computed.
    width, height : int, optional
        Output raster dimensions. Use either resolution_deg or width/height.
    bbox_sr, image_sr : int
        Spatial references for bbox and output image.
    interpolation : str
        ArcGIS interpolation enum.
    fmt : str
        "tiff" is what you want for numeric elevation values.
    """
    out_file = Path(out_file)

    if resolution_deg is not None:
        width = math.ceil((xmax - xmin) / resolution_deg)
        height = math.ceil((ymax - ymin) / resolution_deg)

    if width is None or height is None:
        raise ValueError("Provide either resolution_deg or both width and height.")

    if width > 20000 or height > 20000:
        raise ValueError(
            f"Requested size {width}x{height} exceeds NOAA service limit of 20000x20000."
        )

    params = {
        "f": "image",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": bbox_sr,
        "imageSR": image_sr,
        "size": f"{width},{height}",
        "format": fmt,
        "pixelType": "F32",
        "interpolation": interpolation,
    }

    url = f"{SERVICE}/exportImage?{urlencode(params)}"
    print("GET", url)

    r = requests.get(url, timeout=300)
    r.raise_for_status()

    out_file.write_bytes(r.content)
    print(f"Wrote {out_file}")


def query_sources(xmin, ymin, xmax, ymax, bbox_sr=4326):
    """
    Query source rasters contributing within a bbox.
    Returns ArcGIS feature JSON.
    """
    geom = {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "spatialReference": {"wkid": bbox_sr},
    }

    params = {
        "f": "json",
        "geometryType": "esriGeometryEnvelope",
        "geometry": requests.utils.quote(str(geom).replace("'", '"'), safe="{}\":,"),
        "inSR": bbox_sr,
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "false",
        "outFields": ",".join(
            [
                "OBJECTID",
                "Name",
                "DemName",
                "DateCompleted",
                "CellsizeArcseconds",
                "VerticalDatum",
                "MetadataURL",
                "DEM_ID",
                "ZOrder",
            ]
        ),
        "resultRecordCount": 1000,
    }

    # Build manually because geometry is JSON-like text
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{SERVICE}/query?{query_string}"

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def export_dem_meterish(
    xmin, ymin, xmax, ymax, out_file, target_m=30.0, lat_ref=None
):
    """
    Approximate a target metric resolution while requesting in geographic coords.
    """
    if lat_ref is None:
        lat_ref = 0.5 * (ymin + ymax)

    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))

    dx_deg = target_m / meters_per_deg_lon
    dy_deg = target_m / meters_per_deg_lat

    width = math.ceil((xmax - xmin) / dx_deg)
    height = math.ceil((ymax - ymin) / dy_deg)

    export_dem(
        xmin, ymin, xmax, ymax,
        out_file=out_file,
        width=width,
        height=height,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--domain-path", type=str, default=None)
args = parser.parse_args()
DOMAIN_PATH = args.domain_path

OUTPUT_PATH = f'{DOMAIN_PATH}/model_inputs/gridded_dem.nc'

SERVICE = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer"
OPENTOPOGRAPHY_API_KEY = 'dc10c5a9ed81a06019b050fbf754d1b1'


DOMAIN_MASK_PATH = f'{DOMAIN_PATH}/local_data/outline.kml'
GLACIER_MASK_PATH = '../common_data/area/rgi/rgi_ak/RGI2000-v7.0-C-01_alaska.shp'

domain_mask = geopandas.read_file(DOMAIN_MASK_PATH)
domain_polygon = domain_mask['geometry'][0]
domain_bounds = domain_polygon.bounds

MASK_BUFFER = 0.1
params = Topography.DEFAULT.copy()
params['api_key'] = OPENTOPOGRAPHY_API_KEY
params['dem_type'] = 'COP90'
params['south'] = domain_bounds[1]
params['north'] = domain_bounds[3]
params['west'] = domain_bounds[0]
params['east'] = domain_bounds[2]

topo = Topography(**params)
topo.fetch()

topo_array = topo.load().rename('elevation')
topo_array = topo_array.isel(band=0).drop_vars('band')

if topo_array.rio.crs is None:
    topo_array.rio.write_crs("EPSG:4326", inplace=True)

# Example 1: download at roughly 1 arc-second
export_dem_meterish(
    domain_bounds[0], domain_bounds[1], domain_bounds[2], domain_bounds[3],
    out_file=f"{DOMAIN_PATH}/local_data/noaa_global_mosaic.tif",
    target_m=90.0,
)

bathy_array = xr.load_dataset(f'{DOMAIN_PATH}/local_data/noaa_global_mosaic.tif')
bathy_array = bathy_array.interp_like(topo_array).band_data[0].astype(np.float32)

noaa_bag_directory = Path('../common_data/dem/bathy/bags')
for file in noaa_bag_directory.iterdir():

    da_bag = rioxarray.open_rasterio(file).astype(np.float32)
    dq = da_bag.rio.reproject_match(bathy_array)[0]
    bathy_array = xr.where(dq==1e6, bathy_array,dq)

topo_array.name = 'topography'
bathy_array.name= 'bathymetry'
comb = xr.merge([topo_array,bathy_array])

comb = comb.rio.reproject(crs,resolution=90)
comb['elevation'] = xr.where((comb.topography > 0) | (comb.bathymetry>0), comb.topography, comb.bathymetry)
comb['bathymetry_mask'] = xr.where(comb.topography > 0, False, True)
comb['x'] = comb['x'].astype('float32')
comb['y'] = comb['y'].astype('float32')

# Get the elevation part of projected DEM
valid = comb.elevation.notnull().values

# Get the largest bounding box that contains all valid pixels
y_start, y_end, x_start, x_end = largest_valid_rectangle(valid)
comb = comb.isel(y=slice(y_start, y_end + 1), x=slice(x_start, x_end + 1))

# Convert to a mask
domain_mask_array = comb.elevation.rio.clip([domain_polygon],crs="EPSG:4326",invert=False,drop=False).notnull()
domain_mask_array.attrs['_FillValue'] = False
comb['domain_mask'] = domain_mask_array.astype('bool')

glacier_mask = geopandas.read_file(GLACIER_MASK_PATH)

# Load RGI glaciers and convert to a raster mask
glacier_mask_array = comb.elevation.rio.clip(glacier_mask.geometry.values,glacier_mask.crs,drop=False).notnull()
glacier_mask_array.attrs['_FillValue'] = False
comb['rgi_mask'] = glacier_mask_array.astype('bool')

# Write combined DEM and masks to nc
comb.to_netcdf(OUTPUT_PATH)
