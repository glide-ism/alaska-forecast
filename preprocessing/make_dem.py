"""Build the gridded DEM for a domain.

Combines a topographic DEM (Copernicus 90m via OpenTopography) with NOAA
bathymetry, then reprojects onto an explicit project-CRS grid that spans
the bounding box of the domain outline in projected coordinates.

Pipeline:
    1. Project the outline polygon into the project CRS, take its bbox, and
       snap the bbox outward to multiples of the grid resolution. This bbox
       is the *target* output grid extent.
    2. Inverse-project a dense sample of that bbox boundary back to lat/lon
       to determine the geographic fetch extent. Fetching this slightly
       larger lat/lon area guarantees the rotated DEM, after reprojection,
       fully covers the target bbox with no NaN corners to crop away.
    3. Fetch topography (OpenTopography) and bathymetry (NOAA mosaic + BAGs)
       over the lat/lon fetch extent, splice them, then reproject onto the
       explicit target grid using a fixed destination transform.
    4. Rasterize the domain mask, the RGI glacier mask, and a per-glacier
       integer label field (with a lookup back to raw RGI identifiers).

Output: {domain_path}/model_inputs/gridded_dem.nc
"""
import argparse
import math
import os
from pathlib import Path
from urllib.parse import urlencode

import geopandas
import numpy as np
import pyproj
import rasterio.features
import rasterio.transform
import requests
import rioxarray
import xarray as xr
from bmi_topography import Topography

from projection_dictionary import crs


# Free per-user key from
# https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets
# Set it in your shell, e.g. `export OPENTOPOGRAPHY_API_KEY=...`.
OPENTOPOGRAPHY_API_KEY = os.environ.get("OPENTOPOGRAPHY_API_KEY")
NOAA_DEM_SERVICE = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/"
    "DEM_mosaics/DEM_global_mosaic/ImageServer"
)
#GLACIER_MASK_PATH = '../common_data/area/rgi/rgi_ak/RGI2000-v7.0-C-01_alaska.shp'
GLACIER_MASK_PATH = '../common_data/area/rgi/rgi_ak/RGI2000-v7.0-G-01_alaska.shp'
NOAA_BAG_DIRECTORY = Path('../common_data/dem/bathy/bags')

GRID_RESOLUTION_M = 90.0
DEM_TYPE = 'COP90'

# Number of points sampled along each edge of the projected bbox when
# back-projecting to lat/lon. 100 is plenty for Alaska Albers, where the
# bbox edges are nearly straight in lat/lon space.
BBOX_BOUNDARY_SAMPLES = 100

# Small lat/lon padding on the fetch extent to absorb numerical roundoff
# between forward/inverse projection at the very edges of the target grid.
FETCH_PAD_DEG = 0.01

def export_dem(
    xmin, ymin, xmax, ymax, out_file,
    resolution_deg=None, width=None, height=None,
    bbox_sr=4326, image_sr=4326,
    interpolation="RSP_BilinearInterpolation",
    fmt="tiff",
):
    """Download a DEM subset from NOAA DEM Global Mosaic via ArcGIS exportImage."""
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

    url = f"{NOAA_DEM_SERVICE}/exportImage?{urlencode(params)}"
    print("GET", url)

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    out_file.write_bytes(response.content)
    print(f"Wrote {out_file}")


def query_sources(xmin, ymin, xmax, ymax, bbox_sr=4326):
    """Query source rasters contributing within a bbox. Returns ArcGIS feature JSON."""
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
                "OBJECTID", "Name", "DemName", "DateCompleted",
                "CellsizeArcseconds", "VerticalDatum", "MetadataURL",
                "DEM_ID", "ZOrder",
            ]
        ),
        "resultRecordCount": 1000,
    }

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{NOAA_DEM_SERVICE}/query?{query_string}"

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def export_dem_meterish(xmin, ymin, xmax, ymax, out_file, target_m=30.0, lat_ref=None):
    """Approximate a target metric resolution while requesting in geographic coords."""
    if lat_ref is None:
        lat_ref = 0.5 * (ymin + ymax)

    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))

    width = math.ceil((xmax - xmin) / (target_m / meters_per_deg_lon))
    height = math.ceil((ymax - ymin) / (target_m / meters_per_deg_lat))

    export_dem(xmin, ymin, xmax, ymax, out_file=out_file, width=width, height=height)


def _project_polygon_bbox(polygon, dst_crs, src_crs="EPSG:4326"):
    """Project the exterior coords of `polygon` from src_crs to dst_crs and
    return its (xmin, ymin, xmax, ymax) bbox in dst_crs.

    Tolerates 3D coords (KML often carries an altitude in the z slot).
    """
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    coords = np.asarray(polygon.exterior.coords)
    lons, lats = coords[:, 0], coords[:, 1]
    xs, ys = transformer.transform(lons, lats)
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def _snap_bbox_outward(bbox, resolution):
    """Snap (xmin, ymin, xmax, ymax) outward to multiples of `resolution`.
    The result is the smallest grid-aligned bbox containing the input."""
    xmin, ymin, xmax, ymax = bbox
    return (
        math.floor(xmin / resolution) * resolution,
        math.floor(ymin / resolution) * resolution,
        math.ceil(xmax / resolution) * resolution,
        math.ceil(ymax / resolution) * resolution,
    )


def _projected_bbox_to_latlon_extent(bbox, src_crs, n_samples=BBOX_BOUNDARY_SAMPLES,
                                     pad_deg=FETCH_PAD_DEG):
    """Return the (lon_min, lat_min, lon_max, lat_max) extent that, when
    fetched in EPSG:4326 and reprojected to src_crs, fully covers `bbox`.

    Densely samples the boundary of the projected bbox and inverse-projects
    each sample, taking the lat/lon envelope. A small pad absorbs any
    numerical drift at the corners.
    """
    xmin, ymin, xmax, ymax = bbox
    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    edge = np.linspace(0.0, 1.0, n_samples)
    xs = np.concatenate([
        xmin + edge * (xmax - xmin),    # bottom
        np.full(n_samples, xmax),        # right
        xmax - edge * (xmax - xmin),    # top
        np.full(n_samples, xmin),        # left
    ])
    ys = np.concatenate([
        np.full(n_samples, ymin),
        ymin + edge * (ymax - ymin),
        np.full(n_samples, ymax),
        ymax - edge * (ymax - ymin),
    ])
    lons, lats = transformer.transform(xs, ys)
    return (
        float(np.min(lons)) - pad_deg,
        float(np.min(lats)) - pad_deg,
        float(np.max(lons)) + pad_deg,
        float(np.max(lats)) + pad_deg,
    )


def build_dem(domain_path: str) -> xr.Dataset:
    """Build the gridded DEM dataset for `domain_path` and write it to disk.

    Returns the in-memory Dataset (also written to
    {domain_path}/model_inputs/gridded_dem.nc).
    """
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    domain_outline = geopandas.read_file(domain_path / 'local_data' / 'outline.kml')
    domain_polygon = domain_outline['geometry'][0]

    # Define the target output grid in projected coordinates: bbox of the
    # outline in project CRS, snapped outward to grid resolution.
    target_bbox = _snap_bbox_outward(
        _project_polygon_bbox(domain_polygon, crs),
        GRID_RESOLUTION_M,
    )
    target_xmin, target_ymin, target_xmax, target_ymax = target_bbox
    target_width = int(round((target_xmax - target_xmin) / GRID_RESOLUTION_M))
    target_height = int(round((target_ymax - target_ymin) / GRID_RESOLUTION_M))
    dst_transform = rasterio.transform.from_origin(
        target_xmin, target_ymax, GRID_RESOLUTION_M, GRID_RESOLUTION_M,
    )

    # Lat/lon extent that, after reprojection back to project CRS,
    # fully covers the target grid (no NaN corners after rotation).
    lon_min, lat_min, lon_max, lat_max = _projected_bbox_to_latlon_extent(
        target_bbox, crs,
    )

    # Fetch topography over the lat/lon fetch extent
    if not OPENTOPOGRAPHY_API_KEY:
        raise RuntimeError(
            "Set the OPENTOPOGRAPHY_API_KEY environment variable. Get a free key at "
            "https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets"
        )
    topography_params = Topography.DEFAULT.copy()
    topography_params['api_key'] = OPENTOPOGRAPHY_API_KEY
    topography_params['dem_type'] = DEM_TYPE
    topography_params['south'] = lat_min
    topography_params['north'] = lat_max
    topography_params['west'] = lon_min
    topography_params['east'] = lon_max

    topography = Topography(**topography_params)
    topography.fetch()

    topography_array = topography.load().rename('elevation').isel(band=0).drop_vars('band')
    if topography_array.rio.crs is None:
        topography_array.rio.write_crs("EPSG:4326", inplace=True)

    # Fetch bathymetry over the same lat/lon fetch extent at ~grid resolution
    bathymetry_tif_path = domain_path / 'local_data' / 'noaa_global_mosaic.tif'
    export_dem_meterish(
        lon_min, lat_min, lon_max, lat_max,
        out_file=bathymetry_tif_path,
        target_m=GRID_RESOLUTION_M,
    )
    bathymetry_array = xr.load_dataset(bathymetry_tif_path)
    bathymetry_array = bathymetry_array.interp_like(topography_array).band_data[0].astype(np.float32)

    # Splice in higher-resolution NOAA BAG bathymetry where available.
    # The mosaic uses 1e6 as a no-data sentinel; keep mosaic values there.
    for bag_path in NOAA_BAG_DIRECTORY.iterdir():
        bag_raster = rioxarray.open_rasterio(bag_path).astype(np.float32)
        bag_resampled = bag_raster.rio.reproject_match(bathymetry_array)[0]
        bathymetry_array = xr.where(bag_resampled == 1e6, bathymetry_array, bag_resampled)

    topography_array.name = 'topography'
    bathymetry_array.name = 'bathymetry'
    dem_ds = xr.merge([topography_array, bathymetry_array])

    # Reproject onto the explicit target grid. The fetch extent was sized to
    # guarantee the rotated DEM fully covers this grid.
    dem_ds = dem_ds.rio.reproject(
        crs, transform=dst_transform, shape=(target_height, target_width),
    )

    # Prefer topography on land, bathymetry below sea level
    dem_ds['elevation'] = xr.where(
        (dem_ds.topography > 0) | (dem_ds.bathymetry > 0),
        dem_ds.topography,
        dem_ds.bathymetry,
    )
    dem_ds['bathymetry_mask'] = xr.where(dem_ds.topography > 0, False, True)
    dem_ds['x'] = dem_ds['x'].astype('float32')
    dem_ds['y'] = dem_ds['y'].astype('float32')

    # Rasterize the domain outline and the RGI glacier polygons onto the grid
    domain_mask = dem_ds.elevation.rio.clip(
        [domain_polygon], crs="EPSG:4326", invert=False, drop=False,
    ).notnull()
    domain_mask.attrs['_FillValue'] = False
    dem_ds['domain_mask'] = domain_mask.astype('bool')

    glacier_polygons = geopandas.read_file(GLACIER_MASK_PATH)
    rgi_mask = dem_ds.elevation.rio.clip(
        glacier_polygons.geometry.values, glacier_polygons.crs, drop=False,
    ).notnull()
    rgi_mask.attrs['_FillValue'] = False
    dem_ds['rgi_mask'] = rgi_mask.astype('bool')

    # Per-glacier integer labels. Reproject the RGI polygons into the grid CRS,
    # keep those intersecting the target grid, and assign each a sequential
    # integer (0..N-1). Pixels not covered by any glacier get -1. A 1-D lookup
    # variable (`rgi_id`, indexed by the `glacier` dimension) maps each integer
    # label `k` back to its raw RGI identifier: rgi_id[k].
    glaciers_in_grid = glacier_polygons.to_crs(crs).cx[
        target_xmin:target_xmax, target_ymin:target_ymax
    ].reset_index(drop=True)

    label_raster = rasterio.features.rasterize(
        ((geom, label) for label, geom in enumerate(glaciers_in_grid.geometry)),
        out_shape=(target_height, target_width),
        transform=dst_transform,
        fill=-1,
        dtype='int32',
    )
    rgi_label = xr.DataArray(
        label_raster,
        dims=('y', 'x'),
        coords={'y': dem_ds['y'], 'x': dem_ds['x']},
    )
    rgi_label.attrs['_FillValue'] = -1
    dem_ds['rgi_label'] = rgi_label
    dem_ds['rgi_id'] = xr.DataArray(
        glaciers_in_grid['rgi_id'].to_numpy().astype('U'),
        dims=('glacier',),
        coords={'glacier': np.arange(len(glaciers_in_grid), dtype='int32')},
    )
    # Per-glacier surge classification (RGI surge_type: 0 = no evidence,
    # 1-3 = possible/probable/observed). Indexed by the same `glacier`
    # dimension, so surge_type[k] corresponds to rgi_label == k.
    dem_ds['surge_type'] = xr.DataArray(
        glaciers_in_grid['surge_type'].to_numpy().astype('int32'),
        dims=('glacier',),
        coords={'glacier': np.arange(len(glaciers_in_grid), dtype='int32')},
    )

    dem_ds.to_netcdf(output_path)
    return dem_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_dem(args.domain_path)
