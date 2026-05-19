"""Build the gridded end-of-summer snow/ice mask for a domain.

The source product is a directory of per-glacier GeoTIFFs (organized by RGI
index, EPSG:3338, ~10 m) classifying each pixel as exposed ice or retained
snow at the end of the melt season. This utility selects the tiles that
overlap the domain, mosaics them into a single coverage, and resamples that
mosaic onto the project DEM grid created by make_dem.py.

Pipeline:
    1. Read the target grid (CRS, transform, shape) from gridded_dem.nc.
    2. Header-scan the snowline directory and keep only the tiles whose
       footprint intersects the target grid bbox (in the project CRS).
    3. Mosaic the intersecting tiles (0 = no-data sentinel).
    4. Resample onto the DEM grid two ways:
         - `snow_ice`: nearest-neighbour, preserving the categorical codes.
         - `snow_fraction` / `ice_fraction` / `glacier_fraction`: area
           averages of the native binary masks (true "interpolation" of the
           merged mask onto the coarser grid).

Output: {domain_path}/model_inputs/gridded_snowline.nc
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge


SNOWLINE_DIR = Path('../common_data/snowlines/2018-2022-average')

# Categorical codes used by the source product.
NODATA_CODE = 0
ICE_CODE = 1
SNOW_CODE = 2


def _intersecting_tiles(snowline_dir, bounds, target_crs):
    """Return the snowline tiles whose footprint intersects `bounds`.

    `bounds` is (left, bottom, right, top) in `target_crs`. Only the GeoTIFF
    headers are read, so this stays cheap across the full ~3000-tile archive.
    """
    left, bottom, right, top = bounds
    selected = []
    for tile_path in sorted(snowline_dir.glob('*.tif')):
        with rasterio.open(tile_path) as src:
            tl, tb, tr, tt = rasterio.warp.transform_bounds(
                src.crs, target_crs, *src.bounds,
            )
        if tr < left or tl > right or tt < bottom or tb > top:
            continue
        selected.append(tile_path)
    return selected


def _resample_to_grid(source_da, match_da, resampling):
    """Resample `source_da` onto the grid of `match_da` with `resampling`."""
    return source_da.rio.reproject_match(match_da, resampling=resampling)


def build_snowline(domain_path: str, snowline_dir: str = None) -> xr.Dataset:
    """Build the gridded snow/ice dataset for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    snowline_dir = Path(snowline_dir) if snowline_dir else SNOWLINE_DIR
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_snowline.nc'

    dem = xr.load_dataset(dem_path)
    target_crs = rasterio.crs.CRS.from_wkt(dem.spatial_ref.crs_wkt)
    grid = dem['elevation']
    grid.rio.write_crs(target_crs, inplace=True)

    tiles = _intersecting_tiles(snowline_dir, grid.rio.bounds(), target_crs)
    if not tiles:
        raise RuntimeError(
            f"No snowline tiles in {snowline_dir} intersect the domain grid."
        )
    print(f"Mosaicking {len(tiles)} snowline tile(s) over the domain.")

    # Mosaic the native-resolution tiles. They share the project CRS, so the
    # merge is a straight stitch; 0 is the product's no-data sentinel.
    datasets = [rasterio.open(t) for t in tiles]
    try:
        mosaic, mosaic_transform = rasterio_merge(datasets, nodata=NODATA_CODE)
        mosaic_crs = datasets[0].crs
    finally:
        for ds in datasets:
            ds.close()

    mosaic = mosaic[0].astype(np.int32)
    n_y, n_x = mosaic.shape
    xs = mosaic_transform.c + mosaic_transform.a * (np.arange(n_x) + 0.5)
    ys = mosaic_transform.f + mosaic_transform.e * (np.arange(n_y) + 0.5)

    def _native(values, dtype):
        da = xr.DataArray(
            values.astype(dtype), dims=('y', 'x'), coords={'y': ys, 'x': xs},
        )
        da.rio.write_crs(mosaic_crs, inplace=True)
        da.rio.write_nodata(np.nan if dtype == np.float32 else NODATA_CODE,
                             inplace=True)
        return da

    code_native = _native(mosaic, np.int32)
    valid_native = _native(mosaic != NODATA_CODE, np.float32)
    snow_native = _native(mosaic == SNOW_CODE, np.float32)
    ice_native = _native(mosaic == ICE_CODE, np.float32)

    # Nearest-neighbour keeps the categorical codes intact; the binary
    # fractions are area-averaged onto the (coarser) DEM grid.
    snow_ice = _resample_to_grid(code_native, grid, Resampling.nearest)
    glacier_fraction = _resample_to_grid(valid_native, grid, Resampling.average)
    snow_avg = _resample_to_grid(snow_native, grid, Resampling.average)
    ice_avg = _resample_to_grid(ice_native, grid, Resampling.average)

    # Express snow/ice as a fraction of the classified (glacierized) subarea
    # so partially-covered coarse cells are not diluted by no-data.
    safe = glacier_fraction.where(glacier_fraction > 0)
    snow_fraction = (snow_avg / safe).astype('float32')
    ice_fraction = (ice_avg / safe).astype('float32')

    snowline_ds = dem.copy()
    snowline_ds['snow_ice'] = (('y', 'x'), snow_ice.values.astype('int32'))
    snowline_ds['snow_ice'].attrs = {
        'long_name': 'End-of-summer surface classification (nearest-neighbour)',
        'flag_values': [NODATA_CODE, ICE_CODE, SNOW_CODE],
        'flag_meanings': 'no_data ice snow',
    }
    snowline_ds['snow_fraction'] = (('y', 'x'), snow_fraction.values)
    snowline_ds['snow_fraction'].attrs = {
        'long_name': 'Fraction of glacierized subcell area classified as snow',
        'units': 'dimensionless',
    }
    snowline_ds['ice_fraction'] = (('y', 'x'), ice_fraction.values)
    snowline_ds['ice_fraction'].attrs = {
        'long_name': 'Fraction of glacierized subcell area classified as ice',
        'units': 'dimensionless',
    }
    snowline_ds['glacier_fraction'] = (
        ('y', 'x'), glacier_fraction.values.astype('float32'))
    snowline_ds['glacier_fraction'].attrs = {
        'long_name': 'Fraction of coarse cell with valid snowline classification',
        'units': 'dimensionless',
    }
    snowline_ds.attrs['snowline_source'] = str(snowline_dir)
    snowline_ds.attrs['snowline_tiles'] = len(tiles)

    # Carry only projection metadata; downstream merge supplies the rest.
    for name in ('elevation', 'topography', 'bathymetry', 'bathymetry_mask',
                 'domain_mask', 'rgi_mask'):
        if name in snowline_ds:
            del snowline_ds[name]

    snowline_ds.to_netcdf(output_path)
    return snowline_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--snowline-dir", type=str, default=None,
                        help="Override the default snowline tile directory.")
    args = parser.parse_args()
    build_snowline(args.domain_path, args.snowline_dir)
