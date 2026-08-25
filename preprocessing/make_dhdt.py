"""Build the gridded surface elevation-change rate (dH/dt) field for a domain.

The source product is the Hugonnet et al. (2021) global glacier elevation-change
archive: a directory of 1-degree GeoTIFF tiles giving the mean rate of surface
elevation change (m/yr) over a fixed epoch (here 2000-2020), with a companion
per-pixel uncertainty tile. Unlike the snowline product, the tiles are delivered
in per-zone UTM CRSs, so they cannot be stitched directly — each intersecting
tile is reprojected and resampled onto the project DEM grid and the results are
coalesced.

Pipeline:
    1. Read the target grid (CRS, transform, shape) from gridded_dem.nc.
    2. Header-scan the dhdt directory and keep only the tiles whose footprint
       intersects the target grid bbox (in the project CRS).
    3. Reproject_match each intersecting tile (and its uncertainty twin) onto
       the DEM grid (bilinear), then coalesce the tiles into one coverage.

Output: {domain_path}/model_inputs/gridded_dhdt.nc
"""
import argparse
from pathlib import Path

import rasterio
import rasterio.warp
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling


DHDT_DIR = Path('../common_data/dhdt/dhdt')
DHDT_ERR_DIR = Path('../common_data/dhdt/dhdt_err')

# Tile filename suffixes (the error twin shares the stem up to this suffix).
DHDT_SUFFIX = '_dhdt.tif'
DHDT_ERR_SUFFIX = '_dhdt_err.tif'

# Observation window of the Hugonnet (2021) product used here. Written as
# variable-level attrs; the dH/dt misfit compares the model's two-snapshot
# rate (H(time_end) - H(time_start)) / (time_end - time_start) against it.
TIME_ATTRS = {
    'time_nominal': 2010.0,
    'time_start': 2000.0,
    'time_end': 2020.0,
}


def _intersecting_tiles(dhdt_dir, bounds, target_crs):
    """Return the dhdt tiles whose footprint intersects `bounds`.

    `bounds` is (left, bottom, right, top) in `target_crs`. Only the GeoTIFF
    headers are read, so this stays cheap across the full archive. Tiles live in
    per-zone UTM CRSs, so each tile's bounds are reprojected before comparison.
    """
    left, bottom, right, top = bounds
    selected = []
    for tile_path in sorted(dhdt_dir.glob(f'*{DHDT_SUFFIX}')):
        with rasterio.open(tile_path) as src:
            tl, tb, tr, tt = rasterio.warp.transform_bounds(
                src.crs, target_crs, *src.bounds,
            )
        if tr < left or tl > right or tt < bottom or tb > top:
            continue
        selected.append(tile_path)
    return selected


def _coalesce_onto_grid(tiles, grid, resampling):
    """Reproject each tile onto `grid` and combine into a single coverage.

    Tiles are non-overlapping 1-degree footprints, so a `combine_first` coalesce
    (first valid value wins, NaN elsewhere) reconstructs the mosaic. `masked=True`
    promotes the -9999 no-data sentinel to NaN so it neither bleeds across tile
    edges under bilinear resampling nor survives the coalesce.
    """
    mosaic = None
    for tile_path in tiles:
        tile = rioxarray.open_rasterio(tile_path, masked=True).squeeze('band', drop=True)
        matched = tile.rio.reproject_match(grid, resampling=resampling)
        # reproject_match aligns coords to the grid to floating-point dust;
        # snap to the grid's so combine_first lines up exactly.
        matched = matched.assign_coords(x=grid.x, y=grid.y)
        mosaic = matched if mosaic is None else mosaic.combine_first(matched)
    return mosaic


def build_dhdt(domain_path: str, dhdt_dir: str = None,
               dhdt_err_dir: str = None) -> xr.Dataset:
    """Build the gridded dH/dt dataset for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    dhdt_dir = Path(dhdt_dir) if dhdt_dir else DHDT_DIR
    dhdt_err_dir = Path(dhdt_err_dir) if dhdt_err_dir else DHDT_ERR_DIR
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_dhdt.nc'

    dem = xr.load_dataset(dem_path)
    target_crs = rasterio.crs.CRS.from_wkt(dem.spatial_ref.crs_wkt)
    grid = dem['elevation']
    grid.rio.write_crs(target_crs, inplace=True)

    tiles = _intersecting_tiles(dhdt_dir, grid.rio.bounds(), target_crs)
    if not tiles:
        raise RuntimeError(
            f"No dhdt tiles in {dhdt_dir} intersect the domain grid."
        )
    print(f"Reprojecting {len(tiles)} dhdt tile(s) onto the domain grid.")

    # Pair each rate tile with its uncertainty twin (same stem, _err suffix).
    err_tiles = [dhdt_err_dir / (t.name[:-len(DHDT_SUFFIX)] + DHDT_ERR_SUFFIX)
                 for t in tiles]
    missing = [p.name for p in err_tiles if not p.exists()]
    if missing:
        print(f"Warning: {len(missing)} dhdt_err tile(s) missing, e.g. {missing[0]}")
    err_tiles = [p for p in err_tiles if p.exists()]

    dhdt = _coalesce_onto_grid(tiles, grid, Resampling.bilinear)
    dhdt_err = _coalesce_onto_grid(err_tiles, grid, Resampling.bilinear)

    dhdt_ds = dem.copy()
    dhdt_ds['dhdt'] = (('y', 'x'), dhdt.values.astype('float32'))
    dhdt_ds['dhdt'].attrs = {
        'long_name': 'Mean rate of surface elevation change',
        'units': 'm yr-1',
        **TIME_ATTRS,
    }
    dhdt_ds['dhdt_err'] = (('y', 'x'), dhdt_err.values.astype('float32'))
    dhdt_ds['dhdt_err'].attrs = {
        'long_name': 'Uncertainty (1-sigma) of the elevation-change rate',
        'units': 'm yr-1',
        **TIME_ATTRS,
    }
    dhdt_ds.attrs['dhdt_source'] = str(dhdt_dir)
    dhdt_ds.attrs['dhdt_tiles'] = len(tiles)

    # Carry only projection metadata; downstream merge supplies the rest.
    for name in ('elevation', 'topography', 'bathymetry', 'bathymetry_mask',
                 'domain_mask', 'rgi_mask'):
        if name in dhdt_ds:
            del dhdt_ds[name]

    dhdt_ds.to_netcdf(output_path)
    return dhdt_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--dhdt-dir", type=str, default=None,
                        help="Override the default dhdt tile directory.")
    parser.add_argument("--dhdt-err-dir", type=str, default=None,
                        help="Override the default dhdt_err tile directory.")
    args = parser.parse_args()
    build_dhdt(args.domain_path, args.dhdt_dir, args.dhdt_err_dir)
