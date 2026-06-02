"""Build the gridded supraglacial debris-cover mask for a domain.

The source product is a single Alaska-wide shapefile of debris-covered glacier
polygons (EPSG:3338, > 1 km^2 glaciers). This utility clips it to the domain,
rasterizes the polygons onto the project DEM grid created by make_dem.py, and
also computes a sub-pixel debris area fraction.

Pipeline:
    1. Read the target grid (CRS, transform, shape) from gridded_dem.nc.
    2. Load the debris polygons, reproject to the project CRS if needed, and
       keep only those intersecting the target grid bbox.
    3. Rasterize at the DEM resolution -> boolean `debris_mask`.
    4. Rasterize on a supersampled grid and block-average -> `debris_fraction`
       (the polygon area fraction within each coarse cell).

Output: {domain_path}/model_inputs/gridded_debris.nc
"""
import argparse
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import rasterize
from shapely.geometry import box


DEBRIS_PATH = '../common_data/debris/01Alaska_minGl1km2_debrisCover.shp'

# Linear supersampling factor for the sub-pixel area fraction. 10 -> each
# 90 m cell is evaluated as a 9 m grid before block-averaging.
SUPERSAMPLE = 10


def build_debris(domain_path: str, debris_path: str = None) -> xr.Dataset:
    """Build the gridded debris-cover dataset for `domain_path` and write it."""
    domain_path = Path(domain_path)
    debris_path = debris_path or DEBRIS_PATH
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_debris.nc'

    dem = xr.load_dataset(dem_path)
    target_crs = rasterio.crs.CRS.from_wkt(dem.spatial_ref.crs_wkt)
    grid = dem['elevation']
    grid.rio.write_crs(target_crs, inplace=True)
    transform = grid.rio.transform()
    n_y, n_x = grid.rio.shape
    left, bottom, right, top = grid.rio.bounds()

    # Load only the polygons overlapping the domain bbox.
    domain_bbox = geopandas.GeoSeries([box(left, bottom, right, top)],
                                      crs=target_crs)
    debris = geopandas.read_file(debris_path, bbox=tuple(domain_bbox.total_bounds))
    if debris.crs is None or debris.crs.to_epsg() != target_crs.to_epsg():
        debris = debris.to_crs(target_crs)
    print(f"Rasterizing {len(debris)} debris polygon(s) over the domain.")

    geometries = debris.geometry.values

    if len(geometries):
        # Boolean mask at the DEM resolution (a cell is debris if its centre
        # falls within a polygon).
        debris_mask = rasterize(
            geometries, out_shape=(n_y, n_x), transform=transform,
            fill=0, default_value=1, dtype='uint8',
        ).astype(bool)

        # Sub-pixel area fraction: rasterize on a SUPERSAMPLE-times finer grid
        # then block-average back to the DEM grid.
        fine_transform = transform * rasterio.Affine.scale(
            1.0 / SUPERSAMPLE, 1.0 / SUPERSAMPLE)
        fine = rasterize(
            geometries,
            out_shape=(n_y * SUPERSAMPLE, n_x * SUPERSAMPLE),
            transform=fine_transform, fill=0, default_value=1, dtype='uint8',
        )
        debris_fraction = fine.reshape(
            n_y, SUPERSAMPLE, n_x, SUPERSAMPLE).mean(axis=(1, 3))
    else:
        debris_mask = np.zeros((n_y, n_x), dtype=bool)
        debris_fraction = np.zeros((n_y, n_x))

    debris_ds = dem.copy()
    debris_ds['debris_mask'] = (('y', 'x'), debris_mask)
    debris_ds['debris_mask'].attrs = {
        'long_name': 'Supraglacial debris cover (cell centre within a polygon)',
        '_FillValue': False,
    }
    debris_ds['debris_fraction'] = (('y', 'x'),
                                    debris_fraction.astype('float32'))
    debris_ds['debris_fraction'].attrs = {
        'long_name': 'Polygon area fraction of debris cover within each cell',
        'units': 'dimensionless',
    }
    debris_ds.attrs['debris_source'] = str(debris_path)
    debris_ds.attrs['debris_polygons'] = len(debris)

    # Carry only projection metadata; downstream merge supplies the rest.
    for name in ('elevation', 'topography', 'bathymetry', 'bathymetry_mask',
                 'domain_mask', 'rgi_mask'):
        if name in debris_ds:
            del debris_ds[name]

    debris_ds.to_netcdf(output_path)
    return debris_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--debris-path", type=str, default=None,
                        help="Override the default debris-cover shapefile.")
    args = parser.parse_args()
    build_debris(args.domain_path, args.debris_path)
