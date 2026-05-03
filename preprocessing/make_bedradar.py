"""Build the bed radar flightline point dataset for a domain.

Reads CSV flightlines from the IRUAF and IRARES collections, projects them into
the project CRS, resamples each flight to a uniform along-track spacing, and
writes the points falling inside the DEM bounding box to a GeoPackage.

Output: {domain_path}/model_inputs/flightlines.gpkg
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pyproj
import xarray as xr
from scipy.interpolate import interp1d
from shapely.geometry import Point


FLIGHTLINE_DIRECTORIES = [
    Path('../common_data/flightlines/iruaf'),
    Path('../common_data/flightlines/irares'),
]
ALONG_TRACK_RESOLUTION_M = 90.0


def build_flightlines(domain_path: str) -> gpd.GeoDataFrame:
    """Build the flightline point dataset for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'flightlines.gpkg'

    dem_ds = xr.load_dataset(dem_path)
    dem_crs = pyproj.CRS(dem_ds.spatial_ref.crs_wkt)
    project = pyproj.Proj(dem_crs)

    xs, ys, beds, surfaces = [], [], [], []
    for directory in FLIGHTLINE_DIRECTORIES:
        for file_path in directory.iterdir():
            if not (file_path.is_file() and '.csv' in file_path.name):
                continue

            flight = pl.read_csv(file_path, skip_rows=13)
            x, y = project(
                flight['lon_deg_e'].to_numpy(),
                flight['lat_deg_n'].to_numpy(),
            )
            bed = flight['bed_height_m'].cast(pl.Float64).fill_null(np.nan).to_numpy()
            surface = flight['surface_height_m'].cast(pl.Float64).fill_null(np.nan).to_numpy()

            # Resample each flightline to a uniform along-track spacing.
            # Add 1e-3 to step lengths so cumulative distance is strictly increasing.
            dx, dy = np.diff(x), np.diff(y)
            step = np.hstack(([0], np.sqrt(dx ** 2 + dy ** 2) + 1e-3))
            arclength = np.cumsum(step)
            sample_arclengths = np.arange(arclength.min(), arclength.max(), ALONG_TRACK_RESOLUTION_M)

            xs.append(interp1d(arclength, x)(sample_arclengths))
            ys.append(interp1d(arclength, y)(sample_arclengths))
            surfaces.append(interp1d(arclength, surface)(sample_arclengths))
            beds.append(interp1d(arclength, bed)(sample_arclengths))

    xs = np.hstack(xs)
    ys = np.hstack(ys)
    beds = np.hstack(beds)
    surfaces = np.hstack(surfaces)

    # Keep only points inside the DEM bbox with finite bed and surface heights
    x_min, x_max = dem_ds.x.values.min(), dem_ds.x.values.max()
    y_min, y_max = dem_ds.y.values.min(), dem_ds.y.values.max()
    keep = (
        (xs > x_min) & (xs < x_max)
        & (ys > y_min) & (ys < y_max)
        & ~np.isnan(beds) & ~np.isnan(surfaces)
    )

    points_df = pl.DataFrame({
        "x": xs[keep],
        "y": ys[keep],
        "surface": surfaces[keep],
        "bed": beds[keep],
        "thk": (surfaces - beds)[keep],
    }).to_pandas()

    geometry = [Point(xy) for xy in zip(points_df.x, points_df.y)]
    flightlines_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs=dem_crs.to_wkt())

    flightlines_gdf.to_file(output_path, driver="GPKG")
    return flightlines_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_flightlines(args.domain_path)
