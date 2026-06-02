"""Build the gridded surface-velocity field for a domain.

Resamples ITS_LIVE 120 m vx/vy onto the project DEM grid. The native ITS_LIVE
CRS differs from the project CRS, so we evaluate velocities by finite-differencing
the displacement of grid points after a small forward/backward step in ITS_LIVE
coordinates. This preserves the velocity vector orientation under reprojection.

Output: {domain_path}/model_inputs/gridded_velocity.nc
"""
import argparse
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


VX_PATH = '../common_data/velocity/ITS_LIVE_velocity_120m_RGI01A_0000_V02.1_vx.tif'
VY_PATH = '../common_data/velocity/ITS_LIVE_velocity_120m_RGI01A_0000_V02.1_vy.tif'

# Finite-difference step (years). Small enough to stay locally linear,
# large enough to keep round-trip projection error negligible.
FD_STEP_YEARS = 10.0


def build_velocity(domain_path: str) -> xr.Dataset:
    """Build the gridded velocity dataset for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_velocity.nc'

    dem_ds = xr.load_dataset(dem_path)
    dem_crs = pyproj.CRS(dem_ds.spatial_ref.crs_wkt)

    vx_da = xr.load_dataarray(VX_PATH)
    vy_da = xr.load_dataarray(VY_PATH)
    itslive_crs = pyproj.CRS(vx_da.mapping.crs_wkt)

    # Project DEM grid into ITS_LIVE coords for sampling
    grid_x, grid_y = np.meshgrid(dem_ds.x, dem_ds.y)
    grid_x_itslive, grid_y_itslive = pyproj.transform(dem_crs, itslive_crs, grid_x, grid_y)

    interp_vx = RegularGridInterpolator((vx_da.y, vx_da.x), vx_da.values.squeeze())
    interp_vy = RegularGridInterpolator((vx_da.y, vx_da.x), vy_da.values.squeeze())

    vx_itslive = interp_vx((grid_y_itslive, grid_x_itslive))
    vy_itslive = interp_vy((grid_y_itslive, grid_x_itslive))

    # Centered finite difference of displaced positions back in project CRS:
    # gives velocity components aligned with the project's x/y axes.
    x_plus, y_plus = pyproj.transform(
        itslive_crs, dem_crs,
        grid_x_itslive + FD_STEP_YEARS * vx_itslive,
        grid_y_itslive + FD_STEP_YEARS * vy_itslive,
    )
    x_minus, y_minus = pyproj.transform(
        itslive_crs, dem_crs,
        grid_x_itslive - FD_STEP_YEARS * vx_itslive,
        grid_y_itslive - FD_STEP_YEARS * vy_itslive,
    )

    vx_project = (x_plus - x_minus) / (2 * FD_STEP_YEARS)
    vy_project = (y_plus - y_minus) / (2 * FD_STEP_YEARS)

    velocity_ds = dem_ds.copy()
    velocity_ds['vx'] = (('y', 'x'), vx_project.astype('float32'))
    velocity_ds['vy'] = (('y', 'x'), vy_project.astype('float32'))
    velocity_ds['vmask'] = (('y','x'),(((vx_project**2 + vy_project**2)**0.5) > 1.0).astype('float32'))

    for name in ('elevation', 'topography', 'bathymetry', 'bathymetry_mask',
                 'domain_mask', 'rgi_mask'):
        del velocity_ds[name]

    velocity_ds.to_netcdf(output_path)
    return velocity_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_velocity(args.domain_path)
