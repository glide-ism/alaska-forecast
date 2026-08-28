"""Merge per-variable gridded NetCDFs into a single GLIDE input file.

Reads gridded_dem, gridded_velocity, gridded_snowline, gridded_debris,
gridded_dhdt, gridded_insolation, gridded_climate (CARRA) and
gridded_climate_era5land from {domain_path}/model_inputs and writes their
xr.merge to GLIDE_inputs.nc.
Both climatologies are included to facilitate source intercomparison.

Output: {domain_path}/model_inputs/GLIDE_inputs.nc
"""
import argparse
import os
from pathlib import Path

import xarray as xr


def build_merged(domain_path: str) -> xr.Dataset:
    """Merge the per-variable gridded NetCDFs for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    inputs_dir = domain_path / 'model_inputs'
    output_path = inputs_dir / 'GLIDE_inputs.nc'

    geometry = xr.load_dataset(inputs_dir / 'gridded_dem.nc')
    velocity = xr.load_dataset(inputs_dir / 'gridded_velocity.nc')
    snowline = xr.load_dataset(inputs_dir / 'gridded_snowline.nc')
    debris = xr.load_dataset(inputs_dir / 'gridded_debris.nc')
    dhdt = xr.load_dataset(inputs_dir / 'gridded_dhdt.nc')
    insolation = xr.load_dataset(inputs_dir / 'gridded_insolation.nc')
    climate = xr.load_dataset(inputs_dir / 'gridded_climate.nc')
    climate_era5land = xr.load_dataset(inputs_dir / 'gridded_climate_era5land.nc')

    merged = xr.merge([geometry, velocity, snowline, debris, dhdt, insolation,
                       climate, climate_era5land])

    # The inverse model reads acquisition times from variable-level attrs;
    # guard against a future merge/encoding change silently dropping them.
    for var in ('elevation', 'vx', 'rgi_mask', 'snow_fraction', 'dhdt'):
        if var in merged and 'time_nominal' not in merged[var].attrs:
            print(f"Warning: {var} lost its time attrs in the merge; "
                  f"the inverse model will fall back to t_end for it.")

    # Write to a temporary file and rename into place: netCDF4 truncates the
    # target on open-for-write BEFORE the HDF5 lock check, so writing directly
    # onto a GLIDE_inputs.nc that another process (an inversion, a notebook)
    # holds open would leave an empty file behind when the lock is refused.
    tmp_path = output_path.with_suffix('.nc.tmp')
    merged.to_netcdf(tmp_path)
    os.replace(tmp_path, output_path)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_merged(args.domain_path)
