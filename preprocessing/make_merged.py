"""Merge per-variable gridded NetCDFs into a single GLIDE input file.

Reads gridded_dem, gridded_velocity, gridded_snowline, gridded_debris,
gridded_dhdt, gridded_insolation, gridded_climate (CARRA) and
gridded_climate_era5land from {domain_path}/model_inputs and writes their
xr.merge to GLIDE_inputs.nc.
Both climatologies are included to facilitate source intercomparison.

Output: {domain_path}/model_inputs/GLIDE_inputs.nc
"""
import argparse
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
    merged.to_netcdf(output_path)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_merged(args.domain_path)
