"""Merge per-variable gridded NetCDFs into a single GLIDE input file.

Reads gridded_dem, gridded_velocity, gridded_insolation, and gridded_climate
from {domain_path}/model_inputs and writes their xr.merge to GLIDE_inputs.nc.

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
    insolation = xr.load_dataset(inputs_dir / 'gridded_insolation.nc')
    climate = xr.load_dataset(inputs_dir / 'gridded_climate.nc')

    merged = xr.merge([geometry, velocity, insolation, climate])
    merged.to_netcdf(output_path)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_merged(args.domain_path)
