"""Build the gridded monthly climatology (T2M, precip) for a domain.

Regrids pan-Arctic CARRA2 reanalysis fields onto the project DEM grid using
GLARE's PanCarraBase, applying a 3 K/km lapse rate to temperature.

Output: {domain_path}/model_inputs/gridded_climate.nc
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from glare import PanCarraBase

from projection_dictionary import crs


PANCARRA_BASE = Path('../common_data/climate/pancarra')
TEMPERATURE_LAPSE_RATE_K_PER_M = 0.003
ICE_DENSITY = 917.0
DAYS_PER_YEAR = 365


def build_climate(domain_path: str, year: int) -> xr.Dataset:
    """Build the gridded climate dataset for `domain_path`, `year` and write to disk."""
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_climate.nc'

    precip_path = PANCARRA_BASE / str(year) / 'precip' / 'precip.nc'
    t2m_path = PANCARRA_BASE / str(year) / 't2m' / 't2m.nc'
    orog_path = PANCARRA_BASE / 'topo' / 'topo.grib'

    dem = xr.load_dataset(dem_path)

    pancarra = PanCarraBase(precip_path, t2m_path, orog_path)
    _, t2m_fields, precip_fields = pancarra.regrid_carra2_fields(
        dem, crs, method='linear', t2m_lapse_rate=TEMPERATURE_LAPSE_RATE_K_PER_M,
    )

    months = np.arange(0, 12, dtype=np.float32) / 12
    coords = {"t": months, "y": dem.y, "x": dem.x}
    dims = ['t', 'y', 'x']

    t2m_da = xr.DataArray(
        t2m_fields, dims=dims, coords=coords,
        attrs={
            "units": "Deg C",
            "long_name": "Monthly average temperatures derived from pan-arctic CARRA2",
        },
    )

    # Convert mean precipitation rate (kg/m^2/s as ice equivalent) to m/yr.
    precip_da = xr.DataArray(
        precip_fields / ICE_DENSITY * DAYS_PER_YEAR, dims=dims, coords=coords,
        attrs={
            "units": "m ice equivalent / yr",
            "long_name": "Precipitation rate derived from pan-arctic CARRA2 at monthly time steps",
        },
    )

    climate_ds = dem.copy()
    climate_ds["monthly_t2m"] = t2m_da
    climate_ds["monthly_precip"] = precip_da

    for name in ('elevation', 'domain_mask', 'rgi_mask',
                 'topography', 'bathymetry', 'bathymetry_mask'):
        del climate_ds[name]

    climate_ds.to_netcdf(output_path)
    return climate_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    build_climate(args.domain_path, args.year)
