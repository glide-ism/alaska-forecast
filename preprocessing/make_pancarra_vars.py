"""Build the gridded monthly climatology (T2M, precip) for a domain.

Regrids pan-Arctic CARRA2 reanalysis fields onto the project DEM grid using
GLARE's PanCarraBase for the projection/interpolation, then applies a 3 K/km
lapse rate to temperature.

The temperature lapse correction uses a smoothed-DEM proxy for the model
surface (the project DEM low-pass filtered to ~CARRA's grid scale) rather
than CARRA's topo.grib orography. This matches make_era5land_vars.py so the
two climatologies differ only in their source data, not in how they are
downscaled.

An optional per-month median filter can be applied to precipitation to damp
spatial speckle in the CARRA precip field.

If `multiyear=True`, the source fields come from the multi-year file at
common_data/climate/pancarra/1986_2023 and are reduced to a 12-month
climatology by grouping each timestep on its valid_time month (robust to
the file's storage phase and to gaps in year coverage; `year` is then only
used for output bookkeeping).

Output: {domain_path}/model_inputs/gridded_climate.nc
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from glare import PanCarraBase
from scipy.ndimage import gaussian_filter, median_filter

from projection_dictionary import crs


PANCARRA_BASE = Path('../common_data/climate/pancarra')
MULTIYEAR_BASE = PANCARRA_BASE / '1986_2023'
TEMPERATURE_LAPSE_RATE_K_PER_M = -0.0065
ICE_DENSITY = 917.0
DAYS_PER_YEAR = 365

# CARRA2 native grid spacing (PanCarraBase.Dx) in metres; used as the
# smoothing length that approximates the model surface t2m "sees".
CARRA_GRID_M = 2500.0


def _calendar_month_climatology(ds: xr.Dataset, varname: str) -> xr.Dataset:
    """Reduce `ds[varname]` to a (12, Ny, Nx) calendar-month climatology.

    Groups along the original time dim by valid_time.dt.month so the result
    is correct regardless of where the file starts or whether year coverage
    is contiguous. The new leading dim is named 'time' and ordered Jan..Dec
    so PanCarraBase.interpolate's positional `dataset[time_index]` indexing
    still corresponds to month-of-year.
    """
    da = ds[varname]
    month = ds.valid_time.dt.month
    grouped = da.groupby(month).mean(dim=da.dims[0]).sortby('month')
    grouped = grouped.rename({'month': 'time'})
    return grouped.to_dataset(name=varname)


def build_climate(domain_path: str, year: int,
                  precip_median_window: int = None,
                  multiyear: bool = False) -> xr.Dataset:
    """Build the gridded climate dataset for `domain_path`, `year` and write to disk.

    `precip_median_window`: if set (> 1), the side length in grid cells of a
    square median filter applied to each monthly precipitation field.

    `multiyear`: if True, replace the single-year CARRA fields with a
    calendar-month climatology from common_data/climate/pancarra/1986_2023.
    """
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_climate.nc'

    if multiyear:
        precip_path = MULTIYEAR_BASE / 'precip' / 'precip.nc'
        t2m_path = MULTIYEAR_BASE / 't2m' / 't2m.nc'
    else:
        precip_path = PANCARRA_BASE / str(year) / 'precip' / 'precip.nc'
        t2m_path = PANCARRA_BASE / str(year) / 't2m' / 't2m.nc'
    orog_path = PANCARRA_BASE / 'topo' / 'topo.grib'

    dem = xr.load_dataset(dem_path)
    elevation = dem.elevation.values.astype('float64')

    # PanCarraBase still needs an orography path at construction, but we no
    # longer use it for the lapse correction (smoothed DEM is used instead).
    pancarra = PanCarraBase(precip_path, t2m_path, orog_path)

    if multiyear:
        print("Reducing multi-year CARRA to calendar-month climatology")
        pancarra.precip_dataset = _calendar_month_climatology(
            pancarra.precip_dataset, 'tp')
        pancarra.temp_dataset = _calendar_month_climatology(
            pancarra.temp_dataset, 't2m')

    # DEM grid -> CARRA2 projected query points.
    grid_x, grid_y = np.meshgrid(dem.x.values, dem.y.values)
    query_x, query_y = pancarra.transform_to(grid_x, grid_y, crs)

    # Smoothed-DEM proxy for the CARRA model surface (consistent with
    # make_era5land_vars.py; only the smoothing length differs by source).
    grid_res_m = float(abs(dem.x.values[1] - dem.x.values[0]))
    sigma_px = CARRA_GRID_M / grid_res_m
    z_model = gaussian_filter(elevation, sigma=sigma_px, mode='nearest')

    print("Working on t2m fields")
    t2m_fields = np.stack(
        [
            pancarra.interpolate(query_x, query_y, time_index=i, key='t2m',
                                 method='linear')
            - 273
            + TEMPERATURE_LAPSE_RATE_K_PER_M * (elevation - z_model)
            for i in range(12)
        ],
        axis=0,
    ).astype('float32')

    print("Working on precip fields")
    # CARRA precip 'tp' is a mean rate (kg/m^2/s ice equivalent); convert to
    # m ice equivalent / yr.
    precip_fields = np.stack(
        [
            pancarra.interpolate(query_x, query_y, time_index=i, key='precip',
                                 method='linear')
            / ICE_DENSITY * DAYS_PER_YEAR
            for i in range(12)
        ],
        axis=0,
    ).astype('float32')

    if precip_median_window and precip_median_window > 1:
        print(f"Median-filtering precip (window={precip_median_window})")
        precip_fields = np.stack(
            [median_filter(precip_fields[i], size=precip_median_window)
             for i in range(12)],
            axis=0,
        )

    months = np.arange(0, 12, dtype=np.float32) / 12
    coords = {"t": months, "y": dem.y, "x": dem.x}
    dims = ['t', 'y', 'x']

    source_tag = ("CARRA2 multi-year monthly climatology (1986_2023)"
                  if multiyear else f"CARRA2 monthly, year {year}")

    t2m_da = xr.DataArray(
        t2m_fields, dims=dims, coords=coords,
        attrs={
            "units": "Deg C",
            "long_name": "Monthly average temperatures derived from pan-arctic CARRA2",
            "source": source_tag,
        },
    )
    precip_da = xr.DataArray(
        precip_fields, dims=dims, coords=coords,
        attrs={
            "units": "m ice equivalent / yr",
            "long_name": "Precipitation rate derived from pan-arctic CARRA2 at monthly time steps",
            "median_filter_window": precip_median_window or 0,
            "source": source_tag,
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
    parser.add_argument("--precip-median-window", type=int, default=None,
                        help="Side length (grid cells) of an optional square "
                             "median filter applied to monthly precip.")
    parser.add_argument("--multiyear", action="store_true", default=False,
                        help="Use the multi-year CARRA file in "
                             "common_data/climate/pancarra/1986_2023, reduced "
                             "to a calendar-month climatology.")
    args = parser.parse_args()
    build_climate(args.domain_path, args.year,
                  precip_median_window=args.precip_median_window,
                  multiyear=args.multiyear)
