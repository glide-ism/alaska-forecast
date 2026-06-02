"""Build the precipitation anomaly time series for a domain.

Reads the Mt. Hunter (Denali, AK) ice-core annual accumulation record from
common_data/climate/precip_anomaly/mt_hunter.txt and Gaussian-smooths it in
the time dimension to suppress year-to-year layer-counting noise.

The output is the smoothed annual accumulation series in metres of water
equivalent. Normalisation to a reference year (so e.g. 2012 = 1.0) happens
at runtime in the inverse problem via `cfg.base_precip_year`, mirroring how
`temperature_anomaly.nc` is normalised against `cfg.base_anomaly_year`.

Output: {domain_path}/model_inputs/precip_anomaly.nc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d


MT_HUNTER_PATH = '../common_data/climate/precip_anomaly/mt_hunter.txt'
DEFAULT_SMOOTHING_SIGMA = 10.0  # years


def _read_mt_hunter(path: str) -> pd.DataFrame:
    """Return a year-indexed DataFrame of annual accumulation (m water eq).

    The NOAA file has a long commented header followed by a tab-separated
    table with columns: age_CE, accum_ann, accum-MJJA, accum-SONDJFMA. Only
    accum_ann is fully populated, so that is what we use.
    """
    df = pd.read_csv(path, sep='\t', comment='#', engine='python')
    df = df.rename(columns={'age_CE': 'year', 'accum_ann': 'accum'})
    df = df[['year', 'accum']].dropna()
    df = df.sort_values('year').reset_index(drop=True)
    return df


def build_precip_anomaly(domain_path: str,
                         smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA
                         ) -> xr.DataArray:
    """Build the smoothed Mt. Hunter accumulation series for `domain_path`.

    `smoothing_sigma`: Gaussian kernel width in years (set to 0 to disable).
    """
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / 'precip_anomaly.nc'

    df = _read_mt_hunter(MT_HUNTER_PATH)
    years = df.year.values.astype(int)
    accum = df.accum.values.astype('float64')

    # The record is annual and contiguous — confirm before smoothing in case
    # the file ever changes; gaussian_filter1d assumes uniform spacing.
    if not np.all(np.diff(years) == 1):
        raise ValueError(
            "Mt. Hunter record is not contiguous-annual; smoothing assumes "
            "uniform 1-year spacing."
        )

    if smoothing_sigma > 0:
        smoothed = gaussian_filter1d(accum, sigma=smoothing_sigma, mode='nearest')
    else:
        smoothed = accum

    anomaly = xr.DataArray(
        smoothed.astype('float32'),
        coords={'time': years},
        dims=['time'],
        name='precip_anomaly',
        attrs={
            'units': 'm water equivalent / yr',
            'description': (
                'Annual accumulation from the Mt. Hunter (Denali, AK) ice '
                'core, Gaussian-smoothed in time. Normalised against a base '
                'year at runtime to form a multiplicative precip scaling.'
            ),
            'source': 'NOAA Paleoclimatology Study 21970 (Winski et al. 2017)',
            'smoothing_sigma_years': float(smoothing_sigma),
        },
    )
    anomaly.to_netcdf(output_path)
    return anomaly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--smoothing-sigma", type=float,
                        default=DEFAULT_SMOOTHING_SIGMA,
                        help="Gaussian kernel sigma in years (0 disables).")
    args = parser.parse_args()
    build_precip_anomaly(args.domain_path,
                         smoothing_sigma=args.smoothing_sigma)
