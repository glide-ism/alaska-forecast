"""Build the global mean temperature anomaly time series for a domain.

Splices the PAGES2k paleo reconstruction with the HadCRUT instrumental record:
PAGES2k is bias-corrected to HadCRUT over their 1850-1900 overlap, then the two
are linearly cross-faded across that window so the joined series is continuous.

Output: {domain_path}/model_inputs/temperature_anomaly.nc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HADCRUT_PATH = '../common_data/climate/temp_anomaly/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv'
PAGES_PATH = '../common_data/climate/temp_anomaly/pages2k_ngeo19_recons.nc'

OVERLAP_START = 1850
OVERLAP_END = 1900


def build_temperature_anomaly(domain_path: str) -> xr.DataArray:
    """Build the spliced anomaly series for `domain_path` and write to disk.

    Independent of the DEM; reads only common_data sources.
    """
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / 'temperature_anomaly.nc'

    hadcrut = pd.read_csv(HADCRUT_PATH)
    pages = xr.load_dataset(PAGES_PATH)

    pages_years = pages.year.values
    hadcrut_years = hadcrut.Time.values
    pages_anomaly = pages.DA.mean(axis=1)
    hadcrut_anomaly = hadcrut['Anomaly (deg C)'].values

    years = np.arange(
        min(pages_years.min(), hadcrut_years.min()),
        max(pages_years.max(), hadcrut_years.max()) + 1,
    )
    pages_interp = np.interp(years, pages_years, pages_anomaly)
    hadcrut_interp = np.interp(years, hadcrut_years, hadcrut_anomaly)

    # Bias-correct PAGES2k to HadCRUT over their overlap window
    overlap = (years >= OVERLAP_START) & (years <= OVERLAP_END)
    bias = np.mean(hadcrut_interp[overlap] - pages_interp[overlap])
    pages_aligned = pages_interp + bias

    # Linear cross-fade across the overlap so the join is continuous
    weight = np.clip((years - OVERLAP_START) / (OVERLAP_END - OVERLAP_START), 0.0, 1.0)
    spliced = np.where(
        years < OVERLAP_START,
        pages_aligned,
        np.where(
            years > OVERLAP_END,
            hadcrut_interp,
            (1 - weight) * pages_aligned + weight * hadcrut_interp,
        ),
    )

    anomaly = xr.DataArray(
        spliced,
        coords={"time": years},
        dims=["time"],
        name="temp_anomaly",
        attrs={
            "units": "degC",
            "description": "Global mean temperature anomaly (May-Apr year, PAGES2k + HadCRUT splice)",
        },
    )
    anomaly.to_netcdf(output_path)
    return anomaly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_temperature_anomaly(args.domain_path)
