import numpy as np
import pandas as pd
import xarray as xr

HADCRUT_PATH = '../data/climate/temp_anomaly/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv'
PAGES_PATH = '../data/climate/temp_anomaly/pages2k_ngeo19_recons.nc'

OUTPUT_PATH = './model_inputs/temperature_anomaly.nc'

hadcrut = pd.read_csv(HADCRUT_PATH)
pages = xr.load_dataset(PAGES_PATH)

pages_years = pages.year.values
hadcrut_years = hadcrut.Time.values

pages_delta = pages.DA.mean(axis=1)
hadcrut_delta = hadcrut['Anomaly (deg C)'].values

years = np.arange(min(pages_years.min(),hadcrut_years.min()),max(pages_years.max(),hadcrut_years.max())+1)

years = np.arange(
    min(pages_years.min(), hadcrut_years.min()),
    max(pages_years.max(), hadcrut_years.max()) + 1
)

pages_interp = np.interp(years, pages_years, pages_delta)
had_interp   = np.interp(years, hadcrut_years, hadcrut_delta)

overlap = (years >= 1850) & (years <= 1900)

delta = np.mean(had_interp[overlap] - pages_interp[overlap])
pages_aligned = pages_interp + delta

y0, y1 = 1850, 1900
w = np.clip((years - y0) / (y1 - y0), 0.0, 1.0)

merged = np.where(
    years < y0,
    pages_aligned,
    np.where(
        years > y1,
        had_interp,
        (1 - w) * pages_aligned + w * had_interp
    )
)
xr.DataArray(
    merged,
    coords={"time": years},
    dims=["time"],
    name="temp_anomaly",
    attrs={"units": "degC", "description": "Global mean temperature anomaly (May-Apr year, PAGES2k + HadCRUT splice)"}
).to_netcdf(OUTPUT_PATH)

