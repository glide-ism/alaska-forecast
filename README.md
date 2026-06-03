# alaska-forecast
Development repository for applying Glide/Glare to inverse modelling, forecasting, and data collection planning for all glaciers in Alaska by mountain range.

This is an experiment/orchestration workspace rather than an installable package. It depends on four custom libraries (`glide`, `glare`, `ggapp`, `gtic`, all under [github.com/glide-ism](https://github.com/glide-ism)) plus a scientific/geospatial Python stack.

## Setup

```
pip install -r requirements.txt
```

The geospatial (GDAL/PROJ) and GPU (CUDA via `cupy`/`torch`) dependencies can be awkward through pip; see the notes at the top of `requirements.txt`. If you are actively developing the `glide-ism` libraries alongside this repo, install them editable instead (`pip install -e ../glide`, etc.).

## Common data

As a first step, download the shared model-input bundles (the `common_data/` directory is gitignored and not checked in):

```
python download_common_data.py --extract
```

This pulls the latest input bundle from the default manifest and unpacks it. See `python download_common_data.py --help` for manifest, output-directory, and re-download options.

## Running a domain

Each mountain range is a domain under `domains/` (`chugach`, `delta`, `denali`, `juneau`, `st_elias`, `wrangell`).

1. **Preprocess** the domain's model inputs:

   ```
   python preprocessing/make_all.py --domain-path domains/wrangell
   ```

   This runs the full preprocessing pipeline (DEM, velocity, snowline, climate, etc.) and writes the results into `domains/wrangell/model_inputs/`. Pass `--year` to override the default (2012).

2. **Run the inversion.** Once preprocessing finishes, edit `inverse.py` to point at the same domain by setting the `DOMAIN` variable:

   ```python
   DOMAIN = "domains/wrangell"
   ```

   then run it:

   ```
   python inverse.py
   ```
