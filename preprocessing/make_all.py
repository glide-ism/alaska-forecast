"""Run the full preprocessing pipeline for a single domain.

Calls each per-variable build function in dependency order:

    DEM ─┬─ flightlines
         ├─ velocity
         ├─ insolation (needs year)
         └─ climate    (needs year)
    temperature anomaly  (independent)
    merged = DEM + velocity + insolation + climate

Run as a script:
    python make_all.py --domain-path ../domains/chugach --year 2012
"""
import argparse
from pathlib import Path

from make_dem import build_dem
from make_bedradar import build_flightlines
from make_velocity import build_velocity
from make_insolation import build_insolation
from make_pancarra_vars import build_climate
from make_temperature_anomaly import build_temperature_anomaly
from make_merged import build_merged


def _banner(step: str) -> None:
    print(f"\n=== {step} ===", flush=True)


def run_all(domain_path: str, year: int) -> None:
    """Execute every preprocessing step for `domain_path` in dependency order."""
    Path(domain_path, 'model_inputs').mkdir(parents=True, exist_ok=True)

    _banner("DEM")
    build_dem(domain_path)

    _banner("Flightlines")
    build_flightlines(domain_path)

    _banner("Velocity")
    build_velocity(domain_path)

    _banner(f"Insolation (year={year})")
    build_insolation(domain_path, year=year)

    _banner(f"Climate (year={year})")
    build_climate(domain_path, year=year)

    _banner("Temperature anomaly")
    build_temperature_anomaly(domain_path)

    _banner("Merge")
    build_merged(domain_path)

    _banner("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--year", type=int, default=2012)
    args = parser.parse_args()
    run_all(args.domain_path, args.year)
