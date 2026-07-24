"""
GlacierPriors: just the prior models (4 Matern fields + scalar SMB priors).

Cheap to build — does not construct IceDynamics. Used directly by posterior.py
to map RTO whitened samples back to physical space without standing up the
full forward model. Reused inside GlacierProblem.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from ggapp.model import MaternPrior

from .config import GlacierConfig, PriorHyperparams


def _build_matern_prior(p: PriorHyperparams, n_levels: int, ny: int, nx: int, dx: float) -> MaternPrior:
    m = MaternPrior(n_levels=n_levels, ny=ny, nx=nx, dx=dx)
    m.mg.parameters.sigma.set(p.sigma)
    m.mg.parameters.l.set(p.l)
    m.mg.parameters.nu.set(p.nu)
    m.forward_solver.fas_options.report_norms.set(False)
    return m


def domain_shape(config: GlacierConfig) -> tuple:
    """Open the gridded dataset, apply the same factor-aligned crop the full
    problem uses, and return (ny, nx, dx). Cheap — used by posterior.py to
    build only the priors without standing up IceDynamics.
    """
    gd = xr.open_dataset(Path(config.base_dir) / "model_inputs" / config.gridded_filename)
    factor = 2 ** config.n_levels
    ny0, nx0 = gd.sizes["y"], gd.sizes["x"]
    ny_target = (ny0 // factor) * factor
    nx_target = (nx0 // factor) * factor
    y_start = (ny0 - ny_target) // 2
    x_start = (nx0 - nx_target) // 2
    gd = gd.isel(y=slice(y_start, y_start + ny_target),
                 x=slice(x_start, x_start + nx_target))
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    dx = (gd.x[1] - gd.x[0]).item()
    return ny, nx, dx


class GlacierPriors:
    def __init__(self, config: GlacierConfig, ny: int, nx: int, dx: float):
        self.config = config
        self.ny = ny
        self.nx = nx
        self.dx = dx

        self.bed_model      = _build_matern_prior(config.bed_prior,      config.n_levels, ny, nx, dx)
        self.mean_model     = _build_matern_prior(config.mean_prior,     config.n_levels, ny, nx, dx)
        self.log_beta_model = _build_matern_prior(config.log_beta_prior, config.n_levels, ny, nx, dx)
        self.pbias_model    = _build_matern_prior(config.pbias_prior,    config.n_levels, ny, nx, dx)

        self.mu_log_rf = float(np.log(config.mu_rf))
        self.mu_log_mf = float(np.log(config.mu_mf))
        self.sigma_log_rf = config.sigma_log_rf
        self.sigma_log_mf = config.sigma_log_mf

        # Elevation-dependent precip depletion scalars (normal priors, directly
        # on tau and z0 — not log-normal; tau is itself a log length scale).
        self.mu_tau = config.mu_tau
        self.sigma_tau = config.sigma_tau
        self.mu_z0 = config.mu_z0
        self.sigma_z0 = config.sigma_z0
