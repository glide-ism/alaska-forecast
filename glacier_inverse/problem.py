"""
GlacierProblem: builds the full forward model (IceDynamics + SMB + priors +
observations + whitened parameter tensors) from a GlacierConfig.

A driver script constructs one GlacierProblem and then either:
  * optimizes its `params` against `loss(...)` (inverse, rto),
  * loads RTO samples into its `params` and computes `loss_volume_change(...)`
    for the sensitivity ensemble,
  * inspects its priors only (posterior — use GlacierPriors directly there).

The loss and forward methods take an `Observations` object so RTO can pass a
randomized copy without rebuilding the problem.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cupy as cp
import geopandas as gpd
import numpy as np
import pyproj
import torch
import xarray as xr

from glide.model import IceDynamics
from glare.model import ImprovedTemperatureIndex
from glare.avalanche import AvalancheOperator
from ggapp.torch import GGaPPMap, GGaPPWhiten

from .config import GlacierConfig, SolverConfig
from .forward import simulate, differentiable_restriction, differentiable_prolongation
from .loss import LossTerms, PriorMeans, compute_data_loss, compute_prior
from .priors import GlacierPriors


@dataclass
class Observations:
    u_obs: torch.Tensor
    v_obs: torch.Tensor
    v_mask: torch.Tensor
    S_obs: torch.Tensor         # ice/water surface: max(DEM, 0)
    dem: torch.Tensor           # full DEM (topography + bathymetry); the
                                # off-ice bed anchor
    bed_obs: torch.Tensor
    bed_normed_coords: torch.Tensor
    rgi_mask: torch.Tensor      # 0/1 ice extent
    rgi_label: torch.Tensor
    surge_type: torch.Tensor
    obs_mask: torch.Tensor
    domain_mask: torch.Tensor   # 0/1 simulation domain
    # End-of-summer snowline (ELA proxy). Optional: domains without the
    # product leave these None and the snowline loss term is skipped.
    snow_label: Optional[torch.Tensor] = None   # snow fraction in [0, 1]
    snow_mask: Optional[torch.Tensor] = None    # 0/1 valid (on-ice) cells
    debris: Optional[torch.Tensor] = None
    # Surface elevation-change rate (Hugonnet dH/dt). Optional: domains without
    # the product leave these None and the dH/dt loss term is skipped.
    dhdt: Optional[torch.Tensor] = None         # dH/dt (m/yr)
    dhdt_err: Optional[torch.Tensor] = None     # per-pixel 1-sigma uncertainty (m/yr)
    dhdt_mask: Optional[torch.Tensor] = None    # 0/1 valid (finite, on-domain) cells

    def randomized(
        self,
        sigma_u: float,
        sigma_s: float,
        sigma_bed: float,
        *,
        eps_u: Optional[torch.Tensor] = None,
        eps_v: Optional[torch.Tensor] = None,
        eps_S: Optional[torch.Tensor] = None,
        eps_bed: Optional[torch.Tensor] = None,
    ) -> "Observations":
        """Return a copy with N(0, sigma^2) noise added.

        If `eps_*` whitened noise tensors are supplied they are used in place
        of fresh `torch.randn_like` draws — this is how the RTO driver injects
        QMC noise instead of standard Monte Carlo.
        """
        eps_u   = torch.randn_like(self.u_obs)   if eps_u   is None else eps_u
        eps_v   = torch.randn_like(self.v_obs)   if eps_v   is None else eps_v
        eps_S   = torch.randn_like(self.S_obs)   if eps_S   is None else eps_S
        eps_bed = torch.randn_like(self.bed_obs) if eps_bed is None else eps_bed
        return Observations(
            u_obs=self.u_obs + eps_u * sigma_u,
            v_obs=self.v_obs + eps_v * sigma_u,
            v_mask=self.v_mask,
            S_obs=self.S_obs + eps_S * sigma_s,
            # The off-ice bed anchor shares eps_S with the surface term (as it
            # did when this term read S_obs directly), so RTO noise bookkeeping
            # is unchanged.
            dem=self.dem + eps_S * sigma_s,
            bed_obs=self.bed_obs + eps_bed * sigma_bed,
            bed_normed_coords=self.bed_normed_coords,
            rgi_mask=self.rgi_mask,
            rgi_label=self.rgi_label,
            surge_type=self.surge_type,
            obs_mask=self.obs_mask,
            domain_mask=self.domain_mask,
            # The snowline label is a categorical observation with no
            # Gaussian noise model, so RTO passes it through unperturbed.
            snow_label=self.snow_label,
            snow_mask=self.snow_mask,
            debris=self.debris,
            # dH/dt has a Gaussian noise model, but RTO perturbation is added by
            # the (not-yet-wired) loss term; pass it through unperturbed for now.
            dhdt=self.dhdt,
            dhdt_err=self.dhdt_err,
            dhdt_mask=self.dhdt_mask,
        )


class WhitenedParameters:
    """The eight tensors we optimize over: four whitened fields plus four
    whitened scalars (log melt/radiation factors and the precip-depletion
    tau/z0). The depletion scalars are inert unless precip_lapse_enabled."""

    def __init__(self, z_bed, z_bed_mean, z_log_beta, z_pbias, z_log_mf, z_log_rf,
                 z_tau, z_z0):
        self.z_bed = z_bed
        self.z_bed_mean = z_bed_mean
        self.z_log_beta = z_log_beta
        self.z_pbias = z_pbias
        self.z_log_mf = z_log_mf
        self.z_log_rf = z_log_rf
        self.z_tau = z_tau
        self.z_z0 = z_z0

    def requires_grad_(self) -> "WhitenedParameters":
        for t in (self.z_bed, self.z_bed_mean, self.z_log_beta,
                  self.z_pbias, self.z_log_mf, self.z_log_rf,
                  self.z_tau, self.z_z0):
            t.requires_grad_()
        return self

    def detach_clone(self) -> "WhitenedParameters":
        return WhitenedParameters(
            z_bed=self.z_bed.detach().clone(),
            z_bed_mean=self.z_bed_mean.detach().clone(),
            z_log_beta=self.z_log_beta.detach().clone(),
            z_pbias=self.z_pbias.detach().clone(),
            z_log_mf=self.z_log_mf.detach().clone(),
            z_log_rf=self.z_log_rf.detach().clone(),
            z_tau=self.z_tau.detach().clone(),
            z_z0=self.z_z0.detach().clone(),
        )

    @torch.no_grad()
    def copy_from(self, other: "WhitenedParameters") -> None:
        """In-place reset to `other`'s values. Tensor identities are preserved
        so any optimizer holding references to these tensors keeps working."""
        self.z_bed.copy_(other.z_bed)
        self.z_bed_mean.copy_(other.z_bed_mean)
        self.z_log_beta.copy_(other.z_log_beta)
        self.z_pbias.copy_(other.z_pbias)
        self.z_log_mf.copy_(other.z_log_mf)
        self.z_log_rf.copy_(other.z_log_rf)
        self.z_tau.copy_(other.z_tau)
        self.z_z0.copy_(other.z_z0)


def _apply_solver_settings(target_fas, cfg: SolverConfig) -> None:
    target_fas.set(
        coarsest_steps=cfg.coarsest_steps,
        pre_steps=cfg.pre_steps,
        post_steps=cfg.post_steps,
        finest_steps=cfg.finest_steps,
        relative_tolerance=cfg.relative_tolerance,
        absolute_tolerance=cfg.absolute_tolerance,
        report_norms=cfg.report_norms,
    )


def _crop_to_factor(gridded_data: xr.Dataset, factor: int) -> xr.Dataset:
    """Crop the gridded dataset so ny and nx are multiples of 2^(n_levels-?)."""
    ny0, nx0 = gridded_data.sizes["y"], gridded_data.sizes["x"]
    ny_target = (ny0 // factor) * factor
    nx_target = (nx0 // factor) * factor
    y_start = (ny0 - ny_target) // 2
    x_start = (nx0 - nx_target) // 2
    return gridded_data.isel(
        y=slice(y_start, y_start + ny_target),
        x=slice(x_start, x_start + nx_target),
    )


class GlacierProblem:
    """Full forward model + observations + whitened parameters."""

    def __init__(self, config: GlacierConfig):
        self.config = config
        cfg = config

        inputs_dir = Path(cfg.base_dir) / "model_inputs"
        self.gridded_data = xr.open_dataset(inputs_dir / cfg.gridded_filename)
        self.temperature_anomaly = xr.open_dataset(inputs_dir / cfg.anomaly_filename)
        flightlines_df = gpd.read_file(inputs_dir / cfg.flightline_filename)

        self.crs = pyproj.CRS(self.gridded_data.spatial_ref.crs_wkt)
        self.gridded_data = _crop_to_factor(self.gridded_data, 2 ** cfg.n_levels)

        # Optional end-of-summer snowline product (ELA proxy). Built on the
        # same DEM grid as GLIDE_inputs, so the identical centered crop keeps
        # it aligned. Domains without the file simply skip the snowline loss.
        snowline_path = inputs_dir / cfg.snowline_filename
        if snowline_path.exists():
            self.snowline_data = _crop_to_factor(
                xr.open_dataset(snowline_path), 2 ** cfg.n_levels)
        else:
            self.snowline_data = None

        debris_path = inputs_dir / cfg.debris_filename
        if debris_path.exists():
            self.debris_data = _crop_to_factor(
                xr.open_dataset(debris_path), 2 ** cfg.n_levels)
        else:
            self.debris_data = None

        # Optional surface elevation-change rate (Hugonnet dH/dt). Built on the
        # same DEM grid as GLIDE_inputs, so the identical centered crop keeps it
        # aligned. Domains without the file simply skip the dH/dt loss.
        dhdt_path = inputs_dir / cfg.dhdt_filename
        if dhdt_path.exists():
            self.dhdt_data = _crop_to_factor(
                xr.open_dataset(dhdt_path), 2 ** cfg.n_levels)
        else:
            self.dhdt_data = None

        ny, nx = self.gridded_data.sizes["y"], self.gridded_data.sizes["x"]
        dx = (self.gridded_data.x[1] - self.gridded_data.x[0]).item()
        x0 = self.gridded_data.x[0].item()
        y0 = self.gridded_data.y[0].item()
        self.ny, self.nx, self.dx = ny, nx, dx

        self.model = self._build_ice_dynamics(ny, nx, dx, x0, y0)
        self.mg = self.model.mg

        self.smb_model = self._build_smb_model(ny, nx, dx, x0, y0)
        self.mg.forcing.smb.set(self.smb_model.grid.state.smb.data.mean(axis=0))

        self.priors = GlacierPriors(cfg, ny, nx, dx)
        self.observations = self._build_observations(flightlines_df)
        self.params = self._build_initial_parameters()

        self.base_anomaly = self.temperature_anomaly.sel(
            time=cfg.base_anomaly_year).temp_anomaly.item()
        self.alpha_t2m = torch.tensor(cfg.alpha_t2m, device="cuda")

        # Optional ice-core precip anomaly: applied as a multiplicative
        # scaling on the precip field at each time step.
        precip_anomaly_path = inputs_dir / cfg.precip_anomaly_filename
        if precip_anomaly_path.exists():
            self.precip_anomaly = xr.open_dataset(precip_anomaly_path)
            self.base_precip = self.precip_anomaly.sel(
                time=cfg.base_precip_year).precip_anomaly.item()
            self.alpha_precip = torch.tensor(cfg.alpha_precip, device="cuda")
        else:
            self.precip_anomaly = None
            self.base_precip = None
            self.alpha_precip = None

        # Cached fine-grid forcings/state used in every forward call.
        self.t2m = torch.tensor(self.smb_model.grid.temperature.t2m.data,
                                device="cuda")
        self.precip = torch.tensor(self.smb_model.grid.precipitation.precip.data,
                                   device="cuda")
        self.H_prev = torch.tensor(self.mg[0].state.H_prev.data,
                                   device="cuda")

        self.debris = torch.tensor(self.smb_model.grid.geometry.debris.data,
                                   device="cuda")
    # ----------------------------------------------------------------- builders

    def _build_ice_dynamics(self, ny, nx, dx, x0, y0) -> IceDynamics:
        cfg = self.config
        model = IceDynamics(n_levels=cfg.n_levels, ny=ny, nx=nx, dx=dx,
                            x0=x0, y0=y0, crs=self.crs)
        mg = model.mg

        mg.state.H.set(0.1)
        mg.state.H_prev.set(0.1)

        mg.geometry.bed.set(self.gridded_data.elevation)
        mg.geometry.depth.set(np.maximum(-self.gridded_data.elevation, 0))
        mg.geometry.sigmoid_c.set(cfg.sigmoid_c)
        mg.geometry.sigmoid_k.set(cfg.sigmoid_k)

        mg.rheology.B.set(cfg.B_rate)
        mg.rheology.eps_reg.set(cfg.eps_reg)
        mg.rheology.n.set(float(cfg.n_glen))

        mg.sliding.beta.set(cfg.beta_init)
        mg.sliding.m.set(cfg.sliding_m)
        mg.sliding.water_drag.set(cfg.water_drag)

        mg.calving.calving_rate.set(cfg.calving_rate)

        _apply_solver_settings(model.forward_solver.fas_options, cfg.forward_solver)
        _apply_solver_settings(model.adjoint_solver.fas_options, cfg.adjoint_solver)
        model.adjoint_solver.vanka_options.newton_options.ssa_damping.set(
            cp.float32(cfg.ssa_damping))
        return model

    def _build_smb_model(self, ny, nx, dx, x0, y0) -> ImprovedTemperatureIndex:
        smb = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=12,
                                       dx=dx, dt=1.0 / 12,
                                       x0=x0, y0=y0, crs=self.crs)

        if self.config.use_avalanche_model:
            smb.avalanche = AvalancheOperator(smb.grid,
                    s_crit=30.0, w_trans=8.0, p=1.5, K=25)

        gd = self.gridded_data
        smb.grid.insolation.insol_mean.set(gd.monthly_solar_potential_mean)
        smb.grid.insolation.insol_cos.set(gd.monthly_solar_potential_cos)
        smb.grid.insolation.insol_sin.set(gd.monthly_solar_potential_sin)
        smb.grid.temperature.t2m.set(gd.monthly_t2m)
        smb.grid.precipitation.precip.set(gd.monthly_precip)
        smb.grid.geometry.srf.set(gd.elevation)
        smb.grid.insolation.rf.set(self.config.mu_rf)
        smb.grid.temperature.mf.set(self.config.mu_mf)
        smb.grid.geometry.debris.set(1.0 - self.config.debris_factor*self.debris_data.debris_fraction)
        smb.forward()
        return smb

    def _build_observations(self, flightlines_df) -> Observations:
        gd = self.gridded_data
        domain_mask = torch.tensor(gd.domain_mask.values, device="cuda")
        rgi_mask = torch.tensor(gd.rgi_mask.values, device="cuda")
        rgi_label = torch.tensor(gd.rgi_label.values, device="cuda",dtype=torch.long)
        rgi_label[torch.isnan(rgi_label)] = -1
        surge_type = torch.tensor(gd.surge_type.values,device="cuda",dtype=torch.long)
        obs_mask = rgi_mask

        u_obs = torch.tensor(gd.vx.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_obs = torch.tensor(gd.vy.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_mask = torch.tensor(gd.vmask.values,dtype=torch.float32,device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        dem = torch.tensor(gd.elevation.values, dtype=torch.float32, device="cuda")
        S_obs = torch.clamp(dem, min=0.0)

        flightlines = torch.tensor(flightlines_df[["x", "y", "bed"]].values,
                                    dtype=torch.float32, device="cuda")
        xmin, xmax = gd.x.min().item(), gd.x.max().item()
        ymin, ymax = gd.y.min().item(), gd.y.max().item()
        col_normed = 2.0 * ((flightlines[:, 0] - xmin) / (xmax - xmin)) - 1
        row_normed = -(2.0 * ((flightlines[:, 1] - ymin) / (ymax - ymin)) - 1)
        bed_normed_coords = torch.stack([col_normed, row_normed], dim=-1)
        bed_obs = flightlines[:, 2]

        snow_label, snow_mask = self._build_snowline(domain_mask)
        dhdt, dhdt_err, dhdt_mask = self._build_dhdt(domain_mask)

        return Observations(
            u_obs=u_obs, v_obs=v_obs, v_mask=v_mask, S_obs=S_obs, dem=dem,
            bed_obs=bed_obs, bed_normed_coords=bed_normed_coords,
            rgi_mask=rgi_mask, rgi_label=rgi_label, surge_type=surge_type,
            obs_mask=obs_mask, domain_mask=domain_mask,
            snow_label=snow_label, snow_mask=snow_mask,
            dhdt=dhdt, dhdt_err=dhdt_err, dhdt_mask=dhdt_mask,
        )

    def _build_snowline(self, domain_mask: torch.Tensor):
        """Snow-fraction label and validity mask on the (cropped) grid.

        `snow_fraction` is the fraction of the glacierized subarea classified
        as retained snow (≈ accumulation area / above the ELA). It is only
        defined where the product classified ice at all (`glacier_fraction`
        > 0), so that — intersected with the simulation domain — is the BCE
        validity mask. Returns (None, None) when the domain has no product.
        """
        sd = self.snowline_data
        if sd is None:
            return None, None

        expected = (self.ny, self.nx)
        if sd.snow_fraction.shape != expected:
            raise ValueError(
                f"snowline grid {tuple(sd.snow_fraction.shape)} does not match "
                f"the cropped model grid {expected}; rebuild "
                f"{self.config.snowline_filename} on the DEM grid."
            )

        snow_label = torch.tensor(
            sd.snow_fraction.values, dtype=torch.float32, device="cuda"
        ).nan_to_num().clamp_(0.0, 1.0)
        glacier_fraction = torch.tensor(
            sd.glacier_fraction.values, dtype=torch.float32, device="cuda"
        ).nan_to_num()
        snow_mask = ((glacier_fraction > 0.0) & domain_mask).to(torch.float32)
        snow_label = snow_label.masked_fill(snow_mask == 0.0, 0.0)
        return snow_label, snow_mask

    def _build_dhdt(self, domain_mask: torch.Tensor):
        """Surface elevation-change rate, its uncertainty, and a validity mask
        on the (cropped) grid.

        The Hugonnet product only covers glacierized terrain, so cells are NaN
        off-glacier. Valid cells are those with a finite rate *and* finite
        uncertainty, intersected with the simulation domain; that intersection
        is the mask the (future) dH/dt misfit will reduce over. Returns
        (None, None, None) when the domain has no product.
        """
        dd = self.dhdt_data
        if dd is None:
            return None, None, None

        expected = (self.ny, self.nx)
        if dd.dhdt.shape != expected:
            raise ValueError(
                f"dhdt grid {tuple(dd.dhdt.shape)} does not match the cropped "
                f"model grid {expected}; rebuild {self.config.dhdt_filename} "
                f"on the DEM grid."
            )

        dhdt_raw = torch.tensor(
            dd.dhdt.values, dtype=torch.float32, device="cuda")
        dhdt_err_raw = torch.tensor(
            dd.dhdt_err.values, dtype=torch.float32, device="cuda")
        valid = torch.isfinite(dhdt_raw) & torch.isfinite(dhdt_err_raw)
        dhdt_mask = (valid & domain_mask).to(torch.float32)
        dhdt = dhdt_raw.nan_to_num().masked_fill(dhdt_mask == 0.0, 0.0)
        dhdt_err = dhdt_err_raw.nan_to_num().masked_fill(dhdt_mask == 0.0, 0.0)
        return dhdt, dhdt_err, dhdt_mask

    def _bed_obs_raster(self) -> torch.Tensor:
        """Scatter the scattered flightline bed picks onto the finest grid.

        Uses the same normalized coordinates the loss feeds to grid_sample
        (align_corners=False), so the raster lines up with what the bed misfit
        actually sees. Cells with no flightline coverage are left NaN.
        """
        obs = self.observations
        ny, nx = self.ny, self.nx
        cn = obs.bed_normed_coords[:, 0]
        rn = obs.bed_normed_coords[:, 1]
        col = (((cn + 1.0) * nx - 1.0) / 2.0).round().long().clamp_(0, nx - 1)
        row = (((rn + 1.0) * ny - 1.0) / 2.0).round().long().clamp_(0, ny - 1)
        raster = torch.full((ny, nx), float("nan"),
                            dtype=torch.float32, device="cuda")
        raster[row, col] = obs.bed_obs.to(torch.float32)
        return raster

    def write_observations(self, output_dir: str,
                           base: str = "observations") -> None:
        """Dump the observational products to a single-frame PVD.

        Writes velocity, surface, DEM, snowline, debris, the extent/domain
        masks, and the (rasterized) flightline bed picks at the finest grid.
        Call this once at the start of a run, pointed at the same directory
        the per-iteration diagnostics go to.
        """
        from .io import write_static_vti

        obs = self.observations
        scalars = {
            "srf_obs": obs.S_obs,
            "dem": obs.dem,
            "rgi_mask": obs.rgi_mask,
            "obs_mask": obs.obs_mask,
            "domain_mask": obs.domain_mask,
            "bed_obs": self._bed_obs_raster(),
        }
        if obs.snow_label is not None:
            scalars["snow_label"] = obs.snow_label
            scalars["snow_mask"] = obs.snow_mask
        if obs.dhdt is not None:
            scalars["dhdt"] = obs.dhdt
            scalars["dhdt_err"] = obs.dhdt_err
            scalars["dhdt_mask"] = obs.dhdt_mask
        if self.debris_data is not None:
            scalars["debris_fraction"] = torch.tensor(
                self.debris_data.debris_fraction.values,
                dtype=torch.float32, device="cuda")
        write_static_vti(
            self.mg[0], output_dir, base,
            scalar_fields=scalars,
            vector_fields={"U_obs": (obs.u_obs, obs.v_obs)},
        )

    def _build_initial_parameters(self) -> WhitenedParameters:
        priors = self.priors
        ny, nx = self.ny, self.nx

        log_beta = torch.tensor(cp.log(self.mg[0].sliding.beta.data), device="cuda")
        z_log_beta = GGaPPWhiten.apply(priors.log_beta_model, log_beta)

        bed = torch.tensor(self.mg[0].geometry.bed.data, device="cuda")
        bed_mean = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")
        z_bed = GGaPPWhiten.apply(priors.bed_model, bed - bed_mean)
        z_bed_mean = GGaPPWhiten.apply(priors.mean_model, bed_mean)

        z_pbias = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")

        log_mf = torch.tensor(cp.log(self.smb_model.grid.temperature.mf.value), device="cuda")
        log_rf = torch.tensor(cp.log(self.smb_model.grid.insolation.rf.value), device="cuda")
        z_log_mf = (log_mf - priors.mu_log_mf) / priors.sigma_log_mf
        z_log_rf = (log_rf - priors.mu_log_rf) / priors.sigma_log_rf

        # Precip-depletion scalars start at the prior mean (z = 0). Inert unless
        # precip_lapse_enabled, but always present so save/load and the prior
        # term stay shape-consistent across tasks.
        z_tau = torch.zeros((), dtype=torch.float32, device="cuda")
        z_z0 = torch.zeros((), dtype=torch.float32, device="cuda")

        params = WhitenedParameters(
            z_bed=z_bed, z_bed_mean=z_bed_mean, z_log_beta=z_log_beta,
            z_pbias=z_pbias, z_log_mf=z_log_mf, z_log_rf=z_log_rf,
            z_tau=z_tau, z_z0=z_z0,
        )
        params.requires_grad_()
        return params

    # -------------------------------------------------------------------- core

    def physical_from(self, params: WhitenedParameters):
        """Map whitened parameters back to physical fields. Returns a namespace
        with (bed, bed_mean, log_beta, pbias, log_mf, log_rf, tau, z0)."""
        priors = self.priors

        bed_mean = GGaPPMap.apply(priors.mean_model, params.z_bed_mean)
        bed = GGaPPMap.apply(priors.bed_model, params.z_bed)# * self.observations.obs_mask.to(torch.float) + self.observations.dem*(1.0 - self.observations.obs_mask.to(torch.float))
        log_beta = GGaPPMap.apply(priors.log_beta_model, params.z_log_beta)
        pbias = GGaPPMap.apply(priors.pbias_model, params.z_pbias)
        log_mf = priors.mu_log_mf + priors.sigma_log_mf * params.z_log_mf
        log_rf = priors.mu_log_rf + priors.sigma_log_rf * params.z_log_rf
        # Precip-depletion scalars: plain affine de-whitening of a normal prior.
        tau = priors.mu_tau + priors.sigma_tau * params.z_tau
        z0 = priors.mu_z0 + priors.sigma_z0 * params.z_z0
        return PhysicalParameters(
            bed=bed, bed_mean=bed_mean, log_beta=log_beta,
            pbias=pbias, log_mf=log_mf, log_rf=log_rf, tau=tau, z0=z0,
        )

    def effective_log_pbias(self, physical: "PhysicalParameters") -> torch.Tensor:
        """Joint log-precip bias field (fine grid): the Matern pbias field minus
        the optional elevation-dependent depletion ramp. This is the quantity
        that actually multiplies precip in the forward model
        (precip = base * exp(this)); the diagnostic VTI writes it too.

        z is the static observed DEM elevation, so this field is built once per
        forward call (not per step), avoiding the per-step VRAM blowup the precip
        multiplier dodges. With precip_lapse_enabled=False it is exactly the
        Matern pbias field.
        """
        cfg = self.config
        log_pbias = physical.pbias
        if cfg.precip_lapse_enabled:
            w = cfg.precip_lapse_w
            depletion = torch.exp(-physical.tau) * w * torch.nn.functional.softplus(
                (self.observations.dem - physical.z0) / w)
            log_pbias = log_pbias - depletion
        return log_pbias

    def _initial_thickness_from_geometry(self, bed: torch.Tensor) -> torch.Tensor:
        """Seed integration thickness from the observed surface and the current
        bed iterate.

        Inverts the surface model S = max(bed + H, (1 - r) H) for H given the
        observed surface and the *current optimized bed* (`physical.bed`):
        grounded ice gets S_obs - bed, floating ice the hydrostatic value
        S_obs / (1 - r). The bed is detached, so the seeded thickness tracks
        the bed iterate across the inversion but carries no gradient w.r.t. it.
        Restricted to the observed ice extent and floored at `init_H_floor`.
        """
        cfg = self.config
        obs = self.observations
        r = cfg.rho_ice / cfg.rho_water
        bed = bed.detach()                  # current bed iterate, no gradient
        S_obs = obs.S_obs                   # max(DEM, 0)

        H_grounded = S_obs - bed
        H_float = S_obs / (1.0 - r)
        grounded = bed >= -(r / (1.0 - r)) * S_obs
        H = torch.where(grounded, H_grounded, H_float)

        ice = obs.rgi_mask.to(torch.float32)
        H = torch.clamp(H, min=cfg.init_H_floor) * ice \
            + cfg.init_H_floor * (1.0 - ice)
        return H.to(torch.float32)

    def reset_state(self, level: int) -> None:
        mg = self.mg
        mg.state.u.set(0.0, start_level=level)
        mg.state.v.set(0.0, start_level=level)
        mg.state.H.set(0.1, start_level=level)
        mg.state.H_prev.set(0.1, start_level=level)
        mg.state.mask.set(0.0, start_level=level)
        mg.adjoint.lambda_u.set(0.0, start_level=level)
        mg.adjoint.lambda_v.set(0.0, start_level=level)
        mg.adjoint.lambda_H.set(0.0, start_level=level)

    def update_depth(self, bed: torch.Tensor) -> None:
        """Blend new bed-derived depth into the existing depth field."""
        depth_new = torch.maximum(-bed, torch.zeros_like(bed)).detach()
        w = self.config.depth_blend
        self.mg.geometry.depth.set(
            w * cp.asarray(depth_new) + (1 - w) * self.mg[0].geometry.depth.data
        )

    def simulate_physical(
        self,
        *,
        level: int,
        physical: "PhysicalParameters",
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        record_volumes_at: Optional[list] = None,
        time_writer=None,
    ):
        """Run the forward model from physical parameters directly.

        Used by sensitivity.py (which optimizes in physical space to get
        gradients of ΔV w.r.t. bed/pbias/etc., not whitened coordinates).
        """
        from .forward import differentiable_restriction

        cfg = self.config
        self.update_depth(physical.bed)
        self.reset_state(level)

        # Joint log-precip bias (Matern field minus the optional elevation
        # depletion ramp); see effective_log_pbias for why it is static.
        precip_ = self.precip * torch.exp(self.effective_log_pbias(physical))
        mf = torch.exp(physical.log_mf)
        rf = torch.exp(physical.log_rf)

        bed_ = differentiable_restriction(physical.bed, level, method='avg')
        if cfg.init_from_observed_geometry:
            H_prev_full = self._initial_thickness_from_geometry(physical.bed)
        else:
            H_prev_full = self.H_prev
        H_prev_ = differentiable_restriction(H_prev_full, level)
        log_beta_ = differentiable_restriction(physical.log_beta, level)
        beta_ = torch.exp(log_beta_)

        return simulate(
            model=self.model,
            smb_model=self.smb_model,
            level=level,
            t_start=cfg.t_start if t_start is None else t_start,
            t_end=cfg.t_end if t_end is None else t_end,
            dt=cfg.dt,
            bed_=bed_,
            beta_=beta_,
            H_prev_=H_prev_,
            t2m=self.t2m,
            precip_=precip_,
            debris=self.debris,
            mf=mf,
            rf=rf,
            domain_mask=self.observations.domain_mask,
            temperature_anomaly=self.temperature_anomaly,
            base_anomaly=self.base_anomaly,
            alpha_t2m=self.alpha_t2m,
            precip_anomaly=self.precip_anomaly,
            base_precip=self.base_precip,
            alpha_precip=self.alpha_precip,
            dx_fine=self.dx,
            flotation_factor=1.0 - cfg.rho_ice / cfg.rho_water,
            record_volumes_at=record_volumes_at,
            time_writer=time_writer,
        )

    def simulate(
        self,
        *,
        level: int,
        params: WhitenedParameters,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        record_volumes_at: Optional[list] = None,
        time_writer=None,
    ):
        """Run the forward model from whitened parameters. Returns (sim, physical)."""
        physical = self.physical_from(params)
        sim = self.simulate_physical(
            level=level, physical=physical,
            t_start=t_start, t_end=t_end,
            record_volumes_at=record_volumes_at,
            time_writer=time_writer,
        )
        return sim, physical

    def compute_loss(
        self,
        *,
        sim,
        physical: "PhysicalParameters",
        params: WhitenedParameters,
        observations: Optional[Observations] = None,
        prior_means: Optional[PriorMeans] = None,
        mask: Optional[torch.Tensor] = None,
        iteration: int = 0,
        level: int = 0,
        schedule: bool = False,
    ) -> LossTerms:
        """Assemble all loss terms from a finished simulation.

        `iteration`/`level` resolve any scheduled loss weights in the config to
        their value at this optimizer step. `schedule` selects whether
        continuation ramps are honored (the initial inverse solve, `schedule=True`)
        or each weight collapses to its steady-state value (RTO and analysis, the
        default) — see GlacierConfig.at_iteration. Constant weights are unaffected
        either way.
        """
        observations = observations or self.observations
        prior_means = prior_means or PriorMeans.zeros()
        if mask is None:
            mask = (observations.rgi_mask * observations.domain_mask).to(torch.float32)

        config = self.config.at_iteration(iteration, level, schedule=schedule)
        J_srf, J_vel, J_extent, J_bed, J_snow, J_dhdt = compute_data_loss(
            config=config,
            sim_result=sim,
            physical=physical,
            observations=observations,
            mask=mask,
            dx=self.dx,
        )
        J_prior_terms = compute_prior(
            config=config,
            priors=self.priors,
            params=params,
            physical_bed=physical.bed,
            physical_bed_mean=physical.bed_mean,
            log_rf=physical.log_rf,
            log_mf=physical.log_mf,
            prior_means=prior_means,
        )
        return LossTerms(
            J_srf=J_srf, J_vel=J_vel, J_extent=J_extent, J_bed=J_bed,
            J_snow=J_snow, J_dhdt=J_dhdt,
            J_prior_bed=J_prior_terms[0],
            J_prior_bed_mean=J_prior_terms[1],
            J_prior_beta=J_prior_terms[2],
            J_prior_pbias=J_prior_terms[3],
            J_prior_smb=J_prior_terms[4],
        )


@dataclass
class PhysicalParameters:
    bed: torch.Tensor
    bed_mean: torch.Tensor
    log_beta: torch.Tensor
    pbias: torch.Tensor
    log_mf: torch.Tensor
    log_rf: torch.Tensor
    # Precip-depletion scalars. Optional so physical-space callers (sensitivity)
    # that predate the term still construct a valid object; only read when
    # config.precip_lapse_enabled.
    tau: Optional[torch.Tensor] = None
    z0: Optional[torch.Tensor] = None
