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
from ggapp.torch import GGaPPMap, GGaPPWhiten

from .config import GlacierConfig, SolverConfig
from .forward import simulate
from .loss import LossTerms, PriorMeans, compute_data_misfit, compute_prior
from .priors import GlacierPriors


@dataclass
class Observations:
    u_obs: torch.Tensor
    v_obs: torch.Tensor
    S_obs: torch.Tensor
    bed_obs: torch.Tensor
    bed_normed_coords: torch.Tensor
    rgi_mask: torch.Tensor      # 0/1 ice extent
    domain_mask: torch.Tensor   # 0/1 simulation domain

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
            S_obs=self.S_obs + eps_S * sigma_s,
            bed_obs=self.bed_obs + eps_bed * sigma_bed,
            bed_normed_coords=self.bed_normed_coords,
            rgi_mask=self.rgi_mask,
            domain_mask=self.domain_mask,
        )


class WhitenedParameters:
    """The six tensors we optimize over: four whitened fields plus two whitened scalars."""

    def __init__(self, z_bed, z_bed_mean, z_log_beta, z_pbias, z_log_mf, z_log_rf):
        self.z_bed = z_bed
        self.z_bed_mean = z_bed_mean
        self.z_log_beta = z_log_beta
        self.z_pbias = z_pbias
        self.z_log_mf = z_log_mf
        self.z_log_rf = z_log_rf

    def requires_grad_(self) -> "WhitenedParameters":
        for t in (self.z_bed, self.z_bed_mean, self.z_log_beta,
                  self.z_pbias, self.z_log_mf, self.z_log_rf):
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

        # Cached fine-grid forcings/state used in every forward call.
        self.t2m = torch.tensor(self.smb_model.grid.temperature.t2m.data,
                                device="cuda")
        self.precip = torch.tensor(self.smb_model.grid.precipitation.precip.data,
                                   device="cuda")
        self.H_prev = torch.tensor(self.mg[0].state.H_prev.data,
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
        gd = self.gridded_data
        smb.grid.insolation.insol_mean.set(gd.monthly_solar_potential_mean)
        smb.grid.insolation.insol_cos.set(gd.monthly_solar_potential_cos)
        smb.grid.insolation.insol_sin.set(gd.monthly_solar_potential_sin)
        smb.grid.temperature.t2m.set(gd.monthly_t2m)
        smb.grid.precipitation.precip.set(gd.monthly_precip)
        smb.grid.insolation.rf.set(self.config.mu_rf)
        smb.grid.temperature.mf.set(self.config.mu_mf)
        smb.forward()
        return smb

    def _build_observations(self, flightlines_df) -> Observations:
        gd = self.gridded_data
        domain_mask = torch.tensor(gd.domain_mask.values, device="cuda")
        rgi_mask = torch.tensor(gd.rgi_mask.values, device="cuda")

        u_obs = torch.tensor(gd.vx.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_obs = torch.tensor(gd.vy.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        S_obs = torch.tensor(gd.elevation.values, dtype=torch.float32, device="cuda")

        flightlines = torch.tensor(flightlines_df[["x", "y", "bed"]].values,
                                    dtype=torch.float32, device="cuda")
        xmin, xmax = gd.x.min().item(), gd.x.max().item()
        ymin, ymax = gd.y.min().item(), gd.y.max().item()
        col_normed = 2.0 * ((flightlines[:, 0] - xmin) / (xmax - xmin)) - 1
        row_normed = -(2.0 * ((flightlines[:, 1] - ymin) / (ymax - ymin)) - 1)
        bed_normed_coords = torch.stack([col_normed, row_normed], dim=-1)
        bed_obs = flightlines[:, 2]

        return Observations(
            u_obs=u_obs, v_obs=v_obs, S_obs=S_obs,
            bed_obs=bed_obs, bed_normed_coords=bed_normed_coords,
            rgi_mask=rgi_mask, domain_mask=domain_mask,
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

        params = WhitenedParameters(
            z_bed=z_bed, z_bed_mean=z_bed_mean, z_log_beta=z_log_beta,
            z_pbias=z_pbias, z_log_mf=z_log_mf, z_log_rf=z_log_rf,
        )
        params.requires_grad_()
        return params

    # -------------------------------------------------------------------- core

    def physical_from(self, params: WhitenedParameters):
        """Map whitened parameters back to physical fields. Returns a namespace
        with (bed, bed_mean, log_beta, pbias, log_mf, log_rf)."""
        priors = self.priors

        bed_mean = GGaPPMap.apply(priors.mean_model, params.z_bed_mean)
        bed = GGaPPMap.apply(priors.bed_model, params.z_bed)
        log_beta = GGaPPMap.apply(priors.log_beta_model, params.z_log_beta)
        pbias = GGaPPMap.apply(priors.pbias_model, params.z_pbias)
        log_mf = priors.mu_log_mf + priors.sigma_log_mf * params.z_log_mf
        log_rf = priors.mu_log_rf + priors.sigma_log_rf * params.z_log_rf
        return PhysicalParameters(
            bed=bed, bed_mean=bed_mean, log_beta=log_beta,
            pbias=pbias, log_mf=log_mf, log_rf=log_rf,
        )

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

        precip_ = self.precip + physical.pbias
        mf = torch.exp(physical.log_mf)
        rf = torch.exp(physical.log_rf)

        bed_ = differentiable_restriction(physical.bed, level)
        H_prev_ = differentiable_restriction(self.H_prev, level)
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
            mf=mf,
            rf=rf,
            domain_mask=self.observations.domain_mask,
            temperature_anomaly=self.temperature_anomaly,
            base_anomaly=self.base_anomaly,
            alpha_t2m=self.alpha_t2m,
            dx_fine=self.dx,
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
    ) -> LossTerms:
        """Assemble all loss terms from a finished simulation."""
        observations = observations or self.observations
        prior_means = prior_means or PriorMeans.zeros()
        if mask is None:
            mask = (observations.rgi_mask * observations.domain_mask).to(torch.float32)

        J_srf, J_vel, J_extent, J_bed = compute_data_misfit(
            config=self.config,
            sim_result=sim,
            bed_fine=physical.bed,
            observations=observations,
            mask=mask,
            dx=self.dx,
        )
        J_prior_terms = compute_prior(
            config=self.config,
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
