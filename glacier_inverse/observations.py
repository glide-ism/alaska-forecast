"""
Time-stamped observations: data + acquisition time(s) + loss + noise model.

Each observational product is an `Observation` subclass carrying its data
tensors, the calendar time(s) at which the model must emit state for
comparison (`required_times`), its noise/loss hyperparameters, and its misfit
implementation. The scalar loss weight may be a constant or a `Schedule`
(continuation, honored only by the initial MAP solve — see config.Schedule).

Domain configs do not construct Observation objects directly — data loading
needs the cropped model grid, which only exists inside `GlacierProblem`.
Instead they declare lightweight frozen *spec* dataclasses (hyperparameters +
optional time overrides); `GlacierProblem` calls `spec.build(ctx)` with an
`ObservationBuildContext` to produce the loaded Observation, or `None` when
the domain lacks the product (the graceful-skip behavior: the corresponding
loss term simply does not exist).

Timestamps come from the data files: preprocessing writes `time_nominal` /
`time_start` / `time_end` (calendar-year floats) as variable-level attributes.
A spec-level `time` override wins; files without attrs fall back to the
nominal calibration end `config.t_end` with a warning (legacy inputs keep
working unchanged).
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.functional import grid_sample

from .config import LossWeight, resolve_weight
from .loss import _huber, marginal_velocity_log_likelihood


@dataclass
class DomainData:
    """Shared, non-product context every loss term may draw on."""
    dem: torch.Tensor           # full DEM (topography + bathymetry)
    rgi_mask: torch.Tensor      # 0/1 ice extent
    rgi_label: torch.Tensor     # per-glacier long id, -1 = unlabeled
    surge_type: torch.Tensor
    obs_mask: torch.Tensor
    domain_mask: torch.Tensor   # 0/1 simulation domain


@dataclass
class ObservationBuildContext:
    """Everything a spec needs to load its product on the cropped grid."""
    gridded_data: object                 # xr.Dataset (GLIDE_inputs, cropped)
    snowline_data: Optional[object]      # xr.Dataset or None
    dhdt_data: Optional[object]          # xr.Dataset or None
    flightlines_df: object               # GeoDataFrame
    domain: DomainData
    ny: int
    nx: int
    config: object                       # GlacierConfig


def read_time_attrs(da, *, fallback: float, what: str):
    """(nominal, start, end) acquisition times from variable-level attrs.

    Missing attrs fall back to `fallback` (the nominal calibration end) with a
    one-line warning — legacy model_inputs files keep working, they just pin
    the product to the final epoch as before.
    """
    attrs = getattr(da, "attrs", {})
    nominal = attrs.get("time_nominal")
    start = attrs.get("time_start", nominal)
    end = attrs.get("time_end", nominal)
    if nominal is None:
        warnings.warn(
            f"{what}: no time_nominal/time_start/time_end attrs found; "
            f"assuming t={fallback} (the nominal calibration end). Rebuild "
            f"the product with preprocessing/make_all.py to get real "
            f"acquisition times."
        )
        return float(fallback), float(fallback), float(fallback)
    return float(nominal), float(start), float(end)


# --------------------------------------------------------------------------- #
# Observation classes                                                         #
# --------------------------------------------------------------------------- #

class Observation:
    """Base class: data + required model times + loss + noise model.

    `loss` receives the finished `SimResult` (`sim`) and looks up the model
    state at its own `required_times` via `sim.at(t)`; `GlacierProblem`
    guarantees those times were recorded. `weight` follows the Schedule
    contract: `weight_at` resolves it per optimizer step, honoring
    continuation ramps only when `schedule=True` (the initial MAP solve).

    `randomized(**eps)` returns a perturbed copy for randomize-then-optimize;
    the default passthrough is correct for categorical products. (The RTO
    driver has not yet been migrated to the per-observation API; the hook is
    here so that migration is purely driver-side.)
    """

    name: str = "observation"

    def __init__(self, *, weight: LossWeight):
        self.weight = weight

    @property
    def required_times(self) -> tuple:
        return ()

    def weight_at(self, iteration: int, level: int, *, schedule: bool) -> float:
        return resolve_weight(self.weight, iteration, level, schedule=schedule)

    def loss(self, *, sim, physical, config, domain: DomainData,
             mask: torch.Tensor, dx: float, weight: float) -> torch.Tensor:
        raise NotImplementedError

    def randomized(self, **eps) -> "Observation":
        return self

    def diagnostics(self) -> dict:
        """Static fields for the observations PVD dump (name -> 2-D tensor)."""
        return {}


class SurfaceObservation(Observation):
    """Surface-elevation misfit (Huber) against the DEM at its epoch."""

    name = "srf"

    def __init__(self, *, S_obs, time: float, sigma: float, nu: float,
                 weight: LossWeight):
        super().__init__(weight=weight)
        self.S_obs = S_obs
        self.time = time
        self.sigma = sigma
        self.nu = nu

    @property
    def required_times(self):
        return (self.time,)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        scale = config.loss_scale * dx ** 2
        state = sim.at(self.time)
        r_s = (state.S_fine - self.S_obs) / self.sigma
        return scale * weight * _huber(r_s, self.nu).sum()

    def randomized(self, *, eps_S=None):
        eps_S = torch.randn_like(self.S_obs) if eps_S is None else eps_S
        return SurfaceObservation(
            S_obs=self.S_obs + eps_S * self.sigma, time=self.time,
            sigma=self.sigma, nu=self.nu, weight=self.weight)

    def diagnostics(self):
        return {"srf_obs": self.S_obs}


class VelocityObservation(Observation):
    """Velocity misfit at the mosaic's nominal epoch.

    Either a plain pseudo-Huber on the cell-centered residual, or (when
    `surge_biased` is set) the per-glacier marginal likelihood over a velocity
    scaling eta with surge-dependent Beta priors.
    """

    name = "vel"

    def __init__(self, *, u_obs, v_obs, v_mask, time: float, sigma: float,
                 nu: float, surge_biased: bool, weight: LossWeight,
                 outlier_threshold: float):
        super().__init__(weight=weight)
        self.u_obs = u_obs
        self.v_obs = v_obs
        self.v_mask = v_mask
        self.time = time
        self.sigma = sigma
        self.nu = nu
        self.surge_biased = surge_biased
        self.outlier_threshold = outlier_threshold

    @property
    def required_times(self):
        return (self.time,)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        from numpy.polynomial.legendre import leggauss

        scale = config.loss_scale * dx ** 2
        state = sim.at(self.time)
        # Feature-tracked mosaics observe the SURFACE velocity, u + ud/(n+1)
        # under MOLHO (identical to the depth-averaged u under SSA, ud == 0).
        u_fine = state.u_surf_fine
        v_fine = state.v_surf_fine
        u_pred = (u_fine[:, 1:] + u_fine[:, :-1]) / 2.0
        v_pred = (v_fine[1:, :] + v_fine[:-1, :]) / 2.0

        if self.surge_biased:
            U_obs = torch.stack((self.u_obs.ravel(), self.v_obs.ravel()), dim=1)
            
            U_mod = torch.stack((u_pred.ravel(), v_pred.ravel()), dim=1)
            sigma = self.sigma * torch.ones(U_obs.shape[0], device='cuda',
                                            dtype=torch.float32)
            labels = domain.rgi_label.ravel()
            nodes, weights = leggauss(10)
            eta_nodes = torch.tensor((nodes + 1) / 2, device='cuda',
                                     dtype=torch.float32)
            w_gl = torch.tensor(weights / 2, device='cuda', dtype=torch.float32)

            alpha_surge = 2.0
            alpha_nonsurge = 6.0
            alpha = torch.where(domain.surge_type == 3,
                                alpha_surge, alpha_nonsurge).cuda()

            log_w_eff = (torch.log(w_gl)[:, None]
                         + torch.log(alpha)[None, :]
                         + (alpha[None, :] - 1) * torch.log(eta_nodes[:, None]))

            return marginal_velocity_log_likelihood(
                U_obs, U_mod, sigma, labels, eta_nodes, log_w_eff,
                self.nu, weight, scale,
            )
        else:
            r_u2 = (((u_pred - self.u_obs) ** 2 + (v_pred - self.v_obs) ** 2)
                    / self.sigma ** 2)
            if self.outlier_threshold is not None:
                U_obs2 = self.u_obs**2 + self.v_obs**2
                r_u2[U_obs2 > self.outlier_threshold**2] = 0.0
            return scale * weight * self.nu ** 2 * (
                torch.sqrt(1 + r_u2 / self.nu ** 2) - 1).sum()

    def randomized(self, *, eps_u=None, eps_v=None):
        eps_u = torch.randn_like(self.u_obs) if eps_u is None else eps_u
        eps_v = torch.randn_like(self.v_obs) if eps_v is None else eps_v
        return VelocityObservation(
            u_obs=self.u_obs + eps_u * self.sigma,
            v_obs=self.v_obs + eps_v * self.sigma,
            v_mask=self.v_mask, time=self.time, sigma=self.sigma, nu=self.nu,
            surge_biased=self.surge_biased, weight=self.weight)


class ExtentObservation(Observation):
    """Glacier-extent misfit (Brier-style) at the inventory's epoch.

    Blends a thickness-derived probability with an SMB-derived one via the
    dynamics' active mask. `smb_dt` is the timescale converting SMB to a
    thickness-equivalent logit (historically `config.dt`); it is an explicit
    hyperparameter so a variable step sequence cannot silently change it.
    The extent labels themselves come through `mask` (rgi_mask ∩ domain, or
    the RTO Bernoulli realization).
    """

    name = "extent"

    def __init__(self, *, time: float, s_H: float, smb_dt: float,
                 weight: LossWeight):
        super().__init__(weight=weight)
        self.time = time
        self.s_H = s_H
        self.smb_dt = smb_dt

    @property
    def required_times(self):
        return (self.time,)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        state = sim.at(self.time)
        H_fine = state.H_fine
        active_fine = state.active_fine

        p_extent_dyn = (2.0 / (1 + torch.exp(-H_fine / self.s_H)) - 1
                        ).clip(min=0.001, max=0.999)
        p_extent_smb = (1 / (1 + torch.exp(-self.smb_dt / self.s_H * state.smb_fine))
                        ).clip(min=0.001, max=0.999)
        p_extent = p_extent_dyn * (1 - active_fine) + p_extent_smb * active_fine
        return config.loss_scale * weight * dx ** 2 * (
            mask * ((1 - p_extent) / 0.5) ** 2).sum()


class BedObservation(Observation):
    """Bed misfit: flightline picks + bed-equals-DEM anchor where no ice.

    Time-independent (the bed does not evolve), so `required_times` is empty.
    The off-ice anchor uses the full DEM (topography + bathymetry), not the
    sea-level-clamped surface, so submarine bed is anchored to bathymetry.
    The anchor DEM is owned here (not read from `domain`) so RTO can perturb
    it jointly with the surface observation via a shared eps_S draw.
    """

    name = "bed"

    def __init__(self, *, bed_obs, bed_normed_coords, dem_anchor,
                 sigma: float, sigma_dem: float, nu: float,
                 weight: LossWeight, ny: int, nx: int):
        super().__init__(weight=weight)
        self.bed_obs = bed_obs
        self.bed_normed_coords = bed_normed_coords
        self.dem_anchor = dem_anchor
        self.sigma = sigma
        self.sigma_dem = sigma_dem
        self.nu = nu
        self.ny = ny
        self.nx = nx

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        scale = config.loss_scale * dx ** 2
        bed_fine = physical.bed

        bed_at_flightlines = grid_sample(
            bed_fine[None, None, :, :],
            self.bed_normed_coords[None, None, :, :],
            mode="bilinear", align_corners=False,
        ).squeeze()
        r_bed_fl = (bed_at_flightlines - self.bed_obs) / self.sigma
        r_bed_grid = (bed_fine - self.dem_anchor) / self.sigma_dem * (1 - mask)
        return scale * weight * (_huber(r_bed_fl, self.nu).sum()
                                 + _huber(r_bed_grid, self.nu).sum())

    def randomized(self, *, eps_bed=None, eps_S=None):
        eps_bed = torch.randn_like(self.bed_obs) if eps_bed is None else eps_bed
        eps_S = torch.randn_like(self.dem_anchor) if eps_S is None else eps_S
        # eps_S is shared with the surface observation's draw so the off-ice
        # anchor perturbation matches the surface perturbation, preserving the
        # historical RTO noise bookkeeping.
        return BedObservation(
            bed_obs=self.bed_obs + eps_bed * self.sigma,
            bed_normed_coords=self.bed_normed_coords,
            dem_anchor=self.dem_anchor + eps_S * self.sigma_dem,
            sigma=self.sigma, sigma_dem=self.sigma_dem, nu=self.nu,
            weight=self.weight, ny=self.ny, nx=self.nx)

    def diagnostics(self):
        """Rasterize the scattered picks onto the finest grid (NaN = no data),
        using the same normalized coordinates the loss feeds to grid_sample."""
        cn = self.bed_normed_coords[:, 0]
        rn = self.bed_normed_coords[:, 1]
        col = (((cn + 1.0) * self.nx - 1.0) / 2.0).round().long().clamp_(0, self.nx - 1)
        row = (((rn + 1.0) * self.ny - 1.0) / 2.0).round().long().clamp_(0, self.ny - 1)
        raster = torch.full((self.ny, self.nx), float("nan"),
                            dtype=torch.float32, device="cuda")
        raster[row, col] = self.bed_obs.to(torch.float32)
        return {"bed_obs": raster}


class SnowlineObservation(Observation):
    """Snowline (ELA-proxy) misfit at the composite's nominal epoch.

    The end-of-summer snowline product gives, per cell, the fraction of the
    glacierized subarea that retained snow (`snow_label` in [0, 1]). The model
    produces a logit by scaling the SMB field at the observation time, so
    sigmoid(SMB / s_smb) reads as P(cell is above the ELA). Restricted to
    `snow_mask` (valid, glacierized cells); Brier-scored.
    """

    name = "snow"

    def __init__(self, *, snow_label, snow_mask, time: float, s_smb: float,
                 weight: LossWeight):
        super().__init__(weight=weight)
        self.snow_label = snow_label
        self.snow_mask = snow_mask
        self.time = time
        self.s_smb = s_smb

    @property
    def required_times(self):
        return (self.time,)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        smb = sim.at(self.time).smb_fine
        logits = smb / self.s_smb
        y = torch.nn.functional.sigmoid(logits)
        brier = (self.snow_mask * ((y - self.snow_label) / 0.25) ** 2).sum()
        #brier = (self.snow_mask * self.snow_label * ((1.0 - y) / 0.25)**2).sum()
        return config.loss_scale * weight * dx ** 2 * brier

    def diagnostics(self):
        return {"snow_label": self.snow_label, "snow_mask": self.snow_mask}


class DhdtObservation(Observation):
    """Surface elevation-change-rate misfit (Huber) over the product's window.

    The model rate is a true two-snapshot difference over the observation
    window [t0, t1]:

        dHdt_model = (H(t1) - H(t0)) / (t1 - t0),

    compared against the observed rate (Hugonnet), with residuals normalized
    by the per-pixel uncertainty floored at `sigma_floor`.

    `legacy_final_step` reproduces the historical behavior for input files
    without time attrs: the rate over the final emitted step,
    (H_final - H_prev) / dt_final (the *actual* final step length — with a
    snapped step sequence this is not necessarily config.dt).
    """

    name = "dhdt"

    def __init__(self, *, dhdt, dhdt_err, dhdt_mask, t0: float, t1: float,
                 sigma_floor: float, nu: float, weight: LossWeight,
                 legacy_final_step: bool = False):
        super().__init__(weight=weight)
        self.dhdt = dhdt
        self.dhdt_err = dhdt_err
        self.dhdt_mask = dhdt_mask
        self.t0 = t0
        self.t1 = t1
        self.sigma_floor = sigma_floor
        self.nu = nu
        self.legacy_final_step = legacy_final_step

    @property
    def required_times(self):
        return () if self.legacy_final_step else (self.t0, self.t1)

    def model_rate(self, sim, space: str = "fine") -> torch.Tensor:
        """The model-side elevation-change rate this observation is compared
        against. Both `loss` and the drivers' dhdt diagnostic call this, so
        the VTI field shows exactly the quantity the misfit penalizes.
        `space` selects the fine (prolonged) or coarse grid."""
        if space not in ("fine", "coarse"):
            raise ValueError(f"space must be 'fine' or 'coarse', got {space!r}")
        if self.legacy_final_step:
            H1 = sim.final.H_fine if space == "fine" else sim.final.H
            H0 = sim.H_prev_fine if space == "fine" else sim.H_prev
            return (H1 - H0) / sim.final.dt_step
        s0, s1 = sim.at(self.t0), sim.at(self.t1)
        if space == "fine":
            return (s1.H_fine - s0.H_fine) / (self.t1 - self.t0)
        return (s1.H - s0.H) / (self.t1 - self.t0)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        scale = config.loss_scale * dx ** 2
        dhdt_model = self.model_rate(sim, "fine")
        sigma = torch.clamp(self.dhdt_err, min=self.sigma_floor)
        r = (dhdt_model - self.dhdt) / sigma * self.dhdt_mask
        return scale * weight * _huber(r, self.nu).sum()

    def randomized(self, *, eps_dhdt=None):
        eps_dhdt = torch.randn_like(self.dhdt) if eps_dhdt is None else eps_dhdt
        sigma = torch.clamp(self.dhdt_err, min=self.sigma_floor)
        return DhdtObservation(
            dhdt=self.dhdt + eps_dhdt * sigma * self.dhdt_mask,
            dhdt_err=self.dhdt_err, dhdt_mask=self.dhdt_mask,
            t0=self.t0, t1=self.t1, sigma_floor=self.sigma_floor, nu=self.nu,
            weight=self.weight, legacy_final_step=self.legacy_final_step)

    def diagnostics(self):
        return {"dhdt": self.dhdt, "dhdt_err": self.dhdt_err,
                "dhdt_mask": self.dhdt_mask}


def build_divide_mask(rgi_label, domain_mask, u_obs, v_obs, v_mask, *,
                      u_conf: float, buffer_px: int) -> torch.Tensor:
    """0/1 float mask of drainage-divide pixels where cross-basin flux is
    penalized.

    Starts from inter-glacier boundaries (4-neighbor pairs of distinct
    non-negative RGI labels), dilates by `buffer_px`, then removes pixels the
    velocity mosaic identifies as confluences — valid observations moving
    faster than `u_conf`. At a true divide the ice barely moves (or feature
    tracking has no data), so slow/no-data boundary pixels are kept; where a
    tributary legitimately crosses its RGI boundary into a trunk, the observed
    speed is high and the penalty is dropped. Device-agnostic and standalone
    so it can be validated against the inputs without building a problem.
    """
    lab = rgi_label
    boundary = torch.zeros_like(lab, dtype=torch.bool)
    for a_sl, b_sl in (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ):
        a, b = lab[a_sl], lab[b_sl]
        differs = (a >= 0) & (b >= 0) & (a != b)
        boundary[a_sl] |= differs
        boundary[b_sl] |= differs
    if buffer_px > 0:
        boundary = torch.nn.functional.max_pool2d(
            boundary[None, None].float(), 2 * buffer_px + 1,
            stride=1, padding=buffer_px)[0, 0] > 0
    speed = torch.sqrt(u_obs ** 2 + v_obs ** 2)
    confluence = (v_mask > 0) & (speed > u_conf)
    return (boundary & ~confluence & domain_mask.bool()).to(torch.float32)


class DivideFluxObservation(Observation):
    """Cross-basin flux penalty: no significant ice flux across drainage
    divides between distinct RGI glaciers.

    A prior-style pseudo-observation, not a data product: it encodes the
    expert knowledge that inventory drainage divides (Bering/Yahtse, ...) do
    not exchange mass, which the Matern bed prior is too local to express and
    which the Huberized velocity misfit is too forgiving to enforce (a
    fictitious ice stream saturates the robust loss and is written off as an
    outlier). The penalty is deliberately quadratic — large fictitious fluxes
    must stay expensive — on the flux-magnitude excess over a small allowance
    `q0` (legitimate near-divide flux is O(H * a few m/yr)), normalized by
    `q_scale`. `randomized` is the identity: there is no observational noise
    to perturb.
    """

    name = "divide"

    def __init__(self, *, divide_mask, time: float, q0: float, q_scale: float,
                 weight: LossWeight):
        super().__init__(weight=weight)
        self.divide_mask = divide_mask
        self.time = time
        self.q0 = q0
        self.q_scale = q_scale

    @property
    def required_times(self):
        return (self.time,)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        state = sim.at(self.time)
        # Deliberately the depth-AVERAGED velocity (not surface): ice flux is
        # exactly u_bar * H in both stress schemes.
        u = (state.u_fine[:, 1:] + state.u_fine[:, :-1]) / 2.0
        v = (state.v_fine[1:, :] + state.v_fine[:-1, :]) / 2.0
        q = state.H_fine * torch.sqrt(u ** 2 + v ** 2 + 1e-6)
        excess = torch.relu(q - self.q0) / self.q_scale
        return config.loss_scale * weight * dx ** 2 * (
            self.divide_mask * excess ** 2).sum()

    def diagnostics(self):
        return {"divide_mask": self.divide_mask}


# --------------------------------------------------------------------------- #
# Specs: what a domain config declares                                        #
# --------------------------------------------------------------------------- #

def _resolve_time(override: Optional[float], da, *, fallback: float,
                  what: str) -> float:
    if override is not None:
        return float(override)
    nominal, _, _ = read_time_attrs(da, fallback=fallback, what=what)
    return nominal


@dataclass(frozen=True)
class SurfaceSpec:
    weight: LossWeight = 2e-5
    sigma: float = 10.0
    nu: float = 1.0
    time: Optional[float] = None    # override; None -> file attr -> t_end

    def build(self, ctx: ObservationBuildContext) -> SurfaceObservation:
        cfg = ctx.config
        time = _resolve_time(self.time, ctx.gridded_data.elevation,
                             fallback=cfg.t_end, what="surface DEM (elevation)")
        return SurfaceObservation(
            S_obs=torch.clamp(ctx.domain.dem, min=0.0), time=time,
            sigma=self.sigma, nu=self.nu, weight=self.weight)


@dataclass(frozen=True)
class VelocitySpec:
    weight: LossWeight = 2e-5
    sigma: float = 10.0
    nu: float = 1.0
    surge_biased: bool = False
    time: Optional[float] = None
    outlier_threshold: Optional[float] = None

    def build(self, ctx: ObservationBuildContext) -> VelocityObservation:
        cfg = ctx.config
        gd = ctx.gridded_data
        domain_mask = ctx.domain.domain_mask
        time = _resolve_time(self.time, gd.vx, fallback=cfg.t_end,
                             what="velocity mosaic (vx)")
        u_obs = torch.tensor(gd.vx.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_obs = torch.tensor(gd.vy.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_mask = torch.tensor(gd.vmask.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        return VelocityObservation(
            u_obs=u_obs, v_obs=v_obs, v_mask=v_mask, time=time,
            sigma=self.sigma, nu=self.nu, surge_biased=self.surge_biased,
            weight=self.weight,outlier_threshold=self.outlier_threshold)


@dataclass(frozen=True)
class ExtentSpec:
    weight: LossWeight = 2e-4
    s_H: float = 10.0
    smb_dt: Optional[float] = None  # None -> config.dt (the historical factor)
    time: Optional[float] = None

    def build(self, ctx: ObservationBuildContext) -> ExtentObservation:
        cfg = ctx.config
        time = _resolve_time(self.time, ctx.gridded_data.rgi_mask,
                             fallback=cfg.t_end, what="RGI extent (rgi_mask)")
        smb_dt = cfg.dt if self.smb_dt is None else self.smb_dt
        return ExtentObservation(time=time, s_H=self.s_H, smb_dt=smb_dt,
                                 weight=self.weight)


@dataclass(frozen=True)
class BedSpec:
    weight: LossWeight = 2e-5
    sigma: float = 10.0
    sigma_dem: float = 10.0   # off-ice anchor noise (historically sigma_s)
    nu: float = 1.0

    def build(self, ctx: ObservationBuildContext) -> BedObservation:
        gd = ctx.gridded_data
        flightlines = torch.tensor(
            ctx.flightlines_df[["x", "y", "bed"]].values,
            dtype=torch.float32, device="cuda")
        xmin, xmax = gd.x.min().item(), gd.x.max().item()
        ymin, ymax = gd.y.min().item(), gd.y.max().item()
        col_normed = 2.0 * ((flightlines[:, 0] - xmin) / (xmax - xmin)) - 1
        row_normed = -(2.0 * ((flightlines[:, 1] - ymin) / (ymax - ymin)) - 1)
        bed_normed_coords = torch.stack([col_normed, row_normed], dim=-1)
        return BedObservation(
            bed_obs=flightlines[:, 2], bed_normed_coords=bed_normed_coords,
            dem_anchor=ctx.domain.dem, sigma=self.sigma,
            sigma_dem=self.sigma_dem, nu=self.nu, weight=self.weight,
            ny=ctx.ny, nx=ctx.nx)


@dataclass(frozen=True)
class SnowlineSpec:
    weight: LossWeight = 2e-4
    s_smb: float = 0.2
    time: Optional[float] = None

    def build(self, ctx: ObservationBuildContext) -> Optional[SnowlineObservation]:
        sd = ctx.snowline_data
        if sd is None:
            return None

        expected = (ctx.ny, ctx.nx)
        if sd.snow_fraction.shape != expected:
            raise ValueError(
                f"snowline grid {tuple(sd.snow_fraction.shape)} does not match "
                f"the cropped model grid {expected}; rebuild "
                f"{ctx.config.snowline_filename} on the DEM grid."
            )

        cfg = ctx.config
        time = _resolve_time(self.time, sd.snow_fraction, fallback=cfg.t_end,
                             what="snowline composite (snow_fraction)")
        domain_mask = ctx.domain.domain_mask
        snow_label = torch.tensor(
            sd.snow_fraction.values, dtype=torch.float32, device="cuda"
        ).nan_to_num().clamp_(0.0, 1.0)
        glacier_fraction = torch.tensor(
            sd.glacier_fraction.values, dtype=torch.float32, device="cuda"
        ).nan_to_num()
        snow_mask = ((glacier_fraction > 0.0) & domain_mask).to(torch.float32)
        snow_label = snow_label.masked_fill(snow_mask == 0.0, 0.0)
        return SnowlineObservation(
            snow_label=snow_label, snow_mask=snow_mask, time=time,
            s_smb=self.s_smb, weight=self.weight)


@dataclass(frozen=True)
class DhdtSpec:
    weight: LossWeight = 2e-5
    sigma_floor: float = 0.5
    nu: float = 1.0
    t0: Optional[float] = None    # override; None -> file attrs -> legacy mode
    t1: Optional[float] = None

    def build(self, ctx: ObservationBuildContext) -> Optional[DhdtObservation]:
        dd = ctx.dhdt_data
        if dd is None:
            return None

        expected = (ctx.ny, ctx.nx)
        if dd.dhdt.shape != expected:
            raise ValueError(
                f"dhdt grid {tuple(dd.dhdt.shape)} does not match the cropped "
                f"model grid {expected}; rebuild {ctx.config.dhdt_filename} "
                f"on the DEM grid."
            )

        cfg = ctx.config
        attrs = getattr(dd.dhdt, "attrs", {})
        legacy = False
        if self.t0 is not None and self.t1 is not None:
            t0, t1 = float(self.t0), float(self.t1)
        elif "time_start" in attrs and "time_end" in attrs:
            t0, t1 = float(attrs["time_start"]), float(attrs["time_end"])
        else:
            # Legacy input file: no window recorded. Fall back to the
            # historical final-step rate so old model_inputs keep working.
            warnings.warn(
                f"dhdt product ({cfg.dhdt_filename}): no time_start/time_end "
                f"attrs; falling back to the legacy final-step rate "
                f"(H(t_end) - H_prev)/dt. Rebuild with preprocessing/"
                f"make_dhdt.py to compare over the true observation window."
            )
            t0, t1 = cfg.t_end - cfg.dt, cfg.t_end
            legacy = True
        if t1 <= t0:
            raise ValueError(f"dhdt window must have t1 > t0, got [{t0}, {t1}]")

        domain_mask = ctx.domain.domain_mask
        dhdt_raw = torch.tensor(dd.dhdt.values, dtype=torch.float32,
                                device="cuda")
        dhdt_err_raw = torch.tensor(dd.dhdt_err.values, dtype=torch.float32,
                                    device="cuda")
        valid = torch.isfinite(dhdt_raw) & torch.isfinite(dhdt_err_raw)
        dhdt_mask = (valid & domain_mask).to(torch.float32)
        dhdt = dhdt_raw.nan_to_num().masked_fill(dhdt_mask == 0.0, 0.0)
        dhdt_err = dhdt_err_raw.nan_to_num().masked_fill(dhdt_mask == 0.0, 0.0)
        return DhdtObservation(
            dhdt=dhdt, dhdt_err=dhdt_err, dhdt_mask=dhdt_mask, t0=t0, t1=t1,
            sigma_floor=self.sigma_floor, nu=self.nu, weight=self.weight,
            legacy_final_step=legacy)


@dataclass(frozen=True)
class DivideFluxSpec:
    """Opt-in cross-basin flux penalty (see DivideFluxObservation).

    `u_conf` and `buffer_px` shape the divide mask (validate with the
    `divide_mask` field in the observations PVD dump before trusting a run);
    `z_min` optionally restricts the penalty to divides above an elevation,
    for domains where slow quiescent trunks would otherwise keep confluence
    pixels in the mask. `q0`/`q_scale` are in m^2/yr of per-unit-width flux
    (H * speed): the default allowance corresponds to e.g. 300 m of ice
    moving ~15 m/yr.
    """
    weight: LossWeight = 1e-5
    u_conf: float = 30.0            # obs speed (m/yr) marking a confluence
    buffer_px: int = 1
    q0: float = 5e3                 # flux allowance (m^2/yr)
    q_scale: float = 1e4            # residual normalization (m^2/yr)
    z_min: Optional[float] = None   # keep only divides above this elevation
    time: Optional[float] = None    # evaluation epoch; None -> t_end

    def build(self, ctx: ObservationBuildContext) -> Optional["DivideFluxObservation"]:
        cfg = ctx.config
        gd = ctx.gridded_data
        u_obs = torch.tensor(gd.vx.values, dtype=torch.float32,
                             device="cuda").nan_to_num()
        v_obs = torch.tensor(gd.vy.values, dtype=torch.float32,
                             device="cuda").nan_to_num()
        v_mask = torch.tensor(gd.vmask.values, dtype=torch.float32,
                              device="cuda").nan_to_num()
        divide_mask = build_divide_mask(
            ctx.domain.rgi_label, ctx.domain.domain_mask, u_obs, v_obs,
            v_mask, u_conf=self.u_conf, buffer_px=self.buffer_px)
        if self.z_min is not None:
            divide_mask = divide_mask * (ctx.domain.dem > self.z_min)
        if divide_mask.sum() == 0:
            warnings.warn("DivideFluxSpec: divide mask is empty (no "
                          "inter-glacier boundaries survived the filters); "
                          "dropping the term.")
            return None
        time = float(self.time) if self.time is not None else float(cfg.t_end)
        return DivideFluxObservation(
            divide_mask=divide_mask, time=time, q0=self.q0,
            q_scale=self.q_scale, weight=self.weight)


def default_observation_specs(config=None) -> tuple:
    """The standard six products with library-default hyperparameters.

    When given a config that still carries the legacy global fields
    (sigma_s, lambda_s, ...), those seed the defaults so pre-migration domain
    configs behave identically; on a migrated config the plain defaults apply.
    """
    g = (lambda name, default: getattr(config, name, default)) if config \
        else (lambda name, default: default)
    # Under bed conditioning the bed data lives in the prior map itself, so
    # the soft likelihood must carry weight 0 (double counting otherwise).
    # The observation object is kept: its diagnostics feed the PVD dump.
    bed_conditioned = bool(getattr(getattr(config, "bed_conditioning", None),
                                   "enabled", False))
    return (
        SurfaceSpec(weight=g("lambda_s", 2e-5), sigma=g("sigma_s", 10.0),
                    nu=g("nu_s", 1.0)),
        VelocitySpec(weight=g("lambda_u", 2e-5), sigma=g("sigma_u", 10.0),
                     nu=g("nu_u", 1.0),
                     surge_biased=g("surge_biased_likelihood", False)),
        ExtentSpec(weight=g("lambda_e", 2e-4), s_H=g("s_H", 10.0)),
        BedSpec(weight=0.0 if bed_conditioned else g("lambda_bed", 2e-5),
                sigma=g("sigma_bed", 10.0),
                sigma_dem=g("sigma_s", 10.0), nu=g("nu_bed", 1.0)),
        SnowlineSpec(weight=g("lambda_snow", 2e-4), s_smb=g("s_smb", 0.2)),
        DhdtSpec(weight=g("lambda_dhdt", 2e-5), sigma_floor=g("sigma_dhdt", 0.5),
                 nu=g("nu_dhdt", 1.0)),
    )
