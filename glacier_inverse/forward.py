"""
Forward simulation helpers shared across all four tasks.

The time-stepping loop in `simulate` runs the SMB → ice-dynamics chain over a
step sequence designed by `scheduling.build_step_sequence` (uniform spinup
grid snapped onto every requested emission time) and returns a `SimResult`
holding a `ModelState` snapshot at each requested time plus the final state.
Used identically by inverse, rto_sample, and sensitivity.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cupy as cp
import torch
from torch.nn.functional import avg_pool2d, max_pool2d, interpolate
from torch.utils.checkpoint import checkpoint

from glare.torch import GlareStep
from glide.torch import GlideStep

from .scheduling import build_step_sequence, merge_times

glare_step = GlareStep.apply
glide_step = GlideStep.apply


def compute_smb(smb_model, t2m, t_anomaly, base_anomaly, precip_, precip_multiplier,
                debris, mf, rf, domain_mask):
    # `precip_multiplier` is a scalar applied here, inside the checkpoint, so the
    # full (12, ny, nx) scaled precip field is recomputed in backward rather than
    # stored per time step (otherwise ~50 full-grid copies are retained).
    precip_step = precip_ * precip_multiplier
    smb = glare_step(smb_model, t2m + (t_anomaly - base_anomaly), precip_step, mf, rf, debris).mean(axis=0)
    return smb.masked_fill(~domain_mask, -10)


def differentiable_restriction(field: torch.Tensor, n_times: int, method: str = "avg") -> torch.Tensor:
    if method == "avg":
        fn = avg_pool2d
    elif method == 'max':
        fn = max_pool2d
    else:
        raise NotImplementedError(f"restriction method={method!r} not supported")
    for _ in range(n_times):
        field = fn(field[None, :, :], (2, 2))[0]
    return field


def differentiable_prolongation(field: torch.Tensor, n_times: int, grid_entity: str = "cell", mode: str = 'bilinear') -> torch.Tensor:
    for _ in range(n_times):
        if grid_entity == "cell":
            ny_fine, nx_fine = 2 * field.shape[0], 2 * field.shape[1]
        elif grid_entity == "vfacet":
            ny_fine, nx_fine = 2 * field.shape[0], 2 * (field.shape[1] - 1) + 1
        elif grid_entity == "hfacet":
            ny_fine, nx_fine = 2 * (field.shape[0] - 1) + 1, 2 * field.shape[1]
        else:
            raise ValueError(f"unknown grid_entity={grid_entity!r}")
        field = interpolate(field[None, None, :, :], (ny_fine, nx_fine), mode=mode).squeeze()
    return field


class ModelState:
    """Model state emitted at a single time during the forward integration.

    The coarse tensors are references into the autograd graph the stepping loop
    builds anyway, so retaining a snapshot costs essentially no extra VRAM. The
    fine-grid views are prolonged *lazily* on first access (and cached), so a
    snapshot that a loss term never touches never pays for prolongation.
    """

    def __init__(self, *, t: float, dt_step: float, level: int, u, v, H, active,
                 smb_fine, smb_coarse, bed_coarse, flotation_factor: float):
        self.t = t
        self.dt_step = dt_step   # length of the step that emitted this state
        self.level = level
        self.u = u
        self.v = v
        self.H = H
        self.active = active
        # The SMB step runs on the fine grid and is restricted for the
        # dynamics, so the fine field needs no (lazy) prolongation.
        self.smb_fine = smb_fine
        self.smb_coarse = smb_coarse
        self.bed_coarse = bed_coarse
        self.flotation_factor = flotation_factor
        self._cache = {}

    def _lazy(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def S_coarse(self):
        return self._lazy("S_coarse", lambda: torch.maximum(
            self.bed_coarse + self.H, self.flotation_factor * self.H))

    @property
    def u_fine(self):
        return self._lazy("u_fine", lambda: differentiable_prolongation(
            self.u, self.level, grid_entity="vfacet"))

    @property
    def v_fine(self):
        return self._lazy("v_fine", lambda: differentiable_prolongation(
            self.v, self.level, grid_entity="hfacet"))

    @property
    def H_fine(self):
        return self._lazy("H_fine", lambda: differentiable_prolongation(
            self.H, self.level, grid_entity="cell"))

    @property
    def S_fine(self):
        return self._lazy("S_fine", lambda: differentiable_prolongation(
            self.S_coarse, self.level, grid_entity="cell"))

    @property
    def active_fine(self):
        return self._lazy("active_fine", lambda: differentiable_prolongation(
            self.active, self.level, grid_entity="cell"))


@dataclass
class SimResult:
    """Forward-run output: a state snapshot per requested emission time.

    `states` is keyed by the exact times passed via `record_states_at` (plus
    the final time); `final` is the last emitted state. The accessors below
    delegate to `final`, preserving the historical single-(final-)state API.
    `H_prev`/`H_prev_fine` hold the second-to-last emitted thickness — with
    `final.H_fine` and the last dt they give the model's dH/dt over the final
    step (legacy diagnostic + fallback for untimed dH/dt products).
    """
    states: dict
    final: ModelState
    volumes: dict = field(default_factory=dict)
    H_prev: Optional[torch.Tensor] = None
    _H_prev_fine: Optional[torch.Tensor] = None

    def at(self, t: float, atol: float = 1e-6) -> ModelState:
        """State snapshot at time `t` (approximate float match)."""
        for tk, state in self.states.items():
            if abs(tk - t) <= atol:
                return state
        raise KeyError(
            f"no recorded model state at t={t}; available times: "
            f"{sorted(self.states.keys())} — was {t} passed via "
            f"record_states_at?"
        )

    @property
    def H_prev_fine(self):
        if self._H_prev_fine is None and self.H_prev is not None:
            self._H_prev_fine = differentiable_prolongation(
                self.H_prev, self.final.level, grid_entity="cell")
        return self._H_prev_fine

    # ------------------------------------------------ final-state delegation
    @property
    def u(self): return self.final.u
    @property
    def v(self): return self.final.v
    @property
    def H(self): return self.final.H
    @property
    def active(self): return self.final.active
    @property
    def bed_coarse(self): return self.final.bed_coarse
    @property
    def S_coarse(self): return self.final.S_coarse
    @property
    def u_fine(self): return self.final.u_fine
    @property
    def v_fine(self): return self.final.v_fine
    @property
    def H_fine(self): return self.final.H_fine
    @property
    def S_fine(self): return self.final.S_fine
    @property
    def active_fine(self): return self.final.active_fine
    @property
    def smb_fine(self): return self.final.smb_fine
    @property
    def smb_coarse(self): return self.final.smb_coarse


def simulate(
    *,
    model,
    smb_model,
    level: int,
    t_start: float,
    t_end: float,
    dt: float,
    bed_,
    beta_,
    H_prev_,
    t2m,
    precip_,
    debris,
    mf,
    rf,
    domain_mask,
    temperature_anomaly,
    base_anomaly: float,
    alpha_t2m: float,
    dx_fine: float,
    precip_anomaly=None,
    base_precip: Optional[float] = None,
    alpha_precip=None,
    flotation_factor: float = 0.0,
    record_states_at: Optional[Sequence[float]] = None,
    record_volumes_at: Optional[Sequence[float]] = None,
    time_writer=None,
) -> SimResult:
    """Run the forward model on coarse `level` over a snapped step sequence.

    The run covers (t_start, horizon] where the horizon is max(t_end, latest
    requested emission time) — see `scheduling.build_step_sequence`. State
    snapshots are recorded at every time in `record_states_at` (keyed by the
    caller's exact floats) and scalar ice volumes at every time in
    `record_volumes_at`; the final state is always recorded.

    All field inputs (bed_, beta_, H_prev_) are expected at the coarse grid;
    full-grid quantities (t2m, precip_, domain_mask) are restricted internally.

    `flotation_factor` is `1 - rho_i/rho_w`; the surface is lower-bounded by
    the flotation freeboard `flotation_factor * H` for floating ice. The
    default of 0.0 reduces the bound to a sea-level floor (inert for grounded
    ice); callers pass the physical value derived from the config.
    """
    record_states_at = [float(t) for t in (record_states_at or [])]
    record_volumes_at = [float(t) for t in (record_volumes_at or [])]
    steps = build_step_sequence(
        t_start=t_start, t_end=t_end, dt_max=dt,
        required_times=merge_times(record_states_at, record_volumes_at),
    )

    states: dict = {}
    volumes: dict = {}
    # Penultimate emitted thickness (coarse). Seeded with the initial H_prev_ so
    # a single-step run degrades to (final - seed)/dt instead of erroring.
    H_penult = H_prev_

    # Clip anomaly lookups to the last year in the dataset (relevant for
    # sensitivity.py-style projections that run beyond the observed record).
    year_max = int(temperature_anomaly.time.max().item())
    p_year_max = (int(precip_anomaly.time.max().item())
                  if precip_anomaly is not None else None)

    def anomaly_year(t: float, y_max: int) -> int:
        # Round before truncating so 2011.9999999 reads as 2012, matching the
        # intent of grid times that are integers up to float error.
        return int(min(round(t, 9), y_max))

    t_prev = float(t_start)
    state = None
    for t_next, dt_step in steps:
        t_anomaly_0 = alpha_t2m * temperature_anomaly.sel(time=anomaly_year(t_prev, year_max)).temp_anomaly.item()
        t_anomaly_1 = alpha_t2m * temperature_anomaly.sel(time=anomaly_year(t_next, year_max)).temp_anomaly.item()
        t_anomaly = t_anomaly_1  # 0.5 * (t_anomaly_0 + t_anomaly_1)

        if precip_anomaly is not None:
            p_anom_0 = precip_anomaly.sel(time=anomaly_year(t_prev, p_year_max)).precip_anomaly.item()
            p_anom_1 = precip_anomaly.sel(time=anomaly_year(t_next, p_year_max)).precip_anomaly.item()
            p_ratio = 0.5 * (p_anom_0 + p_anom_1) / base_precip
            precip_multiplier = 1.0 + alpha_precip * (p_ratio - 1.0)
        else:
            precip_multiplier = 1.0

        smb = checkpoint(compute_smb, smb_model,
                         t2m, t_anomaly, base_anomaly, precip_, precip_multiplier,
                         debris,
                         mf, rf, domain_mask, use_reentrant=False)
        smb_ = differentiable_restriction(smb, level)

        u, v, H, active = glide_step(cp.float32(t_prev), cp.float32(dt_step),
                                     model, level, H_prev_, bed_, beta_, smb_)
        # `H_prev_` here is the thickness emitted by the previous step (the input
        # to this one); capturing it before the reassignment leaves it holding
        # the second-to-last emitted thickness once the loop ends.
        H_penult = H_prev_
        H_prev_ = H
        t_prev = t_next

        state = ModelState(
            t=t_next, dt_step=dt_step, level=level, u=u, v=v, H=H, active=active,
            smb_fine=smb, smb_coarse=smb_, bed_coarse=bed_,
            flotation_factor=flotation_factor,
        )

        # Record snapshots/volumes keyed by the caller's exact requested floats.
        for tt in record_states_at:
            if abs(t_next - tt) < 1e-6 and tt not in states:
                states[tt] = state
        for yr in record_volumes_at:
            if abs(t_next - yr) < 1e-6 and yr not in volumes:
                volumes[yr] = torch.sum(H * (dx_fine * 2 ** level) ** 2)

        if time_writer is not None:
            time_writer.append(model.mg[level], time=float(t_next))
            time_writer.write_pvd()

    states[state.t] = state  # the final state is always available

    return SimResult(
        states=states,
        final=state,
        volumes=volumes,
        H_prev=H_penult,
    )
