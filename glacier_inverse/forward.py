"""
Forward simulation helpers shared across all four tasks.

The time-stepping loop in `simulate` runs the SMB → ice-dynamics chain for
each time step and returns the final (u, v, H, S) plus optional intermediate
volume checkpoints. Used identically by inverse, rto_sample, and sensitivity.
"""
from dataclasses import dataclass
from typing import Optional

import cupy as cp
import torch
from torch.nn.functional import avg_pool2d, max_pool2d, interpolate
from torch.utils.checkpoint import checkpoint

from glare.torch import GlareStep
from glide.torch import GlideStep

glare_step = GlareStep.apply
glide_step = GlideStep.apply


def compute_smb(smb_model, t2m, t_anomaly, base_anomaly, precip_, debris, mf, rf, domain_mask):
    smb = glare_step(smb_model, t2m + (t_anomaly - base_anomaly), precip_, mf, rf, debris).mean(axis=0)
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


@dataclass
class SimResult:
    # Coarse-level (post-time-stepping) state.
    u: torch.Tensor
    v: torch.Tensor
    H: torch.Tensor
    active: torch.Tensor
    bed_coarse: torch.Tensor    # the restricted bed used during stepping
    S_coarse: torch.Tensor      # max(bed_coarse + H, flotation freeboard)
    # Final-level prolonged outputs (matched to fine grid).
    u_fine: torch.Tensor
    v_fine: torch.Tensor
    H_fine: torch.Tensor
    active_fine: torch.Tensor
    S_fine: torch.Tensor
    # SMB from the final time step (the calibration-epoch field used by the
    # snowline / ELA loss). `smb_fine` is full resolution; `smb_coarse` is the
    # restricted field actually fed to the ice dynamics. None if no step ran.
    smb_fine: Optional[torch.Tensor] = None
    smb_coarse: Optional[torch.Tensor] = None
    # Volume checkpoints keyed by year — populated only when requested.
    volumes: dict = None


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
    record_volumes_at: Optional[list] = None,
    time_writer=None,
) -> SimResult:
    """Run the forward model from t_start to t_end on coarse `level`.

    All field inputs (bed_, beta_, H_prev_) are expected at the coarse grid;
    full-grid quantities (t2m, precip_, domain_mask) are restricted internally.
    Returns coarse u, v, H plus prolonged u_fine, v_fine, H_fine, S_fine.

    `flotation_factor` is `1 - rho_i/rho_w`; the surface is lower-bounded by
    the flotation freeboard `flotation_factor * H` for floating ice. The
    default of 0.0 reduces the bound to a sea-level floor (inert for grounded
    ice); callers pass the physical value derived from the config.
    """
    record_volumes_at = list(record_volumes_at or [])
    volumes: dict = {}
    last_smb = None      # fine-grid SMB from the most recent step
    last_smb_ = None     # restricted SMB from the most recent step

    t = cp.float32(t_start)
    t_end_c = cp.float32(t_end)
    dt_c = cp.float32(dt)

    # Clip anomaly lookups to the last year in the dataset (relevant for
    # sensitivity.py-style projections that run beyond the observed record).
    year_max = int(temperature_anomaly.time.max().item())
    p_year_max = (int(precip_anomaly.time.max().item())
                  if precip_anomaly is not None else None)

    while t < t_end_c:
        t_anomaly_0 = alpha_t2m * temperature_anomaly.sel(time=int(min(float(t), year_max))).temp_anomaly.item()
        t_anomaly_1 = alpha_t2m * temperature_anomaly.sel(time=int(min(float(t + dt_c), year_max))).temp_anomaly.item()
        t_anomaly = 0.5 * (t_anomaly_0 + t_anomaly_1)

        if precip_anomaly is not None:
            p_anom_0 = precip_anomaly.sel(time=int(min(float(t), p_year_max))).precip_anomaly.item()
            p_anom_1 = precip_anomaly.sel(time=int(min(float(t + dt_c), p_year_max))).precip_anomaly.item()
            p_ratio = 0.5 * (p_anom_0 + p_anom_1) / base_precip
            precip_multiplier = 1.0 + alpha_precip * (p_ratio - 1.0)
            precip_step = precip_ * precip_multiplier
        else:
            precip_step = precip_

        smb = checkpoint(compute_smb, smb_model,
                         t2m, t_anomaly, base_anomaly, precip_step,
                         debris,
                         mf, rf, domain_mask, use_reentrant=False)
        smb_ = differentiable_restriction(smb, level)
        last_smb, last_smb_ = smb, smb_

        u, v, H, active = glide_step(t, dt_c, model, level, H_prev_, bed_, beta_, smb_)
        t += dt_c
        H_prev_ = H

        # Record volumes at requested year boundaries (within a single dt).
        for yr in record_volumes_at:
            if abs(float(t) - float(yr)) < 1e-3 and yr not in volumes:
                volumes[yr] = torch.sum(H * (dx_fine * 2 ** level) ** 2)

        if time_writer is not None:
            time_writer.append(model.mg[level], time=float(t))
            time_writer.write_pvd()

    S_coarse = torch.maximum(bed_ + H, flotation_factor * H)
    u_fine = differentiable_prolongation(u, level, grid_entity="vfacet")
    v_fine = differentiable_prolongation(v, level, grid_entity="hfacet")
    H_fine = differentiable_prolongation(H, level, grid_entity="cell")
    S_fine = differentiable_prolongation(S_coarse, level, grid_entity="cell")
    active_fine = differentiable_prolongation(active,level,grid_entity="cell")

    return SimResult(
        u=u, v=v, H=H, active=active,
        bed_coarse=bed_, S_coarse=S_coarse,
        u_fine=u_fine, v_fine=v_fine, H_fine=H_fine, S_fine=S_fine,
        active_fine=active_fine,
        smb_fine=last_smb, smb_coarse=last_smb_,
        volumes=volumes,
    )
