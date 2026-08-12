"""
I/O helpers: VTI writer setup and whitened-parameter save/load.

Keeps the per-level diagnostic fields (delta, srf, bed_mean_field, p_bias_field)
together so the four drivers don't have to assemble them by hand.
"""
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import torch

from glide.field import Field, GridEntity
from glide.io import VTIWriter


@dataclass
class DiagnosticFields:
    delta: Field
    srf: Field
    bed_mean: Field
    p_bias: Field          # spatial (Matern) log-precip bias only
    p_bias_total: Field    # joint bias: spatial minus elevation-depletion ramp
    t_bias: Field          # additive temperature bias (K); zeros when disabled
    dhdt: Field
    bed_grad: Field        # dJ/d(bed), physical space; zeros before backward


def make_diagnostic_fields(mg_level) -> DiagnosticFields:
    """Allocate the diagnostic Fields on a multigrid level."""
    def _empty():
        return Field(
            cp.zeros((mg_level.ny, mg_level.nx), dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg_level.dx,
            grid=mg_level,
        )
    return DiagnosticFields(delta=_empty(), srf=_empty(), bed_mean=_empty(),
                            p_bias=_empty(), p_bias_total=_empty(),
                            t_bias=_empty(), dhdt=_empty(), bed_grad=_empty())


def make_loss_vti_writer(mg_level, output_dir: str, base: str, diag: DiagnosticFields) -> VTIWriter:
    """VTI writer for per-iteration loss diagnostics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    writer = VTIWriter(
        output_dir, base=base, dx=mg_level.dx,
        dynamic_fields={
            "bed": mg_level.geometry.bed,
            "beta": mg_level.sliding.beta,
            "thk": mg_level.state.H,
            "U": [mg_level.state.u, mg_level.state.v],
            "xi": mg_level.state.xi,
            "srf": diag.srf,
            "delta": diag.delta,
            "p_bias": diag.p_bias,
            "p_bias_total": diag.p_bias_total,
            "t_bias": diag.t_bias,
            "bed_mean": diag.bed_mean,
            "dhdt": diag.dhdt,
            "bed_grad": diag.bed_grad,
            "smb": mg_level.forcing.smb,
        },
    )
    writer.initialize(mg_level)
    return writer


def make_time_vti_writer(mg_level, output_dir: str, base: str = "time") -> VTIWriter:
    """VTI writer for per-time-step diagnostics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return VTIWriter(
        output_dir, base=base, dx=mg_level.dx,
        dynamic_fields={
            "thk": mg_level.state.H,
            "U": [mg_level.state.u, mg_level.state.v],
            "smb": mg_level.forcing.smb,
        },
    )


def _as_cell_field(arr, mg_level) -> Field:
    """Wrap a 2-D array (torch / cupy / numpy) as a cell-centered Field on
    `mg_level`. Torch tensors are detached and moved through the CUDA array
    interface; everything is cast to float32 so VTI export is uniform."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().to(torch.float32)
    data = cp.asarray(arr, dtype=cp.float32)
    return Field(data, grid_entity=GridEntity.CELL, dx=mg_level.dx, grid=mg_level)


def write_static_vti(mg_level, output_dir: str, base: str,
                     scalar_fields: dict, vector_fields: dict = None) -> None:
    """Write a single-frame PVD of static (non-evolving) fields.

    `scalar_fields` maps name -> 2-D array; `vector_fields` maps name ->
    (comp_x, comp_y) 2-D arrays. Used to dump the observational products once
    so they can be flipped through alongside the per-iteration diagnostics.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dynamic_fields = {
        name: _as_cell_field(arr, mg_level)
        for name, arr in scalar_fields.items()
    }
    for name, (cx, cy) in (vector_fields or {}).items():
        dynamic_fields[name] = [_as_cell_field(cx, mg_level),
                                _as_cell_field(cy, mg_level)]
    writer = VTIWriter(output_dir, base=base, dx=mg_level.dx,
                       dynamic_fields=dynamic_fields)
    writer.initialize(mg_level)
    writer.append(mg_level, time=0.0)
    writer.write_pvd()


def update_diagnostic_fields(diag: DiagnosticFields, S_, S_obs_, bed_mean_, pbias_,
                             pbias_total_, dhdt_, tbias_=None,
                             bed_grad_=None) -> None:
    """Copy detached tensors into the cupy-backed diagnostic Fields.

    `pbias_` is the spatial (Matern) log-precip bias; `pbias_total_` is the joint
    bias actually applied to precip (spatial minus the elevation-depletion ramp),
    equal to `pbias_` when precip_lapse_enabled is False. `tbias_` is the
    additive temperature bias (K); None (the term disabled, or a caller that
    predates it) writes zeros. `dhdt_` is the model's
    coarse-grid surface elevation-change rate (m/yr) for this iterate over the
    same interval the dhdt misfit uses — DhdtObservation.model_rate(sim,
    "coarse"): the observation window (H(t1) - H(t0))/(t1 - t0) in two-snapshot
    mode, or the true final step in legacy/no-product mode. `bed_grad_` is the
    physical-space bed gradient dJ/d(bed) restricted to the writer's level
    (available only after backward; None writes zeros).
    """
    diag.delta.data[:, :] = cp.asarray(S_.detach() - S_obs_)
    diag.srf.data[:, :] = cp.asarray(S_.detach())
    diag.bed_mean.data[:, :] = cp.asarray(bed_mean_.detach())
    diag.p_bias.data[:, :] = cp.asarray(pbias_.detach())
    diag.p_bias_total.data[:, :] = cp.asarray(pbias_total_.detach())
    if tbias_ is None:
        diag.t_bias.data[:, :] = 0.0
    else:
        diag.t_bias.data[:, :] = cp.asarray(tbias_.detach())
    diag.dhdt.data[:, :] = cp.asarray(dhdt_.detach())
    if bed_grad_ is None:
        diag.bed_grad.data[:, :] = 0.0
    else:
        diag.bed_grad.data[:, :] = cp.asarray(bed_grad_.detach())


def save_whitened_params(params, path: str, *, extras: dict = None,
                         bed_parametrization: str = "legacy") -> None:
    """Persist whitened parameter tensors. Keys match the historical format.

    Pass `extras` to attach additional payload (e.g., the noise vectors used
    by an RTO sample) under arbitrary keys without rebuilding the file format.
    `bed_parametrization` tags how z_bed is to be interpreted ("legacy":
    bed = Map(z_bed); "conditioned": bed = condition(bed_mean + Map(z_bed)))
    — drivers pass `problem.priors.bed_parametrization`. Untagged historical
    checkpoints read as "legacy".
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log_beta": params.z_log_beta,
        "bed": params.z_bed,
        "bed_mean": params.z_bed_mean,
        "precipitation_bias": params.z_pbias,
        "temperature_bias": params.z_tbias,
        "log_rf": params.z_log_rf,
        "log_mf": params.z_log_mf,
        "tau": params.z_tau,
        "z0": params.z_z0,
        "log_H_atm": params.z_log_H_atm,
        "logit_cloud": params.z_logit_cloud,
        "bed_parametrization": bed_parametrization,
    }
    if extras:
        payload.update(extras)
    torch.save(payload, path)


def _convert_z_bed(z_bed, saved: str, current: str, priors):
    """z_bed conversion between bed parametrizations.

    legacy -> conditioned: the IDENTITY on z_bed. The fluctuation field is
    reinterpreted under the conditioned map, which corrects it onto the bed
    data at data cells and leaves it untouched elsewhere — exactly what a
    warm start wants. (The algebraically 'exact' pre-image u0 = (I+C·D)(bed −
    mu_b) is deliberately NOT used: it amplifies the legacy bed's
    data-cell violations by (sigma_prior/sigma_obs)^2 ~ 1e3 and float32
    cannot survive the cancellation — and faithfully reproducing a
    data-violating bed would defeat the point of conditioning anyway.) The
    rms shift the correction will apply at data cells is logged so the
    warm-start discontinuity is visible.

    conditioned -> legacy: z_new = Whiten(condition(Map(z_old))) — the
    physical bed is preserved (benign direction, no amplification).
    """
    cond = priors.bed_conditioner
    if cond is None:
        # Interpreting a conditioned checkpoint requires the conditioner (its
        # data/precision fields) that defined the saved parametrization.
        raise ValueError(
            f"converting a {saved!r} checkpoint to {current!r} requires a "
            f"GlacierPriors built with bed_conditioning.enabled=True (the "
            f"conditioning data defines the saved parametrization).")
    z_old = cp.asarray(z_bed.detach())
    if saved == "legacy":       # -> conditioned: identity + diagnostic
        bed_old = priors.bed_model.forward(z_old)
        bed_new = cond.condition(bed_old)
        at_data = cond.precision > 0
        rms = float(cp.sqrt(((bed_new - bed_old)[at_data] ** 2).mean()))
        print(f"[io] z_bed reinterpreted legacy -> conditioned (identity); "
              f"the kriging correction moves the warm-start bed by "
              f"{rms:.1f} m rms at data cells")
        return torch.tensor(z_old)
    # conditioned -> legacy
    bed_target = cond.condition(priors.bed_model.forward(z_old))
    z_new = priors.bed_model.whiten(bed_target)
    bed_check = priors.bed_model.forward(z_new)
    err = float(cp.linalg.norm(bed_check - bed_target)
                / cp.linalg.norm(bed_target))
    print(f"[io] converted z_bed conditioned -> legacy "
          f"(bed reconstruction rel. error {err:.2e})")
    return torch.tensor(z_new)


def load_whitened_params_into(params, path: str, *, priors=None) -> None:
    """In-place load: rebinds the existing parameter tensors so the optimizer
    (constructed afterwards) sees the warm-started values.

    Pass `priors` (GlacierPriors) to enable exact conversion of z_bed when the
    checkpoint's bed parametrization differs from the current one; without it,
    a mismatched load raises rather than silently misinterpreting z_bed.
    """
    d = torch.load(path)
    saved = d.get("bed_parametrization", "legacy")
    current = priors.bed_parametrization if priors is not None else "legacy"
    params.z_log_beta = d["log_beta"].requires_grad_()
    if saved == current:
        params.z_bed = d["bed"].requires_grad_()
    elif priors is None:
        raise ValueError(
            f"checkpoint {path} was saved with bed_parametrization={saved!r} "
            f"but is being loaded as {current!r}; pass priors= to "
            f"load_whitened_params_into for exact conversion.")
    else:
        params.z_bed = _convert_z_bed(
            d["bed"], saved, current, priors).requires_grad_()
    params.z_bed_mean = d["bed_mean"].requires_grad_()
    params.z_pbias = d["precipitation_bias"].requires_grad_()
    params.z_log_rf = d["log_rf"].requires_grad_()
    params.z_log_mf = d["log_mf"].requires_grad_()
    # The temperature-bias field, precip-depletion and enthalpy-model scalars
    # are newer than the original checkpoint format; keep the freshly-
    # initialized values (prior median) when warm-starting from a MAP that
    # predates them.
    if "temperature_bias" in d:
        params.z_tbias = d["temperature_bias"].requires_grad_()
    if "tau" in d:
        params.z_tau = d["tau"].requires_grad_()
    if "z0" in d:
        params.z_z0 = d["z0"].requires_grad_()
    if "log_H_atm" in d:
        params.z_log_H_atm = d["log_H_atm"].requires_grad_()
    if "logit_cloud" in d:
        params.z_logit_cloud = d["logit_cloud"].requires_grad_()
