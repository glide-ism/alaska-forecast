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
    dhdt: Field


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
                            p_bias=_empty(), p_bias_total=_empty(), dhdt=_empty())


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
            "srf": diag.srf,
            "delta": diag.delta,
            "p_bias": diag.p_bias,
            "p_bias_total": diag.p_bias_total,
            "bed_mean": diag.bed_mean,
            "dhdt": diag.dhdt,
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
                             pbias_total_, dhdt_) -> None:
    """Copy detached tensors into the cupy-backed diagnostic Fields.

    `pbias_` is the spatial (Matern) log-precip bias; `pbias_total_` is the joint
    bias actually applied to precip (spatial minus the elevation-depletion ramp),
    equal to `pbias_` when precip_lapse_enabled is False. `dhdt_` is the model's
    coarse-grid surface elevation-change rate (m/yr) for this iterate, i.e.
    (H - H_prev) / dt at the current multigrid level.
    """
    diag.delta.data[:, :] = cp.asarray(S_.detach() - S_obs_)
    diag.srf.data[:, :] = cp.asarray(S_.detach())
    diag.bed_mean.data[:, :] = cp.asarray(bed_mean_.detach())
    diag.p_bias.data[:, :] = cp.asarray(pbias_.detach())
    diag.p_bias_total.data[:, :] = cp.asarray(pbias_total_.detach())
    diag.dhdt.data[:, :] = cp.asarray(dhdt_.detach())


def save_whitened_params(params, path: str, *, extras: dict = None) -> None:
    """Persist whitened parameter tensors. Keys match the historical format.

    Pass `extras` to attach additional payload (e.g., the noise vectors used
    by an RTO sample) under arbitrary keys without rebuilding the file format.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log_beta": params.z_log_beta,
        "bed": params.z_bed,
        "bed_mean": params.z_bed_mean,
        "precipitation_bias": params.z_pbias,
        "log_rf": params.z_log_rf,
        "log_mf": params.z_log_mf,
        "tau": params.z_tau,
        "z0": params.z_z0,
    }
    if extras:
        payload.update(extras)
    torch.save(payload, path)


def load_whitened_params_into(params, path: str) -> None:
    """In-place load: rebinds the existing parameter tensors so the optimizer
    (constructed afterwards) sees the warm-started values.
    """
    d = torch.load(path)
    params.z_log_beta = d["log_beta"].requires_grad_()
    params.z_bed = d["bed"].requires_grad_()
    params.z_bed_mean = d["bed_mean"].requires_grad_()
    params.z_pbias = d["precipitation_bias"].requires_grad_()
    params.z_log_rf = d["log_rf"].requires_grad_()
    params.z_log_mf = d["log_mf"].requires_grad_()
    # Precip-depletion scalars are newer than the original checkpoint format;
    # keep the freshly-initialized values when warm-starting from a MAP that
    # predates them.
    if "tau" in d:
        params.z_tau = d["tau"].requires_grad_()
    if "z0" in d:
        params.z_z0 = d["z0"].requires_grad_()
