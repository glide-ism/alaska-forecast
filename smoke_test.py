"""
Smoke test for the glacier_inverse library.

Builds a GlacierProblem from the wrangell config and runs a handful of
cheap consistency checks:

  1. Domain shape is factor-aligned (ny, nx divisible by 2**n_levels).
  2. Each MaternPrior was constructed with the hyperparameters in the config.
  3. GGaPPWhiten/GGaPPMap round-trip is approximate identity in both
     directions, for every field prior.
  4. Initial whitened parameters all live on CUDA with requires_grad=True.
  5. Observations are on CUDA and have shapes matching the domain.
  6. A single forward step at the coarsest level produces finite outputs and
     is differentiable end-to-end (gradient flows back to z_bed).

Prints PASS/FAIL per check; exits non-zero on any failure.

Run from the project root:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import traceback

import torch

from ggapp.torch import GGaPPMap, GGaPPWhiten

from configs.wrangell import WRANGELL
from glacier_inverse import GlacierProblem


_failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _failures
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not condition:
        _failures += 1


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    config = WRANGELL

    header("Building GlacierProblem")
    try:
        problem = GlacierProblem(config)
    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: failed to construct GlacierProblem: {e}")
        return 2
    print(f"  Built. ny={problem.ny}, nx={problem.nx}, dx={problem.dx}")

    header("1. Domain shape")
    factor = 2 ** config.n_levels
    check("ny is factor-aligned", problem.ny % factor == 0,
          f"ny={problem.ny}, factor={factor}")
    check("nx is factor-aligned", problem.nx % factor == 0,
          f"nx={problem.nx}, factor={factor}")
    check("dx > 0", problem.dx > 0, f"dx={problem.dx}")

    header("2. Prior hyperparameters match config")
    priors = problem.priors

    def _prior_matches(model, expected, label):
        got_sigma = float(model.mg.parameters.sigma.value)
        got_l = float(model.mg.parameters.l.value)
        got_nu = float(model.mg.parameters.nu.value)
        ok = (got_sigma == expected.sigma
              and got_l == expected.l
              and got_nu == float(expected.nu))
        check(f"{label} prior σ={expected.sigma}, ℓ={expected.l}, ν={expected.nu}",
              ok,
              f"got σ={got_sigma}, ℓ={got_l}, ν={got_nu}")

    _prior_matches(priors.bed_model,      config.bed_prior,      "bed")
    _prior_matches(priors.mean_model,     config.mean_prior,     "mean")
    _prior_matches(priors.log_beta_model, config.log_beta_prior, "log_beta")
    _prior_matches(priors.pbias_model,    config.pbias_prior,    "pbias")

    header("3. GGaPP whiten/map round-trip")
    rtol = 5e-2  # multigrid Matern solver is iterative — not exact

    def _round_trip(model, label):
        z = torch.randn(problem.ny, problem.nx, dtype=torch.float32, device="cuda")
        phys = GGaPPMap.apply(model, z)
        z_back = GGaPPWhiten.apply(model, phys)
        rel = (z_back - z).norm() / (z.norm() + 1e-12)
        check(f"{label}: ‖W(M(z)) − z‖ / ‖z‖ < {rtol}", rel.item() < rtol,
              f"relative error = {rel.item():.3e}")

    _round_trip(priors.bed_model,      "bed")
    _round_trip(priors.mean_model,     "mean")
    _round_trip(priors.log_beta_model, "log_beta")
    _round_trip(priors.pbias_model,    "pbias")

    header("4. Initial whitened parameters")
    params = problem.params
    for name, tensor in [
        ("z_bed",      params.z_bed),
        ("z_bed_mean", params.z_bed_mean),
        ("z_log_beta", params.z_log_beta),
        ("z_pbias",    params.z_pbias),
        ("z_log_mf",   params.z_log_mf),
        ("z_log_rf",   params.z_log_rf),
    ]:
        check(f"{name} on cuda + requires_grad",
              tensor.is_cuda and tensor.requires_grad,
              f"device={tensor.device}, requires_grad={tensor.requires_grad}")

    header("5. Observations")
    obs = problem.observations
    for name, tensor, expected_shape in [
        ("u_obs",       obs.u_obs,       (problem.ny, problem.nx)),
        ("v_obs",       obs.v_obs,       (problem.ny, problem.nx)),
        ("S_obs",       obs.S_obs,       (problem.ny, problem.nx)),
        ("rgi_mask",    obs.rgi_mask,    (problem.ny, problem.nx)),
        ("domain_mask", obs.domain_mask, (problem.ny, problem.nx)),
    ]:
        check(f"{name} shape == {expected_shape} and on cuda",
              tuple(tensor.shape) == expected_shape and tensor.is_cuda,
              f"got shape={tuple(tensor.shape)}, device={tensor.device}")
    check("bed_obs is 1-D on cuda",
          obs.bed_obs.ndim == 1 and obs.bed_obs.is_cuda,
          f"shape={tuple(obs.bed_obs.shape)}, device={obs.bed_obs.device}")
    check("bed_normed_coords is (N, 2) on cuda",
          obs.bed_normed_coords.ndim == 2
          and obs.bed_normed_coords.shape[1] == 2
          and obs.bed_normed_coords.is_cuda,
          f"shape={tuple(obs.bed_normed_coords.shape)}")

    if obs.snow_label is not None:
        dom = (problem.ny, problem.nx)
        check("snow_label shape == domain, in [0,1], on cuda",
              tuple(obs.snow_label.shape) == dom and obs.snow_label.is_cuda
              and obs.snow_label.min() >= 0.0 and obs.snow_label.max() <= 1.0,
              f"shape={tuple(obs.snow_label.shape)}, "
              f"range=[{obs.snow_label.min():.3f}, {obs.snow_label.max():.3f}]")
        check("snow_mask is 0/1 with some valid cells",
              tuple(obs.snow_mask.shape) == dom
              and set(obs.snow_mask.unique().tolist()) <= {0.0, 1.0}
              and obs.snow_mask.sum() > 0,
              f"valid cells = {int(obs.snow_mask.sum())}")
    else:
        print("  [SKIP] no snowline product for this domain")

    header("6. Single forward step at coarsest level")
    level = config.n_levels - 1  # 5 for default n_levels=6
    problem.model.set_top_level(level)
    try:
        sim, physical = problem.simulate(
            level=level,
            params=params,
            t_start=config.t_end - config.dt,   # exactly one step
            t_end=config.t_end,
        )
    except Exception as e:
        traceback.print_exc()
        check("simulate() runs one step", False, str(e))
        return 1 if _failures else 0

    H_finite = torch.isfinite(sim.H).all().item()
    u_finite = torch.isfinite(sim.u).all().item()
    v_finite = torch.isfinite(sim.v).all().item()
    check("sim.H finite",  H_finite)
    check("sim.u finite",  u_finite)
    check("sim.v finite",  v_finite)

    loss_terms = problem.compute_loss(
        sim=sim, physical=physical, params=params)
    J = loss_terms.J
    check("loss is finite", torch.isfinite(J).item(),
          f"J = {J.item():.4f}")
    check("snowline term is finite",
          torch.isfinite(torch.as_tensor(loss_terms.J_snow)).item(),
          f"J_snow = {float(loss_terms.J_snow):.4f}")
    if obs.snow_label is not None:
        check("snowline term is active (non-zero)",
              float(loss_terms.J_snow) != 0.0,
              f"J_snow = {float(loss_terms.J_snow):.4f}")

    J.backward()
    check("z_bed received a finite gradient",
          params.z_bed.grad is not None
          and torch.isfinite(params.z_bed.grad).all().item(),
          f"grad norm = {params.z_bed.grad.norm().item():.3e}"
          if params.z_bed.grad is not None else "no grad")

    print()
    if _failures:
        print(f"FAILED: {_failures} check(s) did not pass")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
