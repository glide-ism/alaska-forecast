"""
Smoke test for the glacier_inverse library.

Builds a GlacierProblem from the wrangell config and runs a handful of
cheap consistency checks:

  1. Domain shape is factor-aligned (ny, nx divisible by 2**n_levels).
  2. Each MaternPrior was constructed with the hyperparameters in the config.
  3. GGaPPWhiten/GGaPPMap round-trip is approximate identity in both
     directions, for every field prior.
  4. Initial whitened parameters all live on CUDA with requires_grad=True.
  5. The step scheduler snaps onto required times, extends the horizon, and
     rejects impossible requests (pure CPU checks).
  6. Observations built from the config specs are on CUDA, shaped to the
     domain, and carry acquisition times (from file attrs or the t_end
     fallback).
  7. A short forward run at the coarsest level emits a state snapshot at
     every required observation time, produces finite loss terms, and is
     differentiable end-to-end — including through a non-final snapshot.

Prints PASS/FAIL per check; exits non-zero on any failure.

Run from the project root:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import traceback

import torch

from ggapp.torch import GGaPPMap, GGaPPWhiten

from glacier_inverse import GlacierProblem, load_config
from glacier_inverse.scheduling import build_step_sequence, merge_times

# Domain to smoke-test. Override at the call site if you want a different one.
SMOKE_DOMAIN = "domains/wrangell"


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
    config = load_config(SMOKE_DOMAIN)

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
        try:
            got_sigma = float(model.mg.parameters.sigma.value)
            got_l = float(model.mg.parameters.l.value)
            got_nu = float(model.mg.parameters.nu.value)
        except AttributeError:
            print(f"  [SKIP] {label}: installed ggapp does not expose "
                  f"mg.parameters.*.value; hyperparameters not introspectable")
            return
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

    header("5. Step scheduler")
    steps = build_step_sequence(t_start=1012.0, t_end=2012.0, dt_max=20.0)
    check("uniform legacy grid: 50 steps of dt=20 ending at 2012",
          len(steps) == 50 and steps[-1] == (2012.0, 20.0),
          f"n={len(steps)}, last={steps[-1]}")

    req = [2000.0, 2013.0, 2015.0, 2020.0]
    steps = build_step_sequence(t_start=1012.0, t_end=2012.0, dt_max=20.0,
                                required_times=req)
    times = [t for t, _ in steps]
    check("required times snapped exactly; horizon extended past t_end",
          all(r in times for r in req) and times[-1] == 2020.0,
          f"tail={times[-5:]}")
    check("all dt positive and <= dt_max",
          all(0 < dt <= 20.0 + 1e-9 for _, dt in steps))

    try:
        build_step_sequence(t_start=1012.0, t_end=2012.0, dt_max=20.0,
                            required_times=[1000.0])
        check("required time <= t_start raises", False)
    except ValueError:
        check("required time <= t_start raises", True)

    check("merge_times dedupes within eps",
          merge_times([2000.0, 2000.0 + 1e-9], [2013.0], None)
          == (2000.0, 2013.0))

    header("6. Observations")
    obs_names = [o.name for o in problem.observations]
    print(f"  products: {obs_names}")
    check("at least srf/vel/extent/bed present",
          {"srf", "vel", "extent", "bed"} <= set(obs_names))

    dom_shape = (problem.ny, problem.nx)
    domain = problem.domain
    for name, tensor in [("dem", domain.dem),
                         ("rgi_mask", domain.rgi_mask),
                         ("domain_mask", domain.domain_mask)]:
        check(f"domain.{name} shape == {dom_shape} and on cuda",
              tuple(tensor.shape) == dom_shape and tensor.is_cuda,
              f"got shape={tuple(tensor.shape)}, device={tensor.device}")

    srf = problem.get_observation("srf")
    vel = problem.get_observation("vel")
    check("S_obs shape == domain and on cuda",
          tuple(srf.S_obs.shape) == dom_shape and srf.S_obs.is_cuda)
    check("u_obs/v_obs shape == domain and on cuda",
          tuple(vel.u_obs.shape) == dom_shape and vel.u_obs.is_cuda
          and tuple(vel.v_obs.shape) == dom_shape)

    bed = problem.get_observation("bed")
    check("bed_obs is 1-D on cuda",
          bed.bed_obs.ndim == 1 and bed.bed_obs.is_cuda,
          f"shape={tuple(bed.bed_obs.shape)}")
    check("bed_normed_coords is (N, 2) on cuda",
          bed.bed_normed_coords.ndim == 2
          and bed.bed_normed_coords.shape[1] == 2
          and bed.bed_normed_coords.is_cuda,
          f"shape={tuple(bed.bed_normed_coords.shape)}")
    check("bed observation is time-independent", bed.required_times == ())

    snow = problem.get_observation("snow")
    if snow is not None:
        check("snow_label shape == domain, in [0,1], on cuda",
              tuple(snow.snow_label.shape) == dom_shape and snow.snow_label.is_cuda
              and snow.snow_label.min() >= 0.0 and snow.snow_label.max() <= 1.0,
              f"range=[{snow.snow_label.min():.3f}, {snow.snow_label.max():.3f}]")
        check("snow_mask is 0/1 with some valid cells",
              set(snow.snow_mask.unique().tolist()) <= {0.0, 1.0}
              and snow.snow_mask.sum() > 0,
              f"valid cells = {int(snow.snow_mask.sum())}")
    else:
        print("  [SKIP] no snowline product for this domain")

    dhdt = problem.get_observation("dhdt")
    if dhdt is not None:
        mode = ("legacy final-step" if dhdt.legacy_final_step
                else f"two-snapshot [{dhdt.t0}, {dhdt.t1}]")
        print(f"  dhdt mode: {mode}")
    else:
        print("  [SKIP] no dhdt product for this domain")

    required = problem.required_times
    check("every timed observation's epoch is in required_times",
          all(t in required for o in problem.observations
              for t in o.required_times),
          f"required_times={required}")

    header("7. Short forward run at coarsest level")
    level = config.n_levels - 1  # 5 for default n_levels=6
    problem.model.set_top_level(level)
    # Start a few coarse steps before the first observation epoch so the run
    # covers every required time; also request one mid-run snapshot to test
    # non-final-state emission even when all products sit at a single epoch.
    t_first = min(required) if required else config.t_end
    t_start = t_first - 3 * config.dt
    t_probe = t_first - config.dt
    try:
        sim, physical = problem.simulate(
            level=level,
            params=params,
            t_start=t_start,
            record_states_at=list(required) + [t_probe],
        )
    except Exception as e:
        traceback.print_exc()
        check("simulate() runs", False, str(e))
        return 1 if _failures else 0

    check("a state was recorded at every required time",
          all(any(abs(ts - t) < 1e-6 for ts in sim.states) for t in required),
          f"recorded={sorted(sim.states.keys())}")
    check("run reached the horizon",
          sim.final.t >= max([config.t_end] + list(required)) - 1e-6,
          f"final t={sim.final.t}")
    check("sim.H finite", torch.isfinite(sim.H).all().item())
    check("sim.u finite", torch.isfinite(sim.u).all().item())
    check("sim.v finite", torch.isfinite(sim.v).all().item())

    # Gradient through a non-final snapshot only.
    probe = sim.at(t_probe).H_fine.sum()
    probe.backward(retain_graph=True)
    check("gradient reaches z_bed through a NON-final snapshot",
          params.z_bed.grad is not None
          and torch.isfinite(params.z_bed.grad).all().item()
          and params.z_bed.grad.abs().sum() > 0,
          f"|grad|_1 = {params.z_bed.grad.abs().sum().item():.3e}"
          if params.z_bed.grad is not None else "no grad")
    params.z_bed.grad = None

    loss_terms = problem.compute_loss(
        sim=sim, physical=physical, params=params)
    J = loss_terms.J
    check("loss is finite", torch.isfinite(J).item(), f"J = {J.item():.4f}")
    for name, term in loss_terms.data_terms.items():
        check(f"{name} term is finite",
              torch.isfinite(torch.as_tensor(term)).item(),
              f"J_{name} = {float(term):.4f}")
    if snow is not None:
        check("snowline term is active (non-zero)",
              float(loss_terms.data_terms["snow"]) != 0.0)
    if dhdt is not None:
        check("dhdt term is active (non-zero)",
              float(loss_terms.data_terms["dhdt"]) != 0.0)

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
