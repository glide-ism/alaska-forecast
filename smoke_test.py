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
  8. The enthalpy SMB backend (smb_model="enthalpy") builds, runs, is
     deterministic (fixed weather realization), and is differentiable w.r.t.
     its two whitened scalars (skipped if the installed glare predates it).

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
from glacier_inverse.forward import year_overlap_weights
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

    # The optional temperature-bias prior: no hierarchy when disabled; a
    # priors-only build (no IceDynamics) exercises the enabled path cheaply.
    if priors.tbias_model is not None:
        _prior_matches(priors.tbias_model, config.tbias_prior, "tbias")
        _round_trip(priors.tbias_model, "tbias")
    else:
        check("tbias prior absent while disabled",
              priors.tbias_model is None)
        import dataclasses
        from glacier_inverse.priors import GlacierPriors as _GP
        tpriors = _GP(dataclasses.replace(config, tbias_enabled=True),
                      problem.ny, problem.nx, problem.dx)
        _prior_matches(tpriors.tbias_model, config.tbias_prior, "tbias (enabled)")
        _round_trip(tpriors.tbias_model, "tbias (enabled)")
        del tpriors

    header("4. Initial whitened parameters")
    params = problem.params
    for name, tensor in [
        ("z_bed",      params.z_bed),
        ("z_bed_mean", params.z_bed_mean),
        ("z_log_beta", params.z_log_beta),
        ("z_pbias",    params.z_pbias),
        ("z_tbias",    params.z_tbias),
        ("z_log_mf",   params.z_log_mf),
        ("z_log_rf",   params.z_log_rf),
        ("z_log_H_atm",   params.z_log_H_atm),
        ("z_logit_cloud", params.z_logit_cloud),
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

    # Anomaly-integration quadrature weights (pure CPU).
    w = year_overlap_weights(1992.0, 2012.0)
    check("year weights: 20 uniform years over a 20-yr step",
          len(w) == 20 and w[0] == (1992, 0.05) and w[-1] == (2011, 0.05)
          and abs(sum(x for _, x in w) - 1.0) < 1e-12)
    w = year_overlap_weights(2011.5, 2013.0)
    check("year weights: partial-year overlap",
          [y for y, _ in w] == [2011, 2012]
          and abs(w[0][1] - 0.5 / 1.5) < 1e-12
          and abs(w[1][1] - 1.0 / 1.5) < 1e-12)
    sub = {y: wt * 8.0 for y, wt in year_overlap_weights(1995.25, 2003.25)}
    for y, wt in year_overlap_weights(2003.25, 2012.75):
        sub[y] = sub.get(y, 0.0) + wt * 9.5
    whole = dict(year_overlap_weights(1995.25, 2012.75))
    check("year weights: sub-partitions recombine (partition independence)",
          set(sub) == set(whole)
          and all(abs(sub[y] / 17.5 - whole[y]) < 1e-12 for y in whole))

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

    header("8. Enthalpy SMB backend")
    try:
        from glare.enthalpy import EnthalpyModel  # noqa: F401
        enthalpy_available = True
    except ImportError:
        enthalpy_available = False
    if not enthalpy_available:
        print("  [SKIP] installed glare does not provide EnthalpyModel")
    else:
        # Free the temperature-index problem's graph/state before standing up
        # a second full problem on the same GPU.
        del sim, physical, loss_terms, probe, J
        del problem, params, srf, vel, bed, snow, dhdt, domain, priors
        torch.cuda.empty_cache()

        import dataclasses
        econfig = dataclasses.replace(config, smb_model="enthalpy")
        try:
            eproblem = GlacierProblem(econfig)
        except Exception as e:
            traceback.print_exc()
            check("enthalpy GlacierProblem builds", False, str(e))
            print()
            print(f"FAILED: {_failures} check(s) did not pass")
            return 1
        check("smb model is EnthalpyModel",
              type(eproblem.smb_model).__name__ == "EnthalpyModel")
        check("initial enthalpy smb finite",
              bool(torch.isfinite(torch.tensor(
                  eproblem.smb_model.grid.state.smb.data)).all()))
        has_dif = "monthly_diffuse_potential" in eproblem.gridded_data
        dif = eproblem.insol_dif
        check("diffuse-sky potential present in GLIDE_inputs.nc "
              "(else the diffuse term is zero)", has_dif)
        check("insol_dif is (12, ny, nx), finite, in [0, 1]",
              tuple(dif.shape) == (12,) + tuple(eproblem.domain.dem.shape)
              and torch.isfinite(dif).all().item()
              and dif.min().item() >= 0.0 and dif.max().item() <= 1.0 + 1e-6,
              f"shape={tuple(dif.shape)} range=[{dif.min().item():.3f}, {dif.max().item():.3f}]")

        eparams = eproblem.params
        elevel = econfig.n_levels - 1
        eproblem.model.set_top_level(elevel)
        erequired = eproblem.required_times
        et_first = min(erequired) if erequired else econfig.t_end
        et_start = et_first - 3 * econfig.dt
        try:
            esim, ephys = eproblem.simulate(level=elevel, params=eparams,
                                            t_start=et_start)
        except Exception as e:
            traceback.print_exc()
            check("enthalpy simulate() runs", False, str(e))
            print()
            print(f"FAILED: {_failures} check(s) did not pass")
            return 1
        check("enthalpy sim.H finite", torch.isfinite(esim.H).all().item())
        check("enthalpy smb_fine finite",
              torch.isfinite(esim.final.smb_fine).all().item())

        # The fixed weather realization: the grid must hold exactly the seeded
        # deviation sequence after a run (an unseeded redraw would differ), and
        # a re-run with the same physical parameters must reproduce smb to well
        # under the O(0.1 m/yr) a redraw would cause. Bitwise equality is only
        # spoiled by the avalanche operator's atomicAdd deposits (~1e-5,
        # order-nondeterministic, pre-existing on the ETIM path too); without
        # the avalanche model the re-run is exact.
        import cupy as _cp
        import numpy as _np
        check("grid.temp_dev holds the fixed seeded realization",
              _np.array_equal(_cp.asnumpy(eproblem.smb_model.grid.temp_dev),
                              eproblem.temp_dev))
        with torch.no_grad():
            esim2 = eproblem.simulate_physical(level=elevel, physical=ephys,
                                               t_start=et_start)
        smb_diff = (esim.final.smb_fine - esim2.final.smb_fine).abs().max()
        exact = torch.equal(esim.final.smb_fine, esim2.final.smb_fine)
        check("re-run smb matches (fixed realization, no redraw)",
              exact or (econfig.use_avalanche_model
                        and smb_diff.item() < 1e-3),
              f"max |Δsmb| = {smb_diff.item():.3e}"
              + ("" if exact else " (avalanche atomicAdd jitter)"))
        del esim2

        eloss = eproblem.compute_loss(sim=esim, physical=ephys, params=eparams)
        eJ = eloss.J
        check("enthalpy loss finite", torch.isfinite(eJ).item(),
              f"J = {eJ.item():.4f}")
        eJ.backward()
        for name, tensor in [("z_log_H_atm", eparams.z_log_H_atm),
                             ("z_logit_cloud", eparams.z_logit_cloud),
                             ("z_bed", eparams.z_bed)]:
            check(f"gradient reaches {name}",
                  tensor.grad is not None
                  and torch.isfinite(tensor.grad).all().item()
                  and tensor.grad.abs().sum() > 0,
                  f"|grad|_1 = {tensor.grad.abs().sum().item():.3e}"
                  if tensor.grad is not None else "no grad")
        # The inactive model's scalar sits at z = 0, so its only gradient path
        # (the prior term) contributes exactly zero — no data-loss leakage.
        check("z_log_mf gradient is exactly zero (inactive model)",
              eparams.z_log_mf.grad is None
              or float(eparams.z_log_mf.grad.abs()) == 0.0,
              f"grad = {float(eparams.z_log_mf.grad):.3e}"
              if eparams.z_log_mf.grad is not None else "no grad")

    header("9. Bed GP conditioning (posterior-as-prior)")
    try:
        from ggapp.conditioning import ConditionedPrior  # noqa: F401
        from ggapp.torch import GGaPPCondition  # noqa: F401
        conditioning_available = True
    except ImportError:
        conditioning_available = False
    if not conditioning_available:
        print("  [SKIP] installed ggapp predates conditioning "
              "(no ggapp.conditioning)")
    else:
        import dataclasses
        import cupy as _cp
        from glacier_inverse.config import BedConditioningConfig
        from glacier_inverse.priors import (
            GlacierPriors, _cropped_inputs, build_bed_conditioning_data)

        # All file IO happens BEFORE the enthalpy-problem teardown: on this
        # HDF5 stack, opening a netCDF after other handles on it have been
        # garbage-collected can segfault (hence also the eager loads in
        # priors._cropped_inputs).
        ccfg = dataclasses.replace(
            config, bed_conditioning=BedConditioningConfig(enabled=True))
        gd = _cropped_inputs(
            ccfg, variables=["elevation", "rgi_mask", "domain_mask"])
        cny, cnx = gd.sizes["y"], gd.sizes["x"]
        cdx = (gd.x[1] - gd.x[0]).item()
        cpriors = GlacierPriors(ccfg, cny, cnx, cdx)
        _, D_picks = build_bed_conditioning_data(dataclasses.replace(
            ccfg, bed_conditioning=BedConditioningConfig(
                enabled=True, include_off_ice=False)))
        dem = torch.tensor(gd.elevation.values, dtype=torch.float32,
                           device="cuda")
        off_ice = torch.tensor((gd.rgi_mask.values == 0)
                               | (gd.domain_mask.values == 0)).cuda()

        if enthalpy_available:
            del esim, ephys, eloss, eJ, eparams, eproblem
            torch.cuda.empty_cache()

        cond = cpriors.bed_conditioner
        check("conditioner is built", cond is not None)
        check("pcg knobs reached the layer",
              cond.rtol == ccfg.bed_conditioning.pcg_rtol
              and cond.maxiter == ccfg.bed_conditioning.pcg_maxiter)
        n_pick_cells = int((D_picks > 0).sum())
        check("flightline picks landed on the grid", n_pick_cells > 100,
              f"{n_pick_cells} pick cells")

        # z = 0 must give the kriging posterior mean: ~DEM off-ice, ~picks at
        # pick cells (within a few sigma; the data dominate the flat prior).
        z0 = torch.zeros(cny, cnx, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            bed_mean_krig = cpriors.bed_from_whitened(z0, z0)[0]
        check("PCG converged (cold solve)", cond.last_converged,
              f"{cond.last_iters} iterations")
        off_err = (bed_mean_krig - dem)[off_ice].abs().mean().item()
        check("z=0 ⇒ kriging mean ≈ DEM off-ice",
              off_err < ccfg.bed_conditioning.sigma_dem,
              f"mean |bed − DEM| off-ice = {off_err:.2f} m")
        at_picks = torch.tensor(D_picks > 0).cuda()
        b_full = torch.tensor(cond.data)
        pick_err = (bed_mean_krig - b_full)[at_picks].abs()
        check("kriging mean honors the picks",
              pick_err.mean().item() < ccfg.bed_conditioning.sigma_picks
              and pick_err.max().item() < 5 * ccfg.bed_conditioning.sigma_picks,
              f"mean/max |bed − b| at picks = {pick_err.mean().item():.2f}"
              f"/{pick_err.max().item():.2f} m")

        # Conditional round-trip via the checkpoint-conversion algebra
        # (S^{-1} = I + C·D): z -> bed -> z -> bed. Measured in BED space —
        # that is what a converted warm start must reproduce. (The z-space
        # error is amplified by ||S^{-1}|| ~ (sigma_prior/sigma_obs)^2 and is
        # meaninglessly large at solver tolerance.)
        zr = torch.randn(cny, cnx, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            bed_r = cpriors.bed_from_whitened(zr, z0)[0]
        x_rec = cond.latent_from_field(
            _cp.asarray(bed_r), _cp.zeros((cny, cnx), dtype=_cp.float32))
        z_rec = torch.tensor(cpriors.bed_model.whiten(x_rec))
        with torch.no_grad():
            bed_rec = cpriors.bed_from_whitened(z_rec, z0)[0]
        rt_rel = ((bed_rec - bed_r).norm() / bed_r.norm()).item()
        check("conditional round-trip (conversion algebra, bed space) < 5e-2",
              rt_rel < 5e-2, f"relative bed error = {rt_rel:.3e}")

        # Conditional variance: collapsed at data, ~prior sigma far away
        # (6 samples — a loose, cheap check).
        with torch.no_grad():
            draws = torch.stack([
                cpriors.bed_from_whitened(
                    torch.randn(cny, cnx, dtype=torch.float32,
                                device="cuda"), z0)[0]
                for _ in range(6)])
        sd = draws.std(dim=0)
        sd_data = sd[off_ice].mean().item()
        check("posterior sd collapsed at data cells",
              sd_data < 5 * ccfg.bed_conditioning.sigma_dem,
              f"mean sd off-ice = {sd_data:.1f} m (prior "
              f"σ = {ccfg.bed_prior.sigma})")
        from scipy.ndimage import distance_transform_edt
        dist_px = distance_transform_edt(
            _cp.asnumpy(cond.precision) == 0)
        far = torch.tensor(dist_px > 3 * ccfg.bed_prior.l / cdx).cuda()
        if bool(far.any()):
            sd_far = sd[far].mean().item()
            check("posterior sd ≈ prior σ far from data",
                  0.3 * ccfg.bed_prior.sigma < sd_far < 1.8 * ccfg.bed_prior.sigma,
                  f"mean sd far = {sd_far:.1f} m (prior σ = {ccfg.bed_prior.sigma},"
                  f" 6-sample estimate)")
        else:
            print("  [SKIP] no cells farther than 3ℓ from data on this domain")

        # Gradient flows through the conditioning and differs from the
        # unconditional map's gradient.
        zg = torch.randn(cny, cnx, dtype=torch.float32, device="cuda",
                         requires_grad=True)
        wq = torch.randn(cny, cnx, dtype=torch.float32, device="cuda")
        bed_g = cpriors.bed_from_whitened(zg, z0)[0]
        (wq * bed_g).sum().backward()
        g_cond = zg.grad.clone()
        zg2 = zg.detach().clone().requires_grad_()
        (wq * GGaPPMap.apply(cpriors.bed_model, zg2)).sum().backward()
        g_diff = (g_cond - zg2.grad).norm() / zg2.grad.norm()
        check("gradient through conditioning finite & nonzero",
              torch.isfinite(g_cond).all().item()
              and g_cond.abs().sum().item() > 0)
        check("conditioned gradient differs from unconditional",
              g_diff.item() > 1e-3,
              f"relative difference = {g_diff.item():.3e}")

    print()
    if _failures:
        print(f"FAILED: {_failures} check(s) did not pass")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
