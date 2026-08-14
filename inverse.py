"""
Deterministic MAP solve in whitened coordinates.

Thin driver over glacier_inverse. All physical hyperparameters live in
domains/<name>/config.py — per-task knobs (output path, max iters,
learning rates) stay here.
"""
import cupy as cp
import torch
import xarray as xr

from glacier_inverse import GlacierProblem, load_config
from glacier_inverse.forward import differentiable_restriction
from glacier_inverse.io import (
    load_whitened_params_into, make_diagnostic_fields, make_loss_vti_writer,
    make_time_vti_writer, save_whitened_params, update_diagnostic_fields,
)

# Available domains: domains/{chugach,delta,denali,juneau,st_elias,wrangell}
DOMAIN = "domains/wrangell"
config = load_config(DOMAIN)

OUTPUT_PATH = config.output_dir
WARM_START_PATH = None  # e.g. f"{OUTPUT_PATH}/level_0/torch_vars.p"
#WARM_START_PATH = f"{DOMAIN}/inverse_molho/level_0/torch_vars.p"

problem = GlacierProblem(config)
params = problem.params

if WARM_START_PATH is not None:
    load_whitened_params_into(params, WARM_START_PATH, priors=problem.priors)

# Dump the observational products once, alongside the finest-level diagnostics.
problem.write_observations(f"{OUTPUT_PATH}/level_{config.min_level}/vti")

optimizer_sgd = torch.optim.SGD([
    {"params": params.z_bed,      "lr": config.lr_z_bed},
    {"params": params.z_bed_mean, "lr": config.lr_z_bed_mean},
    {"params": params.z_log_beta, "lr": config.lr_z_log_beta},
], momentum=0.5)

adam_groups = [
    {"params": params.z_pbias,  "lr": config.lr_z_pbias},
]
if config.tbias_enabled:
    # Additive temperature bias field; inert (z = 0) when disabled.
    adam_groups += [
        {"params": params.z_tbias, "lr": config.lr_z_tbias},
    ]
if config.smb_model == "enthalpy":
    # Enthalpy SMB scalars replace the temperature-index mf/rf pair — the
    # inactive model's z never enter the forward graph and get no gradient.
    adam_groups += [
        {"params": params.z_log_H_atm,   "lr": config.lr_z_log_H_atm},
        {"params": params.z_logit_cloud, "lr": config.lr_z_logit_cloud},
    ]
else:
    adam_groups += [
        {"params": params.z_log_mf, "lr": config.lr_z_log_mf},
        {"params": params.z_log_rf, "lr": config.lr_z_log_rf},
    ]
if config.precip_lapse_enabled:
    # Elevation-dependent precip depletion: learn tau/z0 in the Adam block.
    adam_groups += [
        {"params": params.z_tau, "lr": config.lr_z_tau},
        {"params": params.z_z0,  "lr": config.lr_z_z0},
    ]
optimizer_adam = torch.optim.Adam(adam_groups, betas=(0.5, 0.99))

def write_loss_vti(diag, vti_writer, sim, physical, level, i):
    bed_mean_coarse = differentiable_restriction(physical.bed_mean, level)
    pbias_coarse = differentiable_restriction(physical.pbias, level)
    pbias_total_coarse = differentiable_restriction(
        problem.effective_log_pbias(physical), level)
    tbias_coarse = (differentiable_restriction(physical.tbias, level)
                    if physical.tbias is not None else None)
    S_obs_coarse = differentiable_restriction(
        torch.clamp(problem.domain.dem, min=0.0), level)
    # The model rate over the same interval the dhdt misfit uses (the
    # observation window in two-snapshot mode, the true final step in legacy
    # mode), so the VTI field is directly comparable to the observed raster.
    # Without a dhdt product, fall back to the final-step rate as a
    # near-steady-state diagnostic.
    dhdt_obs = problem.get_observation("dhdt")
    if dhdt_obs is not None:
        dhdt_coarse = dhdt_obs.model_rate(sim, "coarse")
    else:
        dhdt_coarse = (sim.H - sim.H_prev) / sim.final.dt_step
    update_diagnostic_fields(diag, sim.S_coarse, S_obs_coarse, bed_mean_coarse,
                             pbias_coarse, pbias_total_coarse, dhdt_coarse,
                             tbias_=tbias_coarse)
    vti_writer.append(problem.mg[level], time=i)
    vti_writer.write_pvd()

for level in range(config.max_level, config.min_level - 1, -1):
    problem.model.set_top_level(level)
    diag = make_diagnostic_fields(problem.mg[level])
    level_dir = f"{OUTPUT_PATH}/level_{level}/vti"
    vti_writer = make_loss_vti_writer(problem.mg[level], level_dir,
                                       config.vti_base_name, diag)

    for i in range(config.max_iters[level]):
        optimizer_sgd.zero_grad()
        optimizer_adam.zero_grad()

        # Periodically emit a per-time-step VTI series.
        time_writer = (make_time_vti_writer(problem.mg[level], level_dir)
                       if i % 10 == 0 else None)

        sim, physical = problem.simulate(
            level=level, params=params, time_writer=time_writer)
        loss_terms = problem.compute_loss(
            sim=sim, physical=physical, params=params,
            iteration=i, level=level, schedule=True)
        loss_terms.log(i)

        write_loss_vti(diag, vti_writer, sim, physical, level, i)
        
        loss_terms.J.backward()
        optimizer_sgd.step()
        optimizer_adam.step()

        # Drop this iteration's snapshots/graph now: `sim` pins fine-grid
        # tensors per observation epoch, and letting them survive into the
        # next simulate() call doubles the snapshot footprint.
        del sim, physical, loss_terms

        # Return freed blocks to the driver: torch and cupy each hoard their
        # own caching pool, and with the big transients interleaving between
        # the two allocators either can OOM while the other sits on free
        # memory. Costs milliseconds per iteration against a full solve.
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

    # Final evaluation (no backward) so the multigrid state matches the
    # converged parameters before we save it out.
    problem.simulate(level=level, params=params)

    mg_lvl = problem.mg[level]
    ds = xr.merge([
        mg_lvl.state.u.to_dataarray(),
        mg_lvl.state.v.to_dataarray(),
        mg_lvl.state.ud.to_dataarray(),
        mg_lvl.state.vd.to_dataarray(),
        mg_lvl.state.H.to_dataarray(),
        mg_lvl.geometry.bed.to_dataarray(),
        mg_lvl.sliding.beta.to_dataarray(),
        mg_lvl.forcing.smb.to_dataarray(),
    ])
    ds.to_netcdf(f"{OUTPUT_PATH}/level_{level}/inverse_soln.nc")
    save_whitened_params(params, f"{OUTPUT_PATH}/level_{level}/torch_vars.p",
                         bed_parametrization=problem.priors.bed_parametrization)
