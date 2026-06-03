"""
Deterministic MAP solve in whitened coordinates.

Thin driver over glacier_inverse. All physical hyperparameters live in
domains/<name>/config.py — per-task knobs (output path, max iters,
learning rates) stay here.
"""
import torch
import xarray as xr

from glacier_inverse import GlacierProblem, load_config
from glacier_inverse.forward import differentiable_restriction
from glacier_inverse.io import (
    load_whitened_params_into, make_diagnostic_fields, make_loss_vti_writer,
    make_time_vti_writer, save_whitened_params, update_diagnostic_fields,
)

# Available domains: domains/{chugach,delta,denali,juneau,st_elias,wrangell}
DOMAIN = "domains/juneau"
config = load_config(DOMAIN)

OUTPUT_PATH = config.output_dir
WARM_START_PATH = None  # e.g. f"{OUTPUT_PATH}/level_0/torch_vars.p"

problem = GlacierProblem(config)
params = problem.params

if WARM_START_PATH is not None:
    load_whitened_params_into(params, WARM_START_PATH)

# Dump the observational products once, alongside the finest-level diagnostics.
problem.write_observations(f"{OUTPUT_PATH}/level_{config.min_level}/vti")

optimizer_sgd = torch.optim.SGD([
    {"params": params.z_bed,      "lr": config.lr_z_bed},
    {"params": params.z_bed_mean, "lr": config.lr_z_bed_mean},
    {"params": params.z_log_beta, "lr": config.lr_z_log_beta},
], momentum=0.5)

optimizer_adam = torch.optim.Adam([
    {"params": params.z_pbias,  "lr": config.lr_z_pbias},
    {"params": params.z_log_mf, "lr": config.lr_z_log_mf},
    {"params": params.z_log_rf, "lr": config.lr_z_log_rf},
], betas=(0.5, 0.99))

def write_loss_vti(diag, vti_writer, sim, physical, level, i):
    bed_mean_coarse = differentiable_restriction(physical.bed_mean, level)
    pbias_coarse = differentiable_restriction(physical.pbias, level)
    S_obs_coarse = differentiable_restriction(problem.observations.S_obs, level)
    update_diagnostic_fields(diag, sim.S_coarse, S_obs_coarse, bed_mean_coarse, pbias_coarse)
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
                       if i % 20 == 0 else None)

        sim, physical = problem.simulate(
            level=level, params=params, time_writer=time_writer)
        loss_terms = problem.compute_loss(
            sim=sim, physical=physical, params=params)
        loss_terms.log(i)

        write_loss_vti(diag, vti_writer, sim, physical, level, i)
        
        loss_terms.J.backward()
        optimizer_sgd.step()
        optimizer_adam.step()

    # Final evaluation (no backward) so the multigrid state matches the
    # converged parameters before we save it out.
    problem.simulate(level=level, params=params)

    mg_lvl = problem.mg[level]
    ds = xr.merge([
        mg_lvl.state.u.to_dataarray(),
        mg_lvl.state.v.to_dataarray(),
        mg_lvl.state.H.to_dataarray(),
        mg_lvl.geometry.bed.to_dataarray(),
        mg_lvl.sliding.beta.to_dataarray(),
        mg_lvl.forcing.smb.to_dataarray(),
    ])
    ds.to_netcdf(f"{OUTPUT_PATH}/level_{level}/inverse_soln.nc")
    save_whitened_params(params, f"{OUTPUT_PATH}/level_{level}/torch_vars.p")
