"""
Deterministic MAP solve in whitened coordinates.

Thin driver over glacier_inverse. All physical hyperparameters live in
configs/wrangell.py — per-task knobs (output path, max iters, learning rates)
stay here.
"""
import torch
import xarray as xr

#from configs.denali import DENALI as config
#from configs.juneau import JUNEAU as config
#from configs.delta import DELTA as config
#from configs.wrangell import WRANGELL as config
from configs.chugach import CHUGACH as config
from glacier_inverse import GlacierProblem
from glacier_inverse.forward import differentiable_restriction
from glacier_inverse.io import (
    load_whitened_params_into, make_diagnostic_fields, make_loss_vti_writer,
    make_time_vti_writer, save_whitened_params, update_diagnostic_fields,
)


OUTPUT_PATH = f"{config.base_dir}/inverse_brier_test/"
WARM_START_PATH = None  # e.g. f"{OUTPUT_PATH}/level_0/torch_vars.p"

MAX_LEVEL = 2
MIN_LEVEL = 0
MAX_ITERS = [20, 50, 500]

problem = GlacierProblem(config)
params = problem.params

if WARM_START_PATH is not None:
    load_whitened_params_into(params, WARM_START_PATH)

# Dump the observational products once, alongside the finest-level diagnostics.
problem.write_observations(f"{OUTPUT_PATH}/level_{MIN_LEVEL}/vti")

optimizer_sgd = torch.optim.SGD([
    {"params": params.z_bed, "lr": 0.0325},
    #{"params": params.z_bed, "lr": 0.5},
    {"params": params.z_bed_mean, "lr": 0.5},
    {"params": params.z_log_beta, "lr": 0.25},
], momentum=0.5)

# Good params!
#optimizer_sgd = torch.optim.SGD([
#    {"params": params.z_bed, "lr": 0.5},
#    {"params": params.z_bed_mean, "lr": 0.5},
#    {"params": params.z_log_beta, "lr": 0.25},
#], momentum=0.5)

optimizer_adam = torch.optim.Adam([
    {"params": params.z_pbias, "lr": 0.001},
    {"params": params.z_log_mf, "lr": 0.01},
    {"params": params.z_log_rf, "lr": 0.01},
], betas=(0.5, 0.99))

def write_loss_vti(diag, vti_writer, sim, physical, level, i):
    bed_mean_coarse = differentiable_restriction(physical.bed_mean, level)
    pbias_coarse = differentiable_restriction(physical.pbias, level)
    S_obs_coarse = differentiable_restriction(problem.observations.S_obs, level)
    update_diagnostic_fields(diag, sim.S_coarse, S_obs_coarse, bed_mean_coarse, pbias_coarse)
    vti_writer.append(problem.mg[level], time=i)
    vti_writer.write_pvd()


for level in range(MAX_LEVEL, MIN_LEVEL - 1, -1):
    problem.model.set_top_level(level)
    diag = make_diagnostic_fields(problem.mg[level])
    level_dir = f"{OUTPUT_PATH}/level_{level}/vti"
    vti_writer = make_loss_vti_writer(problem.mg[level], level_dir,
                                       config.vti_base_name, diag)

    for i in range(MAX_ITERS[level]):
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
