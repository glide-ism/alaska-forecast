from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from ggapp.model import MaternPrior
from ggapp.torch import GGaPPWhiten, GGaPPMap

BASE_DIR = './domains/wrangell/'
INPUT_PATH = f'{BASE_DIR}/inverse_white_v2/'

ny = 1344
nx = 1856
dx = 90.
N_LEVELS=6

bed_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
bed_model.mg.parameters.sigma.set(250.0)
bed_model.mg.parameters.l.set(1000.0)
bed_model.mg.parameters.nu.set(1)
bed_model.forward_solver.fas_options.report_norms.set(False)

mean_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
mean_model.mg.parameters.sigma.set(1000.0)
mean_model.mg.parameters.l.set(10000.0)
mean_model.mg.parameters.nu.set(3)
mean_model.forward_solver.fas_options.report_norms.set(False)

log_beta_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
log_beta_model.mg.parameters.sigma.set(1.0)
log_beta_model.mg.parameters.l.set(1000.0)
log_beta_model.mg.parameters.nu.set(1)
log_beta_model.forward_solver.fas_options.report_norms.set(False)

pbias_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
pbias_model.mg.parameters.sigma.set(0.3)
pbias_model.mg.parameters.l.set(10000.0)
pbias_model.mg.parameters.nu.set(3)
pbias_model.forward_solver.fas_options.report_norms.set(False)

sigma_log_rf = 0.1
sigma_log_mf = 0.1
mu_rf = 20.0
mu_mf = 2.0
mu_log_rf = np.log(mu_rf)
mu_log_mf = np.log(mu_mf)

p = Path(f'{INPUT_PATH}/rto/')


beds = []
bed_means = []
pbiases = []
log_betas = []
log_mfs = []
log_rfs = []

for d in p.iterdir():
    try:
        data = torch.load(f'{d}/level_2/torch_vars.p')
        log_beta = GGaPPMap.apply(log_beta_model,data['log_beta']).cpu().detach()
        bed = GGaPPMap.apply(bed_model,data['bed']).cpu().detach()
        precipitation_bias = GGaPPMap.apply(pbias_model,data['precipitation_bias']).cpu().detach()
        log_mf = data['log_mf'].cpu().detach() * sigma_log_mf + mu_log_mf
        log_rf = data['log_rf'].cpu().detach() * sigma_log_rf + mu_log_rf

        beds.append(bed)
        pbiases.append(precipitation_bias)
        log_betas.append(log_beta)
        log_mfs.append(log_mf)
        log_rfs.append(log_rf)

    except FileNotFoundError:
        print(d)

ny,nx = beds[0].shape

max_samples = 100000
bed_samples = torch.stack(beds[:max_samples],axis=0)
pbias_samples = torch.stack(pbiases[:max_samples],axis=0)
log_beta_samples = torch.stack(log_betas[:max_samples],axis=0)
log_mf_samples = torch.stack(log_mfs[:max_samples],axis=0)
log_rf_samples = torch.stack(log_rfs[:max_samples],axis=0)

bed_flat = torch.stack([b.ravel() for b in bed_samples],axis=-1)
bed_flat = (bed_flat - bed_flat.mean(axis=1,keepdims=True))/np.sqrt(bed_flat.shape[1] - 1)
u_bed,s_bed,_ = torch.linalg.svd(bed_flat,full_matrices=False)
L_bed = u_bed * s_bed

pbias_flat = torch.stack([p.ravel() for p in pbias_samples],axis=-1)
pbias_flat = (pbias_flat - pbias_flat.mean(axis=1,keepdims=True))/np.sqrt(pbias_flat.shape[1] - 1)
u_pbias,s_pbias,_ = torch.linalg.svd(pbias_flat,full_matrices=False)
L_pbias = u_pbias * s_pbias





