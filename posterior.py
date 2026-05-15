"""
Empirical posterior covariance from RTO samples.

Loads every sample under {INPUT_PATH}/rto/, maps each whitened parameter
back to physical space using the shared prior models, and computes SVD-based
covariance factorizations for bed and precipitation bias.
"""
from pathlib import Path

import numpy as np
import torch

from ggapp.torch import GGaPPMap

#from configs.wrangell import WRANGELL
from configs.delta import DELTA as config
from glacier_inverse.priors import GlacierPriors, domain_shape


INPUT_PATH = f"{config.base_dir}/inverse_refactor/"
MAX_SAMPLES = 100000

ny, nx, dx = domain_shape(config)
priors = GlacierPriors(config, ny, nx, dx)

beds = []
pbiases = []
log_betas = []
log_mfs = []
log_rfs = []

for d in Path(f"{INPUT_PATH}/rto_lr_decay/").iterdir():
    try:
        data = torch.load(f"{d}/level_2/torch_vars.p")
        beds.append(GGaPPMap.apply(priors.bed_model, data["bed"]).cpu().detach())
        pbiases.append(GGaPPMap.apply(priors.pbias_model, data["precipitation_bias"]).cpu().detach())
        log_betas.append(GGaPPMap.apply(priors.log_beta_model, data["log_beta"]).cpu().detach())
        log_mfs.append(data["log_mf"].cpu().detach() * priors.sigma_log_mf + priors.mu_log_mf)
        log_rfs.append(data["log_rf"].cpu().detach() * priors.sigma_log_rf + priors.mu_log_rf)
    except FileNotFoundError:
        print(d)

bed_samples      = torch.stack(beds[:MAX_SAMPLES],      axis=0)
pbias_samples    = torch.stack(pbiases[:MAX_SAMPLES],   axis=0)
log_beta_samples = torch.stack(log_betas[:MAX_SAMPLES], axis=0)
log_mf_samples   = torch.stack(log_mfs[:MAX_SAMPLES],   axis=0)
log_rf_samples   = torch.stack(log_rfs[:MAX_SAMPLES],   axis=0)


def _centered_factor(samples):
    flat = torch.stack([s.ravel() for s in samples], axis=-1)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / np.sqrt(flat.shape[1] - 1)
    u, s, _ = torch.linalg.svd(flat, full_matrices=False)
    return u*s, s


L_bed, s_bed = _centered_factor(bed_samples)
L_pbias, s_pbias = _centered_factor(pbias_samples)
