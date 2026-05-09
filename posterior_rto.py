from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np

BASE_DIR = './domains/wrangell/'
INPUT_PATH = f'{BASE_DIR}/inverse_long/'

p = Path(f'{INPUT_PATH}/rto/')


beds = []
pbiases = []
log_betas = []
log_mfs = []
log_rfs = []


for d in p.iterdir():
    try:
        data = torch.load(f'{d}/level_2/torch_vars.p')
        log_beta = data['log_beta'].cpu().detach()
        bed = data['bed'].cpu().detach()
        bed_mean = data['bed_mean']
        precipitation_bias = data['precipitation_bias'].cpu().detach()
        log_mf = data['log_mf'].cpu().detach()
        log_rf = data['log_rf'].cpu().detach()

        beds.append(bed)
        pbiases.append(precipitation_bias)
        log_betas.append(log_beta)
        log_mfs.append(log_mf)
        log_rfs.append(log_rf)

    except FileNotFoundError:
        pass

ny,nx = beds[0].shape

bed_samples = torch.stack(beds,axis=0)
pbias_samples = torch.stack(pbiases,axis=0)
log_beta_samples = torch.stack(log_betas,axis=0)
log_mf_samples = torch.stack(log_mfs,axis=0)
log_rf_samples = torch.stack(log_rfs,axis=0)

bed_flat = torch.stack([b.ravel() for b in bed_samples],axis=-1)
bed_flat = (bed_flat - bed_flat.mean(axis=1,keepdims=True))/np.sqrt(bed_flat.shape[1] - 1)
u,s,_ = torch.linalg.svd(bed_flat,full_matrices=False)
L_bed = u * s

pbias_flat = torch.stack([p.ravel() for p in pbias_samples],axis=-1)
pbias_flat = (pbias_flat - pbias_flat.mean(axis=1,keepdims=True))/np.sqrt(pbias_flat.shape[1] - 1)
u,s,_ = torch.linalg.svd(pbias_flat,full_matrices=False)
L_pbias = u * s





