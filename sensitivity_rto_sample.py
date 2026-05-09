"""
Mountain glacier forward simulation example, in
which we build a glacier system over the Bitterroot
Mountains in western Montana

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
import cupy as cp
import torch
import numpy as np
import pyproj

from torch.nn.functional import avg_pool2d, interpolate, grid_sample
from torch.utils.checkpoint import checkpoint

from glide.model import IceDynamics
from glide.io import VTIWriter
from glide.torch import GlideStep
from glide.field import Field,GridEntity

from glare.model import ImprovedTemperatureIndex
from glare.torch import GlareStep

from glide.data import load_wrangell_preprocessed

from ggapp.model import MaternPrior
from ggapp.torch import GGaPPWhiten, GGaPPMap, PSGD, PreconditionedAdam

from torchvision.transforms.functional import gaussian_blur

import xarray as xr
import geopandas as gpd

from pathlib import Path

# =============================================================================
# Load data
# =============================================================================

BASE_DIR = './domains/wrangell/'

INPUT_PATH = f'{BASE_DIR}/inverse_long/'
OUTPUT_PATH = f'{BASE_DIR}/inverse_long/sens/'

print("Loading geometry...")

N_LEVELS = 6

GRIDDED_FILENAME = f'{BASE_DIR}/model_inputs/GLIDE_inputs.nc'
FLIGHTLINE_FILENAME = f'{BASE_DIR}/model_inputs/flightlines.gpkg'
ANOMALY_FILENAME = f'{BASE_DIR}/model_inputs/temperature_anomaly.nc'
gridded_data = xr.open_dataset(GRIDDED_FILENAME)
temperature_anomaly = xr.open_dataset(ANOMALY_FILENAME)
flightlines = gpd.read_file(FLIGHTLINE_FILENAME)

crs = pyproj.CRS(gridded_data.spatial_ref.crs_wkt)

factor = 2**N_LEVELS
ny_0,nx_0 = gridded_data.sizes['y'],gridded_data.sizes['x']

ny_target = (ny_0 // factor) * factor
nx_target = (nx_0 // factor) * factor

# Center the subregion
y_start = (ny_0 - ny_target) // 2
x_start = (nx_0 - nx_target) // 2

gridded_data = gridded_data.isel(
        y=slice(y_start,y_start + ny_target),
        x=slice(x_start,x_start + nx_target)
        )

ny,nx = gridded_data.sizes['y'],gridded_data.sizes['x']
dx = (gridded_data.x[1]-gridded_data.x[0]).item()
x0 = gridded_data.x[0].item()
y0 = gridded_data.y[0].item()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
model = IceDynamics(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx,
        x0=x0,y0=y0,
        crs=crs)
mg = model.mg

### Initialize state
mg.state.H.set(0.1)
mg.state.H_prev.set(0.1)

### Initialize geometry
mg.geometry.bed.set(gridded_data.elevation)
mg.geometry.depth.set(np.maximum(-gridded_data.elevation,0))
mg.geometry.sigmoid_c.set(0.1)
mg.geometry.sigmoid_k.set(4.0)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = 1e-16 ** (-1.0 / 3.0) / (917 * 9.81)
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-5)
mg.rheology.n.set(3.0)

mg.sliding.beta.set(2.0)
mg.sliding.m.set(1./3.)
mg.sliding.water_drag.set(0.01)

mg.calving.calving_rate.set(1000.0)

smb_model = ImprovedTemperatureIndex(ny=ny,nx=nx,nt=12,
        dx=dx,dt=1./12,
        x0=x0,y0=y0,
        crs=crs)

# L2 Regularization - smb offset
sigma_log_rf = 0.1
sigma_log_mf = 0.1
mu_rf = 20.0
mu_mf = 2.0
mu_log_rf = np.log(mu_rf)
mu_log_mf = np.log(mu_mf)

smb_model.grid.insolation.insol_mean.set(gridded_data.monthly_solar_potential_mean)
smb_model.grid.insolation.insol_cos.set(gridded_data.monthly_solar_potential_cos)
smb_model.grid.insolation.insol_sin.set(gridded_data.monthly_solar_potential_sin)
smb_model.grid.temperature.t2m.set(gridded_data.monthly_t2m)
smb_model.grid.precipitation.precip.set(gridded_data.monthly_precip)
smb_model.grid.insolation.rf.set(mu_rf)
smb_model.grid.temperature.mf.set(mu_mf)
smb_model.forward()

### Initialize forcing
mg.forcing.smb.set(smb_model.grid.state.smb.data.mean(axis=0))

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=50, finest_steps=50,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=False)

model.adjoint_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=50, finest_steps=50,
        relative_tolerance=1e-3, absolute_tolerance=1e-6, # Note that adjoint var
        report_norms=False)                               # adjoint var is small 
                                                          # in magnitude

model.adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(1.0))

bed_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
bed_model.mg.parameters.sigma.set(500.0)
bed_model.mg.parameters.l.set(500.0)
bed_model.mg.parameters.nu.set(3)
bed_model.forward_solver.fas_options.report_norms.set(False)
bed_map = GGaPPMap.apply

mean_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
mean_model.mg.parameters.sigma.set(2000.0)
mean_model.mg.parameters.l.set(10000.0)
mean_model.mg.parameters.nu.set(1)
mean_model.forward_solver.fas_options.report_norms.set(False)
mean_map = GGaPPMap.apply

log_beta_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
log_beta_model.mg.parameters.sigma.set(3.0)
log_beta_model.mg.parameters.l.set(1000.0)
log_beta_model.mg.parameters.nu.set(1)
log_beta_model.forward_solver.fas_options.report_norms.set(False)
log_beta_map = GGaPPMap.apply

pbias_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
pbias_model.mg.parameters.sigma.set(0.1)
pbias_model.mg.parameters.l.set(10000.0)
pbias_model.mg.parameters.nu.set(3)
pbias_model.forward_solver.fas_options.report_norms.set(False)
pbias_map = GGaPPMap.apply


# Thin Pytorch wrapper of a single glide time step
glide_step = GlideStep.apply
glare_step = GlareStep.apply

log_beta = torch.tensor(cp.log(mg[0].sliding.beta.data),
        device='cuda',requires_grad=True)
H_prev = torch.tensor(mg[0].state.H_prev.data,
        device='cuda',requires_grad=False)
bed = torch.tensor(mg[0].geometry.bed.data,
        device='cuda',requires_grad=True)

bed_mean = torch.zeros(ny,nx, dtype=torch.float32,
        device='cuda',requires_grad=True)

t2m = torch.tensor(smb_model.grid.temperature.t2m.data,
        device='cuda',requires_grad=False)
precip = torch.tensor(smb_model.grid.precipitation.precip.data,
        device='cuda',requires_grad=False)

precipitation_bias = torch.zeros(ny,nx,
        dtype=torch.float32,
        requires_grad=True,device='cuda')

log_mf = torch.tensor(cp.log(smb_model.grid.temperature.mf.value),
        device='cuda',requires_grad=True)
log_rf = torch.tensor(cp.log(smb_model.grid.insolation.rf.value),
        device='cuda',requires_grad=True)
alpha_t2m = torch.tensor(2.2,
        device='cuda',requires_grad=False)

WARM_START_PATH = f'{INPUT_PATH}/level_2/torch_vars.p'
if WARM_START_PATH is not None:
    d = torch.load(WARM_START_PATH)
    log_beta = d['log_beta']
    bed = d['bed']
    bed_mean = d['bed_mean']
    precipitation_bias = d['precipitation_bias']
    log_mf = d['log_mf']
    log_rf = d['log_rf']

p = Path(f'{INPUT_PATH}/rto/')

def get_rto_samples(p):

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
    pbias_flat = torch.stack([p.ravel() for p in pbias_samples],axis=-1)
    log_beta_flat = torch.stack([p.ravel() for p in log_beta_samples],axis=-1)
    log_mf_flat = torch.stack(log_mfs).reshape(1,-1)
    log_rf_flat = torch.stack(log_rfs).reshape(1,-1)

    p_flat = torch.vstack((bed_flat,pbias_flat,log_beta_flat,log_mf_flat,log_rf_flat))
    p_mean = p_flat.mean(axis=1,keepdims=True)

    return p_flat, p_mean

p_flat, p_mean = get_rto_samples(p)

rgi_mask = torch.tensor(gridded_data.rgi_mask.values,device='cuda')
domain_mask = torch.tensor(gridded_data.domain_mask.values,device='cuda')
base_anomaly = temperature_anomaly.sel(time=2012).temp_anomaly.item()

sigma_s = 10.0
sigma_u = 10.0

u_obs = torch.tensor(gridded_data.vx.values,
        dtype=torch.float32,device='cuda').nan_to_num().masked_fill(~domain_mask,0.0)
v_obs = torch.tensor(gridded_data.vy.values,
        dtype=torch.float32,device='cuda').nan_to_num().masked_fill(~domain_mask,0.0)
S_obs = torch.tensor(gridded_data.elevation.values,
        dtype=torch.float32,device='cuda')
flightlines = torch.tensor(flightlines[['x','y','bed']].values,
        dtype=torch.float32,device='cuda')

xmin,xmax = gridded_data.x.min().item(),gridded_data.x.max().item()
ymin,ymax = gridded_data.y.min().item(),gridded_data.y.max().item()
col_normed =  (2.0*((flightlines[:,0] - xmin)/(xmax - xmin)) - 1)
row_normed = -(2.0*((flightlines[:,1] - ymin)/(ymax - ymin)) - 1)
bed_normed_coords = torch.stack([col_normed,row_normed],dim=-1)
bed_obs = flightlines[:,2]

def compute_smb(smb_model,t2m,t_anomaly,base_anomaly,precip_,mf,rf,domain_mask):
    smb = glare_step(smb_model,t2m + (t_anomaly - base_anomaly),precip_,mf,rf).mean(axis=0)
    smb_ = smb.masked_fill(~domain_mask,-10)
    return smb_

def differentiable_restriction(field,n_times,method='avg'):
    if method=='avg':
        fn = avg_pool2d
    if method=='max':
        fn = max_pool2d
    for _ in range(n_times):
        field = fn(field[None,:,:],(2,2))[0]
    return field

n_samples = p_flat.shape[1]
deltas = []
for s in range(n_samples):
    print(s)
    DT = 20.0
    level = 2

    model.set_top_level(level)
    
    delta = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    srf = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    bed_mean_field = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    p_bias_field = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])
    
    mask = torch.tensor(rgi_mask*domain_mask,
            dtype=torch.float32,device='cuda')

    bed.data[:,:] = p_flat[:ny*nx,s].reshape(ny,nx)
    precipitation_bias.data[:,:] = p_flat[ny*nx:2*ny*nx,s].reshape(ny,nx)
    log_beta.data[:,:] = p_flat[2*ny*nx:3*ny*nx,s].reshape(ny,nx)
    log_mf.data.fill_(p_flat[3*ny*nx,s])
    log_rf.data.fill_(p_flat[3*ny*nx+1,s])

    def evaluate_delta_v(i,compute_gradient=True,write_vti=True):

        depth = torch.maximum(-bed,torch.zeros_like(bed)).detach()
        mg.geometry.depth.set(0.1*cp.asarray(depth) + 0.9*mg[0].geometry.depth.data)

        mg.state.u.set(0.0,start_level=level)
        mg.state.v.set(0.0,start_level=level)
        mg.state.H.set(0.1,start_level=level)
        mg.state.H_prev.set(0.1,start_level=level)
        mg.state.mask.set(0.0,start_level=level)

        mg.adjoint.lambda_u.set(0.0,start_level=level)
        mg.adjoint.lambda_v.set(0.0,start_level=level)
        mg.adjoint.lambda_H.set(0.0,start_level=level)

        precip_ = precip + precipitation_bias 
        mf = torch.exp(log_mf)
        rf = torch.exp(log_rf)

        bed_ = differentiable_restriction(bed,level)
        bed_mean_ = differentiable_restriction(bed_mean,level)
        precipitation_bias_ = differentiable_restriction(precipitation_bias,level)
        H_prev_ = differentiable_restriction(H_prev,level)
        log_beta_ = differentiable_restriction(log_beta,level)
        S_obs_ = differentiable_restriction(S_obs,level)

        beta_ = torch.exp(log_beta_)

        t = cp.float32(2112-1100)
        t_end = cp.float32(2112)
        dt = cp.float32(DT)

        time_writer = VTIWriter(f'{OUTPUT_PATH}/level_{level}/vti', base='time', dx=mg[level].dx,
                dynamic_fields={'thk':mg[level].state.H,
                                'U':[mg[level].state.u,mg[level].state.v],
                               'smb':mg[level].forcing.smb}
            )
        

        while t < t_end:
            t_anomaly_0 = alpha_t2m*temperature_anomaly.sel(time=int(min(t,2012))).temp_anomaly.item()
            t_anomaly_1 = alpha_t2m*temperature_anomaly.sel(time=int(min(t+dt,2012))).temp_anomaly.item()
            t_anomaly = 0.5*(t_anomaly_0 + t_anomaly_1)
            
            smb = checkpoint(compute_smb,smb_model,
                    t2m,t_anomaly,base_anomaly,precip_,
                    mf,rf,domain_mask,use_reentrant=False)

            smb_ = differentiable_restriction(smb,level)

            u,v,H = glide_step(t,dt,model,level,H_prev_,bed_,beta_,smb_)

            t += dt
            H_prev_ = H

            if t==2012:
                V_0 = torch.sum(H * (dx*2**level)**2)

            time_writer.append(mg[level],time=t)
            time_writer.write_pvd()
        
        V_1 = torch.sum(H * (dx*2**level)**2)

        Delta_V = V_1 - V_0

        #Delta_V.backward()
        return Delta_V

    Delta_V = evaluate_delta_v(0,write_vti=False)
    deltas.append(Delta_V.detach())

Q = torch.stack(deltas).cpu()
Q_ = Q - Q.mean()

X_ = p_flat - p_flat.mean(axis=1,keepdims=True)

c = (X_ @ Q_)/(n_samples-1)
v = (X_ * X_).sum(axis=1)/(n_samples-1)

c_bed = c[:nx*ny].reshape(ny,nx)
v_bed = v[:nx*ny].reshape(ny,nx)

c_pbias = c[nx*ny:2*nx*ny].reshape(ny,nx)
v_pbias = v[nx*ny:2*nx*ny].reshape(ny,nx)

delta_V_bed = c_bed**2 / (v_bed + 1**2)
delta_V_pbias = c_pbias**2 / (v_pbias + 0.0001**2)




