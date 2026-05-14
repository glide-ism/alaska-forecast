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

import xarray as xr
import geopandas as gpd

# =============================================================================
# Load data
# =============================================================================

BASE_DIR = './domains/wrangell/'

OUTPUT_PATH = f'{BASE_DIR}/inverse_white_v2/'

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
bed_model.mg.parameters.sigma.set(250.0)
bed_model.mg.parameters.l.set(1000.0)
bed_model.mg.parameters.nu.set(1)
bed_model.forward_solver.fas_options.report_norms.set(False)
bed_map = GGaPPMap.apply

mean_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
mean_model.mg.parameters.sigma.set(1000.0)
mean_model.mg.parameters.l.set(10000.0)
mean_model.mg.parameters.nu.set(3)
mean_model.forward_solver.fas_options.report_norms.set(False)
mean_map = GGaPPMap.apply

log_beta_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
log_beta_model.mg.parameters.sigma.set(1.0)
log_beta_model.mg.parameters.l.set(1000.0)
log_beta_model.mg.parameters.nu.set(1)
log_beta_model.forward_solver.fas_options.report_norms.set(False)
log_beta_map = GGaPPMap.apply

pbias_model = MaternPrior(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx)
pbias_model.mg.parameters.sigma.set(0.3)
pbias_model.mg.parameters.l.set(10000.0)
pbias_model.mg.parameters.nu.set(3)
pbias_model.forward_solver.fas_options.report_norms.set(False)
pbias_map = GGaPPMap.apply


# Thin Pytorch wrapper of a single glide time step
glide_step = GlideStep.apply
glare_step = GlareStep.apply

log_beta = torch.tensor(cp.log(mg[0].sliding.beta.data),
        device='cuda',requires_grad=False)
z_log_beta = GGaPPWhiten.apply(log_beta_model,log_beta)
z_log_beta.requires_grad_()

H_prev = torch.tensor(mg[0].state.H_prev.data,
        device='cuda',requires_grad=False)

bed = torch.tensor(mg[0].geometry.bed.data,
        device='cuda',requires_grad=False)

bed_mean = torch.zeros(ny,nx, dtype=torch.float32,
        device='cuda',requires_grad=False)

z_bed = GGaPPWhiten.apply(bed_model,bed - bed_mean)
z_bed_mean = GGaPPWhiten.apply(mean_model, bed_mean)

z_bed.requires_grad_()
z_bed_mean.requires_grad_()

t2m = torch.tensor(smb_model.grid.temperature.t2m.data,
        device='cuda',requires_grad=False)
precip = torch.tensor(smb_model.grid.precipitation.precip.data,
        device='cuda',requires_grad=False)

z_precipitation_bias = torch.zeros(ny,nx,
        dtype=torch.float32,
        requires_grad=True,device='cuda')

log_mf = torch.tensor(cp.log(smb_model.grid.temperature.mf.value),
        device='cuda',requires_grad=False)
log_rf = torch.tensor(cp.log(smb_model.grid.insolation.rf.value),
        device='cuda',requires_grad=False)

z_log_mf = (log_mf - mu_log_mf) / sigma_log_mf
z_log_rf = (log_rf - mu_log_rf) / sigma_log_rf

z_log_mf.requires_grad_()
z_log_rf.requires_grad_()

alpha_t2m = torch.tensor(2.2,
        device='cuda',requires_grad=False)

WARM_START_PATH = None#f'{OUTPUT_PATH}/level_0/torch_vars.p'
if WARM_START_PATH is not None:
    d = torch.load(WARM_START_PATH)
    log_beta = d['log_beta']
    bed = d['bed']
    bed_mean = d['bed_mean']
    precipitation_bias = d['precipitation_bias']
    log_mf = d['log_mf']
    log_rf = d['log_rf']

rgi_mask = torch.tensor(gridded_data.rgi_mask.values,device='cuda')
domain_mask = torch.tensor(gridded_data.domain_mask.values,device='cuda')
base_anomaly = temperature_anomaly.sel(time=2012).temp_anomaly.item()

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

def differentiable_prolongation(field,n_times,grid_entity='cell',method='bilinear'):
    for _ in range(n_times):
        if grid_entity=='cell':
            ny_fine,nx_fine = 2*field.shape[0],2*field.shape[1]
        elif grid_entity=='vfacet':
            ny_fine,nx_fine = 2*field.shape[0],2*(field.shape[1] - 1) + 1
        elif grid_entity=='hfacet':
            ny_fine,nx_fine = 2*(field.shape[0] - 1) + 1,2*field.shape[1]
        field = interpolate(field[None,None,:,:],(ny_fine,nx_fine),mode='bilinear').squeeze()
    return field

optimizer_sgd = torch.optim.SGD([
    {'params':z_bed,
     'lr':0.2},
    {'params':z_bed_mean,
     'lr':1.0},
    {'params':z_log_beta,
     'lr':0.75}
    ],
    momentum=0.5
  )

optimizer_adam = torch.optim.Adam([
    {'params':z_precipitation_bias,
     'lr':0.0003},
    {'params':z_log_mf,
     'lr':0.01},
    {'params':z_log_rf,
     'lr':0.01}
    ],
    betas=(0.5,0.99)
  )

MAX_LEVEL = 2
MIN_LEVEL = 0
DT = 20.0
max_iters = [20,50,500]
for level in range(MAX_LEVEL,MIN_LEVEL-1,-1):
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
    
    
    vti_writer = VTIWriter(f'{OUTPUT_PATH}/level_{level}/vti', base='wrangell', dx=mg[level].dx,
            dynamic_fields={'bed':mg[level].geometry.bed,
                            'beta':mg[level].sliding.beta,
                            'thk':mg[level].state.H,
                            'U':[mg[level].state.u,mg[level].state.v],
                            'srf':srf,
                            'delta':delta,
                            'p_bias':p_bias_field,
                            'bed_mean':bed_mean_field,
                            'smb':mg[level].forcing.smb}
        )

    vti_writer.initialize(mg[level])
    
    def evaluate_loss(i,compute_gradient=True,write_vti=True):
        optimizer_sgd.zero_grad()
        optimizer_adam.zero_grad()

        bed_mean = GGaPPMap.apply(mean_model,z_bed_mean)
        bed = GGaPPMap.apply(bed_model,z_bed)# + bed_mean

        log_beta = GGaPPMap.apply(log_beta_model,z_log_beta)
        precipitation_bias = GGaPPMap.apply(pbias_model,z_precipitation_bias)

        log_mf = mu_log_mf + sigma_log_mf * z_log_mf
        log_rf = mu_log_rf + sigma_log_rf * z_log_rf

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

        t = cp.float32(2012-1000)
        t_end = cp.float32(2012)
        dt = cp.float32(DT)

        if i % 10 == 0:
            time_writer = VTIWriter(f'{OUTPUT_PATH}/level_{level}/vti', base='time', dx=mg[level].dx,
                dynamic_fields={'thk':mg[level].state.H,
                                'U':[mg[level].state.u,mg[level].state.v],
                               'smb':mg[level].forcing.smb}
            )
        

        while t < t_end:
            t_anomaly_0 = alpha_t2m*temperature_anomaly.sel(time=int(t)).temp_anomaly.item()
            t_anomaly_1 = alpha_t2m*temperature_anomaly.sel(time=int(t+dt)).temp_anomaly.item()
            t_anomaly = 0.5*(t_anomaly_0 + t_anomaly_1)
            
            smb = checkpoint(compute_smb,smb_model,
                    t2m,t_anomaly,base_anomaly,precip_,
                    mf,rf,domain_mask,use_reentrant=False)

            smb_ = differentiable_restriction(smb,level)

            u,v,H = glide_step(t,dt,model,level,H_prev_,bed_,beta_,smb_)

            t += dt
            H_prev_ = H
            
            if i % 20 == 0:
                time_writer.append(mg[level],time=t)
                time_writer.write_pvd()

        # Surface elevation loss
        S = bed_ + H
        S_ = S

        u = differentiable_prolongation(u,level,grid_entity='vfacet')
        v = differentiable_prolongation(v,level,grid_entity='hfacet')
        H = differentiable_prolongation(H,level,grid_entity='cell')
        S = differentiable_prolongation(S,level,grid_entity='cell')

        loss_scale = 1e-4

        lambda_s = 2e-5
        sigma_s = 10.0
        nu_s = 1.0
        r_s = (S-S_obs)/sigma_s
        J_srf = loss_scale * lambda_s * dx**2 * nu_s**2 * (torch.sqrt(1 + (r_s / nu_s)**2) - 1).sum()

        u_pred = (u[:,1:] + u[:,:-1])/2.
        v_pred = (v[1:,:] + v[:-1,:])/2.     

        lambda_u = 2e-5
        sigma_u = 10.0
        nu_u = 1.0
        r_u2 = ((u_pred - u_obs)**2 + (v_pred - v_obs)**2)/sigma_u**2

        J_vel = loss_scale * lambda_u * dx**2 * nu_u**2 * (torch.sqrt(1 + r_u2/nu_u**2) - 1).sum()
        

        # Extent loss
        lambda_e = 2e-4
        s_H = 10.0

        p_extent = (2./(1+torch.exp(-H/s_H))-1).clip(min=0.001,max=0.999)
        #J_extent = -loss_scale * lambda_e * dx**2 * (mask*torch.log(p_extent) + (1-mask)*torch.log(1-p_extent)).sum()
        J_extent = -loss_scale * lambda_e * dx**2 * (mask*torch.log(p_extent)).sum()

        bed_pred = grid_sample(bed[None,None,:,:],bed_normed_coords[None,None,:,:],mode='bilinear').squeeze()
        lambda_bed = 2e-5
        sigma_bed = 10.0
        nu_bed = 1.0
        #r_bed = (bed_pred - bed_obs)/sigma_bed 

        #J_bed = loss_scale * lambda_bed * dx**2 * nu_bed**2 * (torch.sqrt(1 + (r_bed / nu_bed)**2) - 1).sum()

        r_bed_flightlines = (bed_pred - bed_obs)/sigma_bed 
        r_bed_grid = (bed-S_obs)/sigma_s*(1-mask)
        J_bed = loss_scale * lambda_bed * dx**2 * nu_bed**2 * ((torch.sqrt(1 + (r_bed_flightlines / nu_bed)**2) - 1).sum() + (torch.sqrt(1 + (r_bed_grid / nu_bed)**2) - 1).sum())

        # Tikhonov Regularization - bed
        z_bed_ = GGaPPWhiten.apply(bed_model,bed - bed_mean)
        J_prior_bed = loss_scale*(z_bed_**2).sum()
        J_prior_bed_mean = loss_scale*(z_bed_mean**2).sum()
        J_prior_beta = loss_scale*(z_log_beta**2).sum()
        J_prior_pbias = loss_scale*(z_precipitation_bias**2).sum()
        J_prior_smb = ((log_rf - mu_log_rf)/sigma_log_rf)**2 + ((log_mf - mu_log_mf)/sigma_log_mf)**2

        J_data = (J_srf + J_vel + J_extent + J_bed)
        J_prior = (J_prior_bed + J_prior_bed_mean + J_prior_beta + J_prior_pbias + J_prior_smb)


        J = J_data + J_prior

        if write_vti:
            delta.data[:,:] = cp.asarray(S_.detach() - S_obs_)
            srf.data[:,:] = cp.asarray(S_.detach())
            bed_mean_field.data[:,:] = cp.asarray(bed_mean_.detach())
            p_bias_field.data[:,:] = cp.asarray(precipitation_bias_.detach())
            vti_writer.append(mg[level],time=i)
            vti_writer.write_pvd()
        

        print(f"="*60)
        print(f"Iteration: {i}, Total Loss: {J.item():.2f}, Data Loss: {J_data.item():.2f}, Prior Loss: {J_prior.item():.2f}")
        print(f"Srf Loss: {J_srf.item():.2f}, U Loss: {J_vel.item():.2f}, Ext Loss: {J_extent.item():.2f}, Bed Loss: {J_bed.item():.2f}")
        print(f"Bed Prior: {J_prior_bed:.2f}, Bed Mean Prior: {J_prior_bed_mean:.2f}, Beta Prior: {J_prior_beta:.2f}, Pbias Prior: {J_prior_pbias:.2f}")
        print(f"="*60)
        if compute_gradient:
            J.backward()
        return J


    for i in range(0,max_iters[level]):
        evaluate_loss(i)
        optimizer_sgd.step()
        optimizer_adam.step()
    evaluate_loss(0,write_vti=False,compute_gradient=False)

    u_xr = mg[level].state.u.to_dataarray()
    v_xr = mg[level].state.v.to_dataarray()
    H_xr = mg[level].state.H.to_dataarray()
    bed_xr = mg[level].geometry.bed.to_dataarray()
    beta_xr = mg[level].sliding.beta.to_dataarray()
    smb_xr = mg[level].forcing.smb.to_dataarray()
    ds = xr.merge([u_xr,v_xr,H_xr,bed_xr,beta_xr,smb_xr])
    ds.to_netcdf(f'{OUTPUT_PATH}/level_{level}/inverse_soln.nc')
    torch.save({'log_beta':z_log_beta,'bed':z_bed,'bed_mean':z_bed_mean,'precipitation_bias':z_precipitation_bias,'log_rf':z_log_rf,'log_mf':z_log_mf},f'{OUTPUT_PATH}/level_{level}/torch_vars.p')
