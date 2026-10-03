"""
Juneau domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import (
    BedConditioningConfig, FingerprintNuisance, GlacierConfig, MaternNoise,
    PriorHyperparams, Schedule,
)
from glacier_inverse.observations import (
    BedSpec, BedSlopeSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)
import numpy as np

_HERE = Path(__file__).parent


CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="juneau",
    results_subdir="inverse_tempered_restored_bgrad",
    smb_model = "enthalpy",
    anomaly_integration="mean_anomaly",
    stress_scheme='molho',
    grad_start_time=1712,

    max_level=2,
    max_iters=(50,50,500),
    
    init_from_observed_geometry = True,
    use_avalanche_model = True,
    avalanche_hoisted=True,

    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    log_beta_prior = PriorHyperparams(sigma=1./3.,    l=1000.0, nu=1),
    h_atm_prior=PriorHyperparams(sigma=0.2, l=80000.0, nu=1),
    cloud_prior=PriorHyperparams(sigma=0.25, l=80000.0, nu=1),
    tbias_prior=PriorHyperparams(sigma=1.0,     l=10000.0, nu=1),

    influence_cap={"z_log_H_atm": 0.3, "z_logit_cloud": 0.3,
                   "z_tbias": 0.3},
    influence_transfer='log',
    
    observations=(
        SurfaceSpec(noise=MaternNoise(sigma=12.0, l=1000.0, nu=0.5, nugget=10.0),
                    weight=1.0, nu=3),
        VelocitySpec(noise=MaternNoise(sigma=12.0, l=3000.0, nugget=10.0), weight=1.0,
                     surge_biased=False, nu=3, alpha_nonsurge=20),
        ExtentSpec(weight=1.0, s_H=10.0, sigma_p=0.3,
                   logit_error=MaternNoise(sigma=0.3, l=1000.0),
                   nuisance_inner_steps=2, eps_max=1.0),
        BedSpec(weight=0.0e-6),
        SnowlineSpec(weight=1.0,
                     s_smb=0.5, sigma_p=0.3,
                     logit_error=MaternNoise(sigma=0.3, l=1000.0),
                     nuisance_inner_steps=2, eps_max=1.0),
        DhdtSpec(noise=MaternNoise(sigma=0.5, l=3000.0, nu=0.5, nugget=0.5),
                 weight=1.0),
        BedSlopeSpec(weight=1e-5,s_scale=5.0),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
                         enabled=True,
                         sigma_picks=100,
                         sigma_dem=100,
                         pcg_rtol=1e-3,
                         pcg_rtol_adjoint=1e-2),

    lr_z_bed=0.0125,
    lr_z_log_beta=0.05*9*9,
    lr_z_pbias=Schedule(final=0.05, ramp=lambda i,level:0.0 if (i<0 and level==2) else 0.05),
    lr_z_tbias=Schedule(final=10.0,ramp=lambda i,level:0.0 if (i<0 and level==2) else 10.0),
    lr_z_log_H_atm=Schedule(final=10.0,ramp=lambda i,level:0.0 if (i<0 and level==2) else 10.0),
    lr_z_logit_cloud=Schedule(final=10.0,ramp=lambda i,level:0.0 if (i<0 and level==2) else 10.0),
)


