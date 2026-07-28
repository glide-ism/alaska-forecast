"""
Denali domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams, Schedule
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="denali",
    results_subdir="inverse_enthalpy",
    smb_model = "enthalpy",
    anomaly_integration="mean_anomaly",
    observations=(
        SurfaceSpec(weight=2.0e-6),
        VelocitySpec(weight=2.0e-6, surge_biased=True),
        ExtentSpec(weight=2e-5, s_H=10.0),
        BedSpec(weight=2.0e-6),
        SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5),
                     s_smb=0.5),
        DhdtSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5)),
    ),
    loss_scale=1e-3,
    ssa_damping=1.0,
    sliding_m=1./4.,
    beta_init=3.0,
    init_from_observed_geometry = True,
    use_avalanche_model = True,
    debris_factor=0.5,
    sigma_log_rf = 0.1,
    sigma_log_mf = 0.2,
    lr_z_bed = 0.2,
    lr_z_log_beta = 0.05,
    lr_z_log_mf = 0.05,
    lr_z_log_rf = 0.05,
    pbias_prior = PriorHyperparams(sigma=0.05,l=10000.0, nu=1),
    lr_z_log_H_atm = 0.05,
    lr_z_logit_cloud = 0.05,
    precip_lapse_enabled=False,
    alpha_t2m=2.5,
    dt=20.0,
    mu_H_atm=20.0)

"""
CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="denali",
    results_subdir="inverse",
    lambda_u=2.0e-6,
    lambda_s=2.0e-6,
    lambda_bed=2.0e-6,
    lambda_e=2e-5,
    lambda_snow=2e-5,
    lambda_dhdt=0,
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1,
    init_from_observed_geometry = False,
    debris_factor=0.5,
    #bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    #lr_z_bed=0.0325
    bed_prior = PriorHyperparams(sigma=250,    l=1000.0, nu=1),
    lr_z_bed=0.5
)
"""
