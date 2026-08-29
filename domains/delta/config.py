"""
Delta domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""

from pathlib import Path

from glacier_inverse.config import BedConditioningConfig, GlacierConfig, PriorHyperparams, Schedule
from glacier_inverse.observations import (
    BedSpec, BedSlopeSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)
import numpy as np

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="delta",
    results_subdir="inverse_molho_diffuse",
    smb_model = "enthalpy",
    # Constant (non-albedo-scaled) surface-flux offset: a-priori interior sky
    # longwave deficit + evaporative cooling. With the offset explicit, H_atm is
    # the pure dT slope, so its prior median moves to the first-principles
    # sensible+latent+longwave value and the prior is widened to honest ignorance.
    q_lw0=-40.0,
    mu_H_atm=15.0,
    sigma_log_H_atm=0.35,
    # Shortwave: the inverted scalar f is the CLEAR-SKY fraction (1 - cloud
    # fraction); direct = f*S0*I (S0 = q_sw_clear = 1361, tau^airmass lives in
    # the direct potential I) and diffuse = (f*k_clr + (1-f)*k_cld)*S0*I_dif from
    # the same f, with I_dif = monthly_diffuse_potential (sky-view x cos zenith).
    # Interior-Alaska summer cloud fraction ~0.65 -> f ~ 0.35; at that median
    # June global over the RGI is ~190 W m-2 with a diffuse fraction ~0.55.
    # NB: checkpoints from the direct-only era hold z_logit_cloud ~ +6 (f = 0.87
    # under the old 0.6 prior) -- zero it on warm start rather than carry it.
    q_sw_clear=1361.0,
    mu_cloud_factor=0.35,
    sigma_logit_cloud=0.25,
    k_diffuse_clear=0.10,
    k_diffuse_cloud=0.30,
    anomaly_integration="mean_anomaly",
    stress_scheme='molho',
    grad_start_time=1712,
    #t_start=1712,
    observations=(
        SurfaceSpec(weight=2.0e-6,nu=3),
        VelocitySpec(weight=2.0e-6, surge_biased=True,nu=3,alpha_nonsurge=20),
        ExtentSpec(weight=2e-5, s_H=10.0),
        BedSpec(weight=0.0e-6),
        SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5),
                     s_smb=0.5),
        DhdtSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5)),
        BedSlopeSpec(weight=1e-5,s_scale=5.0),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
                         enabled=True,
                         sigma_picks=100,
                         sigma_dem=100,
                         pcg_rtol=1e-3,
                         pcg_rtol_adjoint=1e-2),
    sliding_m=1./3.,
    u_reg=1.0,
    beta_init=5.0,
    init_from_observed_geometry = True,
    use_avalanche_model = True,
    avalanche_hoisted=True,
    debris_factor=0.5,
    bed_prior = PriorHyperparams(sigma=500,    l=1000.0, nu=1),
    lr_z_bed=0.05,
    alpha_t2m=2.5,
    dt=20.0,
    tbias_enabled=True,
    mu_log_beta = np.log(5.0),
    log_beta_prior = PriorHyperparams(sigma=1./3.,    l=1000.0, nu=1),
    lr_z_log_beta=0.05*9*9,
    max_level=2,
    max_iters=(50,50,500),
    lr_z_pbias=Schedule(final=0.001, ramp=lambda i,level:0.0 if (i<50 and level==2) else 0.001),
    lr_z_tbias=Schedule(final=0.001,ramp=lambda i,level:0.0 if (i<50 and level==2) else 0.001),
    lr_z_log_H_atm=Schedule(final=0.05,ramp=lambda i,level:0.0 if (i<50 and level==2) else 0.05),
    lr_z_logit_cloud=Schedule(final=0.05,ramp=lambda i,level:0.0 if (i<50 and level==2) else 0.05),
)

"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams, Schedule
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)



_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="delta",
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
    sliding_m=0.25,
    #u_reg=10.0,
    beta_init=3.0,
    init_from_observed_geometry = True,
    use_avalanche_model = True,
    debris_factor=0.5,
    sigma_log_rf = 0.1,
    sigma_log_mf = 0.1,
    pbias_prior = PriorHyperparams(sigma=0.05,l=10000.0,nu=1),
    lr_z_bed = 0.2,
    #lr_z_pbias = 0.004,
    #lr_z_log_mf = 0.05,
    #lr_z_log_rf = 0.05,
    lr_z_log_H_atm = 0.05,
    lr_z_logit_cloud = 0.05,
    precip_lapse_enabled=False,
    alpha_t2m=2.5,
    dt=20.0
    
)
"""
