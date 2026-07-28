"""
St. Elias domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams, Schedule
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, DivideFluxSpec, ExtentSpec, SnowlineSpec, SurfaceSpec,
    VelocitySpec,
)

_HERE = Path(__file__).parent

# Note that the first 100 iterations or so should be done with lambda_snow=0 to allow the model to avoid basin piracy

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="st_elias",
    results_subdir="inverse_enthalpy_2",
    #t_start = 1012,
    smb_model = "enthalpy",
    anomaly_integration="mean_anomaly",
    # St. Elias is large; start one level coarser than the other domains, and
    # the bed prior is wider here so the bed step needs to be smaller.
    min_level=1,
    max_level=2,
    max_iters=(0, 20, 500),#, 500),
    observations=(
        SurfaceSpec(weight=Schedule(final=2e-6, ramp=lambda i, level: 0.0 if (i < 0 and level == 3) else 2e-6)),
        VelocitySpec(weight=2.0e-6,nu=3),
        ExtentSpec(weight=2e-5, s_H=10.0),
        BedSpec(weight=2.0e-6,sigma_dem=1.0),
        SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 3) else 1e-5),
                     s_smb=0.5),
        DhdtSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 3) else 1e-5)),
        # Cross-basin flux penalty: quadratic (unlike the Huberized velocity
        # misfit, which a fictitious ice stream saturates), so excavating the
        # Bagley->Yahtse pass stays expensive at any flux. Constant weight —
        # with no flux at init it exerts no pull until piracy is attempted.
        # z_min keeps only divides ABOVE that elevation: without it, RGI
        # boundaries through slow coalesced piedmont ice (Agassiz/Seward in
        # the Malaspina) get penalized and the optimizer walls them off.
        #DivideFluxSpec(weight=1e-4, z_min=800.0),
    ),
    loss_scale=1e-3,
    ssa_damping=1.0,
    sliding_m=1./3,
    beta_init=1.0,
    init_from_observed_geometry = True,
    use_avalanche_model=True,
    # Hoisted avalanche: R applied once per iteration instead of per-step —
    # valid here because smb_model="enthalpy" and per-step precip variation is
    # a scalar multiplier. See GlacierConfig.avalanche_hoisted before copying
    # this to another domain or changing the precip forcing.
    avalanche_hoisted=True,
    debris_factor=0.5,
    sigma_log_rf = 0.1,
    sigma_log_mf = 0.2,
    lr_z_bed = 0.2,
    lr_z_log_beta = 0.05,
    #bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    #lr_z_bed=0.0175,
    pbias_prior = PriorHyperparams(sigma=0.05,l=10000.0, nu=1),
    lr_z_log_H_atm = 0.05,
    lr_z_logit_cloud = 0.05,
    precip_lapse_enabled=False,
    alpha_t2m=2.5,
    dt=20.0,
    mu_H_atm=10.0
)
