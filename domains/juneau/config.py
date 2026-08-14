"""
Juneau domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import BedConditioningConfig,GlacierConfig, PriorHyperparams, Schedule
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="juneau",
    results_subdir="inverse_molho_2",
    smb_model = "enthalpy",
    anomaly_integration="mean_anomaly",
    stress_scheme='molho',
    grad_start_time=1812,    
    observations=(
        SurfaceSpec(weight=2.0e-6),
        VelocitySpec(weight=2.0e-6, surge_biased=True),
        ExtentSpec(weight=2e-5),
        BedSpec(weight=2.0e-6),
        SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5),
                     s_smb=0.5),
        DhdtSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5)),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
                         enabled=False,
                         sigma_picks=25,
                         sigma_dem=25,
                         pcg_rtol=1e-3,
                         pcg_rtol_adjoint=1e-2),
    sliding_m=1.0,
    u_reg=10.0,
    beta_init=0.5,
    init_from_observed_geometry = True,
    use_avalanche_model=True,
    avalanche_hoisted=True,
    debris_factor=0.5,

    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    lr_z_bed=0.025,
    alpha_t2m=2.5,
    dt=20.0,
    tbias_enabled=True,
    #max_level=0
    
)

