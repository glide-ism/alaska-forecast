"""

Chugach domain configuration

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import (BedConditioningConfig, GlacierConfig,
                                    PriorHyperparams, Schedule)
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, DivideFluxSpec, ExtentSpec, SnowlineSpec, SurfaceSpec,
    VelocitySpec,
)

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="chugach",
    results_subdir="inverse",
    smb_model = "enthalpy",
    anomaly_integration="mean_anomaly",
    min_level=0,
    max_level=2,
    max_iters=(20, 50, 500),
    observations=(
        SurfaceSpec(weight=Schedule(final=2e-6, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 2e-6)),
        VelocitySpec(weight=2.0e-6,outlier_threshold=3000.0),
        ExtentSpec(weight=2e-5, s_H=10.0),
        BedSpec(weight=0.0e-6),
        SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5),
                     s_smb=0.5),
        DhdtSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1e-5)),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
                         enabled=True,
                         sigma_picks=25,
                         sigma_dem=25,
                         pcg_rtol=1e-3,
                         pcg_rtol_adjoint=1e-2),
    ssa_damping=1.0,
    sliding_m=1.0 / 3,
    u_reg=10.0,
    beta_init=1.0,
    init_from_observed_geometry = True,
    use_avalanche_model=True,
    avalanche_hoisted=True,
    debris_factor=0.5,
    
    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    lr_z_bed=0.025,
    alpha_t2m=2.5,
    dt=20.0,
    water_drag=0.0001,
)
