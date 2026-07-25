"""
Delta domain configuration.

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
    vti_base_name="delta",
    results_subdir="inverse_more_nonlin",
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
    u_reg=10.0,
    beta_init=10.0,
    init_from_observed_geometry = True,
    use_avalanche_model = True,
    debris_factor=0.5,
    sigma_log_rf = 0.1,
    sigma_log_mf = 0.1,
    pbias_prior = PriorHyperparams(sigma=0.1,l=10000.0,nu=1),
    #lr_z_pbias = 0.004,
    lr_z_log_mf = 0.05,
    lr_z_log_rf = 0.05,
    precip_lapse_enabled=False,
    alpha_t2m=2.5,
    dt=20.0
    
)
