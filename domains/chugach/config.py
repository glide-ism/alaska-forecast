"""
Chugach domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="chugach",
    results_subdir="inverse_brier_test",
    observations=(
        SurfaceSpec(weight=2.0e-6),
        VelocitySpec(weight=2.0e-6),
        ExtentSpec(weight=2e-5),
        BedSpec(weight=2.0e-6),
        SnowlineSpec(weight=2e-5, s_smb=0.5),
        DhdtSpec(),
    ),
    loss_scale=1e-3,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1,
    init_from_observed_geometry = True,
    debris_factor=0.5,
    sigma_log_mf = 1.0,
    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
)
