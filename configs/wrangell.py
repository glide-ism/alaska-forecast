"""
Wrangell domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from glacier_inverse.config import GlacierConfig

WRANGELL = GlacierConfig(
    base_dir="./domains/wrangell/",
    vti_base_name="wrangell",
    lambda_u=2.0e-6,
    lambda_s=2.0e-6,
    lambda_bed=2.0e-6,
    lambda_e=2e-5,
    lambda_snow=2e-5,
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1, 
    depth_blend=0.0,
    init_from_observed_geometry = False,
    debris_factor=0.0
)
