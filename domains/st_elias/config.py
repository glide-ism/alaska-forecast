"""
St. Elias domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams, Schedule

_HERE = Path(__file__).parent

# Note that the first 100 iterations or so should be done with lambda_snow=0 to allow the model to avoid basin piracy

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="st_elias",
    results_subdir="inverse",
    # St. Elias is large; start one level coarser than the other domains, and
    # the bed prior is wider here so the bed step needs to be smaller.
    min_level=1,
    max_level=3,
    max_iters=(0, 20, 50, 500),
    lambda_u=4.0e-6,
    lambda_s=2.0e-6,
    lambda_bed=2.0e-6,
    lambda_e=2e-5,
    lambda_snow=Schedule(final=1e-5,ramp=lambda i,level: 0.0 if (i<50 and level==3) else 1e-5),
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1,
    init_from_observed_geometry = True,
    debris_factor=0.5,
    sigma_log_mf = 1.0,
    sigma_log_rf = 0.2,
    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    lr_z_bed=0.0175,
    depth_blend=0.1
)
