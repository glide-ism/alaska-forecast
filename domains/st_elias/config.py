"""
St. Elias domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams

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
    lr_z_bed=0.0175,
    #t_start=1512,
    lambda_u=4.0e-6,
    lambda_s=2.0e-6,
    lambda_bed=2.0e-6,
    lambda_e=2e-5,
    lambda_snow=0e-5,
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1,
    init_from_observed_geometry = True,
    debris_factor=0.5,
    #pbias_prior = PriorHyperparams(sigma=0.1,    l=10000.0, nu=1),
    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    #log_beta_prior = PriorHyperparams(sigma=3.0,    l=1000.0,  nu=1),
    depth_blend=0.1
    #A_glen=2e-16
)
"""
CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="st_elias",
    #t_start=512,
    lambda_u=5.0e-6,
    lambda_s=1.0e-6,
    lambda_bed=1.0e-6,
    lambda_e=1e-5,
    lambda_snow=5e-6,
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    #sliding_m=1.0,
    #beta_init=0.05,
    init_from_observed_geometry = True,
    debris_factor=0.0,
    #pbias_prior = PriorHyperparams(sigma=0.1,    l=10000.0, nu=1),
    bed_prior = PriorHyperparams(sigma=500,    l=1000.0, nu=1),
    log_beta_prior = PriorHyperparams(sigma=3.0,    l=1000.0,  nu=1),
    depth_blend=0.1
    #A_glen=2e-16
)
"""
