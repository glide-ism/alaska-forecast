"""
Wrangell domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from pathlib import Path

from glacier_inverse.config import GlacierConfig, PriorHyperparams, Schedule

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="wrangell",
    results_subdir="inverse_newclimate",
    lambda_u=2.0e-6,
    lambda_s=2.0e-6,
    lambda_bed=2.0e-6,
    lambda_e=2e-5,
    lambda_snow=Schedule(final=1e-5,ramp=lambda i,level: 0.0 if (i<50 and level==2) else 1e-5),
    lambda_dhdt=Schedule(final=2e-6,ramp=lambda i,level: 0.0 if (i<50 and level==2) else 2e-6),
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1,
    init_from_observed_geometry = True,
    use_avalanche_model=True,
    debris_factor=0.5,
    sigma_log_rf = 0.1,
    sigma_log_mf = 0.2,
    lr_z_log_mf = 0.05,
    lr_z_log_rf = 0.05,
    surge_biased_likelihood=False,
    pbias_prior = PriorHyperparams(sigma=0.05,l=10000.0, nu=1)
)

"""
CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="wrangell",
    results_subdir="inverse",
    lambda_u=2.0e-6,
    lambda_s=2.0e-6,
    lambda_bed=2.0e-6,
    lambda_e=2e-5,
    lambda_snow=2e-5,
    lambda_dhdt=1e-5,
    loss_scale=1e-3,
    s_smb=0.5,
    ssa_damping=1.0,
    sliding_m=1.0,
    beta_init=0.1,
    init_from_observed_geometry = True,
    debris_factor=0.5,
    sigma_log_rf = 0.1,
    sigma_log_mf = 0.1,
    #bed_prior = PriorHyperparams(sigma=250,    l=1000.0, nu=1),
    #lr_z_bed = 0.5,
    lr_z_log_mf = 0.05,
    lr_z_log_rf = 0.05
)
"""
