"""
St. Elias domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from glacier_inverse.config import GlacierConfig

DENALI = GlacierConfig(
    base_dir="./domains/denali/",
    vti_base_name="denali",
    sigma_log_mf = 1.,
    sigma_log_rf = 1.,
    sigma_bed=30.0,
    mu_rf=50.0,
    init_from_observed_geometry=True
)
