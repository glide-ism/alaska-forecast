"""
Wrangell domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from glacier_inverse.config import GlacierConfig

JUNEAU = GlacierConfig(
    base_dir="./domains/juneau/",
    vti_base_name="juneau",
    sigma_log_mf=1.0,
    sigma_log_rf=1.0

    #alpha_t2m=1.0,
    #mu_mf=1.0
)
