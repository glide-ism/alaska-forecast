"""
Wrangell domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""
from glacier_inverse.config import GlacierConfig

ST_ELIAS = GlacierConfig(
    base_dir="./domains/st_elias/",
    vti_base_name="st_elias",
)
