"""
Configuration dataclasses for glacier inverse problems.

A single GlacierConfig instance captures every numerical knob — physical
hyperparameters, prior hyperparameters, observation noise, solver tolerances,
loss weights, paths — so the four tasks (inverse, rto, posterior, sensitivity)
share the same physical model by construction.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PriorHyperparams:
    sigma: float
    l: float
    nu: int


@dataclass(frozen=True)
class SolverConfig:
    coarsest_steps: int = 200
    pre_steps: int = 10
    post_steps: int = 50
    finest_steps: int = 50
    relative_tolerance: float = 1e-2
    absolute_tolerance: float = 10.0
    report_norms: bool = False


@dataclass(frozen=True)
class GlacierConfig:
    base_dir: str

    # Grid / multigrid
    n_levels: int = 6

    # Time stepping
    dt: float = 20.0
    t_start: float = 1012.0
    t_end: float = 2012.0
    base_anomaly_year: int = 2012
    alpha_t2m: float = 2.2
    
    base_precip_year: int = 2012
    alpha_precip: float = 0.2

    # Field priors (Matern)
    bed_prior:      PriorHyperparams = PriorHyperparams(sigma=250.0,  l=1000.0,  nu=1)
    mean_prior:     PriorHyperparams = PriorHyperparams(sigma=1000.0, l=10000.0, nu=1)
    log_beta_prior: PriorHyperparams = PriorHyperparams(sigma=3.0,    l=1000.0,  nu=1)
    pbias_prior:    PriorHyperparams = PriorHyperparams(sigma=0.1,    l=10000.0, nu=1)

    # Scalar SMB priors (log-normal). mu_log_* is derived as log(mu_*).
    mu_rf: float = 50.0
    mu_mf: float = 2.0
    sigma_log_rf: float = 0.1
    sigma_log_mf: float = 0.1
    debris_factor: float = 0.5 # Amount by which debris cover reduces melt of bare ice. 

    # Observation noise
    sigma_s: float = 10.0
    sigma_u: float = 10.0
    sigma_bed: float = 10.0

    # Loss weights. lambda_bed unified to 2e-5 (was 1e-5 in pre-refactor rto_sample.py).
    loss_scale: float = 1e-4
    lambda_s:    float = 2e-5
    lambda_u:    float = 2e-5
    lambda_e:    float = 2e-4
    lambda_bed:  float = 2e-5
    lambda_snow: float = 2e-4    # weight on the snowline (ELA) BCE term
    nu_s:   float = 1.0
    nu_u:   float = 1.0
    nu_bed: float = 1.0
    s_H:    float = 10.0
    # SMB -> logit scale for the snowline term. The model SMB (m ice-eq/yr)
    # is divided by this before the sigmoid, so sigmoid(SMB / s_smb) reads as
    # P(cell is above the ELA, i.e. in the accumulation area).
    s_smb:  float = 0.2

    # Ice rheology
    rho_ice:  float = 917.0
    rho_water: float = 1000.0   # kg/m^3, freshwater (proglacial-lake termini)
    gravity:  float = 9.81
    n_glen:   int   = 3
    A_glen:   float = 1e-16    # Pa^-n s^-1
    eps_reg:  float = 1e-5

    # Sliding
    beta_init:   float = 2.0
    sliding_m:   float = 1.0 / 3.0
    water_drag:  float = 0.01

    # Calving / geometry
    calving_rate: float = 250.0
    sigmoid_c:    float = 0.1
    sigmoid_k:    float = 4.0
    depth_blend:  float = 0.1    # weight on new bed-derived depth vs prior depth
    # Seed the integration from the thickness implied by the observed surface
    # (S_obs - bed, with the hydrostatic value for floating ice) instead of the
    # ice-free state. Matters for tidewater hysteresis. Not differentiated.
    init_from_observed_geometry: bool = False
    init_H_floor: float = 0.1    # minimum/ice-free thickness used when seeding

    # Solver settings
    forward_solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        relative_tolerance=1e-2, absolute_tolerance=10.0))
    adjoint_solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        relative_tolerance=1e-3, absolute_tolerance=1e-6))
    ssa_damping: float = 1.0

    # Input filenames (relative to base_dir/model_inputs/)
    gridded_filename:    str = "GLIDE_inputs.nc"
    flightline_filename: str = "flightlines.gpkg"
    anomaly_filename:    str = "temperature_anomaly.nc"
    precip_anomaly_filename: str = "precip_anomaly.nc"  # optional; multiplicative
    snowline_filename:   str = "gridded_snowline.nc"  # optional; ELA proxy
    debris_filename:   str = "gridded_debris.nc"  # optional; ELA proxy

    # Diagnostics
    vti_base_name: str = "glacier"

    # Experiment subdirectory (under base_dir). Writers (inverse) write here;
    # readers (posterior, sensitivity, rto continuation) read from here. Edit
    # the domain config to switch experiments rather than editing each driver.
    results_subdir: str = "inverse"

    # Multigrid schedule. max_level is the coarsest grid the solver starts on;
    # min_level is the finest grid it ends on. max_iters[level] is the number
    # of optimizer iterations spent at each level (indexed by the level number,
    # so unused entries below min_level can be 0 or any placeholder).
    min_level: int = 0
    max_level: int = 2
    max_iters: tuple = (20, 50, 500)

    # Per-parameter learning rates. These are tightly coupled to the prior
    # hyperparameters above — in whitened coordinates the natural step is set
    # by the prior curvature, so a domain that changes a prior typically has
    # to retune the corresponding lr. SGD on the field params, Adam on the
    # scalar / smooth params.
    lr_z_bed:      float = 0.0325
    lr_z_bed_mean: float = 0.5
    lr_z_log_beta: float = 0.25
    lr_z_pbias:    float = 0.001
    lr_z_log_mf:   float = 0.01
    lr_z_log_rf:   float = 0.01

    @property
    def output_dir(self) -> str:
        """Absolute path to this domain's active experiment directory."""
        return f"{self.base_dir}/{self.results_subdir}"

    @property
    def B_rate(self) -> float:
        """Computed rate factor used by IceDynamics rheology.B."""
        return self.A_glen ** (-1.0 / self.n_glen) / (self.rho_ice * self.gravity)
