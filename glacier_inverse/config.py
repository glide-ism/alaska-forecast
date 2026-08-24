"""
Configuration dataclasses for glacier inverse problems.

A single GlacierConfig instance captures every numerical knob — physical
hyperparameters, prior hyperparameters, observation noise, solver tolerances,
loss weights, paths — so the four tasks (inverse, rto, posterior, sensitivity)
share the same physical model by construction.
"""
import inspect
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Union

# The GlacierConfig fields that may be given as a schedule instead of a constant.
# Per-observation weights are also schedulable, but live on the observation
# specs (see observations.py) and are resolved by Observation.weight_at.
SCHEDULABLE_WEIGHTS = ("loss_scale",)


def _accepts_two_positional(fn: Callable) -> bool:
    """True if `fn` can be called with two positional args, f(i, level); False if
    it only takes one, f(i). Used to support both schedule arities."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return False  # builtins without an introspectable signature: assume f(i)
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    has_var_positional = any(p.kind is p.VAR_POSITIONAL for p in params)
    return has_var_positional or len(positional) >= 2


@dataclass(frozen=True)
class Schedule:
    """A loss weight that follows a continuation ramp during the initial MAP
    solve only, collapsing to a single steady-state value everywhere else.

    `final` is the contract weight shared by all four tasks: RTO, posterior, and
    sensitivity always use it, so they target the same objective the MAP solve
    converges to. `ramp` is a callable f(i) or f(i, level) honored only when a
    driver opts into scheduling (inverse.py); it should asymptote to `final`.
    """
    final: float
    ramp: Callable[..., float]

    def at(self, iteration: int, level: int) -> float:
        fn = self.ramp
        return fn(iteration, level) if _accepts_two_positional(fn) else fn(iteration)


# A schedulable loss weight. `schedule=` on GlacierConfig.at_iteration selects
# whether continuation is honored (the initial inverse solve) or the steady-state
# value is used (RTO / posterior / sensitivity).
LossWeight = Union[float, Schedule, Callable[..., float]]


def resolve_weight(value: LossWeight, iteration: int, level: int, *,
                   schedule: bool) -> float:
    """Resolve a (possibly scheduled) loss weight to a float.

    * constants pass through unchanged;
    * a Schedule yields its continuation value when `schedule` is True, else its
      steady-state `final`;
    * a bare callable is an inverse-only shorthand: honored when `schedule` is
      True, but rejected otherwise — it declares no steady state, so RTO and the
      analysis tasks cannot use it consistently (wrap it in Schedule(final=...)).
    """
    if isinstance(value, Schedule):
        return value.at(iteration, level) if schedule else value.final
    if callable(value):
        if not schedule:
            raise TypeError(
                "loss weight is a bare schedule callable, but this task does not "
                "use scheduling. Wrap it as Schedule(final=..., ramp=...) so the "
                "steady-state value is defined for RTO / posterior / sensitivity."
            )
        return value(iteration, level) if _accepts_two_positional(value) else value(iteration)
    return value


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
class BedConditioningConfig:
    """GP-posterior-as-prior for the bed: condition the Matern bed prior on
    the flightline picks and (optionally) bed=DEM over every ice-free /
    out-of-domain pixel, so z_bed parametrizes the CONDITIONAL field
    (Matheron's rule: bed = bed_mean + Map(z_bed) + (Q+D)^-1 D (b - ...)).

    Enabling this CHANGES THE BED PARAMETRIZATION:
      - checkpoints carry a "bed_parametrization" tag and are converted
        exactly on load (io.load_whitened_params_into needs `priors=`);
      - the BedObservation soft likelihood must carry weight 0 (the data is
        already in the map — GlacierProblem warns loudly otherwise);
      - lr_z_bed generally needs per-domain retuning: the conditional map has
        different curvature near data. Start from the legacy value and watch
        the first level.

    The correction is solved matrix-free by flexible PCG (preconditioned with
    the prior multigrid solve); cold iteration counts scale like
    sigma_prior/sigma_obs, so keep the sigmas ~10 m — conditioning at +-10 m
    already pins the bed far harder than the old soft anchor did (do NOT port
    a sigma_dem=1.0 soft-anchor experiment here; it only buys PCG iterations).
    """
    enabled: bool = False
    sigma_picks: float = 10.0    # m, per-pick noise (BedSpec.sigma semantics)
    sigma_dem: float = 10.0      # m, off-ice bed=DEM noise (BedSpec.sigma_dem)
    include_off_ice: bool = True
    # Cells where the DEM is exactly 0.0 are void fill at the land-DEM /
    # bathymetry seam (nearshore, fjord heads), not real bed — anchoring them
    # pins fjord bottoms at sea level. Excluded by default: the conditional
    # field there is interpolated from the surrounding anchored pixels, and a
    # spurious sill near a terminus gets corrected by the surface misfit
    # (ice overrides it -> surface too high -> bed pushed down). Real land at
    # exactly 0.0f is essentially nonexistent in float composites.
    exclude_zero_dem: bool = True
    pcg_rtol: float = 1e-4
    # Backward-pass tolerance: a ~1% gradient is ample for SGD, and rough
    # loss cotangents stall float32 PCG well before 1e-4 (burning maxiter
    # every iteration — the dominant cost otherwise).
    pcg_rtol_adjoint: float = 1e-2
    # "shifted" (default): precondition with the diagonally shifted factor
    # (L + d)^-2, d = (tau/dx)sqrt(D) — O(10) iterations for every solve,
    # independent of the sigma ratio and rhs roughness, at one extra
    # multigrid hierarchy of memory. "prior": the original C preconditioner
    # (kept for comparison/fallback; iterations ~ sigma_prior/sigma_obs).
    pcg_preconditioner: str = "shifted"
    # Cold solves at sigma_prior/sigma_obs = 25 land around 250-500 iterations
    # (rough right-hand sides are the worst); warm-started solves during
    # optimization take a handful.
    pcg_maxiter: int = 800
    warm_start: bool = True


@dataclass(frozen=True)
class GlacierConfig:
    base_dir: str

    # Grid / multigrid
    n_levels: int = 6

    # Time stepping
    dt: float = 20.0
    t_start: float = 1012.0
    t_end: float = 2012.0

    # Truncated backpropagation through time: steps ending at or before this
    # time run under torch.no_grad - full spin-up physics (J is unchanged),
    # no adjoint solves, no checkpointed state. None (default) differentiates
    # the whole run.
    #
    # CAUTION - the misfit forcing decaying to nothing backward in time does
    # NOT mean the deep-time chain contributes nothing: for time-constant
    # parameters (beta, bed) the per-step contributions accumulate coherently
    # over the glacier response time, and an FD test on delta (level 2,
    # dt=20) confirms the deep content is real gradient signal. The cutoff
    # must precede the first observation epoch by several response times.
    # Measured gradient error on delta (epochs 2000-2020): cutoff 1500 ->
    # 0.5% (beta) / 0.2% (bed) at ~45% backward savings; 1700 -> 7%/3%;
    # 1850 -> 31%/13%; 1950 -> 82%/31%. Rerun that sweep (vary this value,
    # compare grads against None) when adopting it on a new domain.
    #
    # Guards: simulate() raises if any recorded state/volume time falls in
    # the no-grad window, and SimResult.H_boundary.grad (populated by
    # backward) is the adjoint seed of everything discarded - it tracks the
    # error monotonically (delta: |.|~2e-5 at 0.5% error, ~9e-4 at 30%), so
    # watch it drift as the optimizer moves the state. Shared by all
    # gradient-computing tasks (MAP, RTO, sensitivity) so they optimize the
    # same objective approximation.
    grad_start_time: Optional[float] = None
    base_anomaly_year: int = 2012
    alpha_t2m: float = 2.2
    
    base_precip_year: int = 2012
    alpha_precip: float = 0.0

    # How the multi-year anomaly signal is integrated over each ice-dynamics
    # step. The anomaly record is piecewise constant per calendar year and smb
    # is an exogenous source, so the correct discrete forcing is its exact
    # interval integral, not a point sample:
    #   "end"          — legacy: sample the anomaly at the step's end time and
    #                    hold it for the whole step. Aliases interannual
    #                    variability and makes the forcing depend on how the
    #                    scheduler partitions time; kept for comparison.
    #   "mean_anomaly" — one SMB evaluation per step at the overlap-weighted
    #                    interval-mean anomaly. Fixes aliasing and partition
    #                    dependence; retains a small Jensen bias (smb is
    #                    nonlinear in temperature). The default.
    #   "annual"       — exact: one SMB evaluation per calendar year overlapped
    #                    by the step, combined with overlap weights (years with
    #                    identical clamped anomalies merge into one call). Cost
    #                    scales with simulated years rather than steps.
    # The precip-anomaly multiplier follows the same weights ("end" keeps its
    # legacy endpoint trapezoid) and is held at its step mean in every mode.
    anomaly_integration: str = "mean_anomaly"

    # Field priors (Matern)
    bed_prior:      PriorHyperparams = PriorHyperparams(sigma=250.0,    l=1000.0,  nu=1)
    mean_prior:     PriorHyperparams = PriorHyperparams(sigma=1000.0,   l=10000.0, nu=1)
    log_beta_prior: PriorHyperparams = PriorHyperparams(sigma=3.0,      l=1000.0,  nu=1)
    pbias_prior:    PriorHyperparams = PriorHyperparams(sigma=0.05,     l=10000.0, nu=1)
    tbias_prior:    PriorHyperparams = PriorHyperparams(sigma=0.5,     l=10000.0, nu=1)

    # Optional additive temperature bias field (units: K). A Matern GP field
    # added to the monthly t2m before the anomaly shift — the spatial,
    # time-constant complement of the (uniform, time-varying) anomaly record.
    # Its natural role is absorbing error in the preprocessing lapse
    # correction (fixed 6.5 K/km against a fixed DEM) and the DEM-vs-model-
    # surface mismatch. Unlike pbias it is not logarithmized: it acts
    # additively and may be negative. Gated off by default so already-tuned
    # domains are bit-identical: when disabled, z_tbias stays at 0 (= prior
    # median = 0 K), never enters the optimizer or the forward graph, and no
    # Matern hierarchy is built for it. Caveats when enabling:
    #   * lr_z_tbias is coupled to tbias_prior like every other lr/prior pair;
    #   * t2m becomes a differentiable SMB input, so each checkpoint-segment
    #     backward transiently copies a (12, ny, nx) t2m gradient off the
    #     glare grid;
    #   * the spatially uniform component is degenerate with the anomaly
    #     scaling (alpha_t2m) over the calibration window — the GP prior is
    #     what pins it, exactly as for pbias;
    #   * under the enthalpy backend, tbias shifts t2m ONLY: t_base (the
    #     static substrate proxy) deliberately stays at the unbiased
    #     climatology, matching how the temperature anomaly is treated.
    tbias_enabled: bool = False

    # SMB backend: "temperature_index" (glare's ImprovedTemperatureIndex, the
    # default) or "enthalpy" (glare's EnthalpyModel — an enthalpy formulation of
    # annual snow dynamics; same forcing, same specific SMB, different
    # parameters). Both models' parameters and priors live in this config
    # simultaneously; only the active model's scalars enter the optimizer and
    # the forward graph (the inactive model's whitened z stay at 0 = prior
    # median, contributing exactly zero to the prior loss).
    smb_model: str = "temperature_index"

    # Scalar SMB priors (log-normal). mu_log_* is derived as log(mu_*).
    # These belong to the temperature-index model.
    mu_rf: float = 50.0
    mu_mf: float = 1.0
    sigma_log_rf: float = 0.1
    sigma_log_mf: float = 0.1
    debris_factor: float = 0.5 # Amount by which debris cover reduces melt of bare ice.

    # Enthalpy-model inverted scalars. H_atm (lumped sensible/longwave heat
    # transfer) is inferred as log(H_atm in W m-2 K-1); the effective normal
    # shortwave is inferred as logit(f) of the cloud factor f in (0, 1), with
    # q_sw_insol = f * q_sw_clear. Physical values passed to the model are
    # converted to J m-2 yr-1 (K-1) via SECONDS_PER_YEAR.
    mu_H_atm:          float = 10.0   # W m-2 K-1, prior median
    sigma_log_H_atm:   float = 0.4
    mu_cloud_factor:   float = 0.6    # prior median of f in (0, 1)
    sigma_logit_cloud: float = 0.5

    # Enthalpy-model fixed constants (not inverted; per-second SI units,
    # converted by SECONDS_PER_YEAR where they are fluxes).
    q_sw_clear:  float = 1000.0  # W m-2, direct normal shortwave at sea level
    q_sw_bulk:   float = 0.0     # W m-2, insolation-independent shortwave
    H_base0:     float = 0.6     # W m-2 K-1, basal conductance at zero snow mass
    albedo_snow: float = 0.9
    albedo_ice:  float = 0.4
    M_albedo:    float = 20.0    # kg m-2, snow-cover albedo transition mass
    # Sub-monthly stochastic temperature deviations: one seeded (12, n_substeps)
    # realization is drawn once at problem build and reused for every time step,
    # iteration, and checkpoint recomputation, keeping the objective (and the
    # checkpointed adjoint) deterministic.
    enthalpy_n_substeps: int = 30
    enthalpy_seed:       int = 0

    # Keep glare's six diagnostic state cubes (M, E, runoff, ice_melt,
    # t_surface, albedo) individually resident. The inversion only consumes
    # smb, so by default they share one scratch buffer (~5 x (12, ny, nx) of
    # VRAM back); flip this on — or call
    # problem.smb_model.grid.materialize_state_fields() + one forward — when
    # the diagnostics are wanted. Ignored by glare versions predating the flag.
    enthalpy_materialize_state: bool = False

    # Elevation-dependent precip depletion (optional). Adds a softplus ramp,
    #   exp(-tau) * w * log(1 + exp((z - z0) / w)),
    # that is *subtracted* from the log-precip bias, reducing precip above the
    # onset elevation z0 to capture high-elevation moisture loss the climate
    # forcing misses. tau (log of the depletion length scale) and z0 are two
    # learnable scalars with normal priors N(mu, sigma) below; w is fixed. The
    # term is gated off by default so already-tuned domains are unaffected until
    # they set precip_lapse_enabled=True (and, since lr is coupled to the prior,
    # retune lr_z_tau / lr_z_z0 if they change the prior widths).
    precip_lapse_enabled: bool = False
    precip_lapse_w: float = 300.0   # m, fixed transition sharpness
    mu_tau:    float = 9.0          # prior mean of tau = log(depletion length scale [m])
    sigma_tau: float = 1.0
    mu_z0:     float = 2000.0       # m, prior mean of depletion-onset elevation
    sigma_z0:  float = 500.0

    # Observations. A tuple of frozen spec objects (SurfaceSpec, VelocitySpec,
    # ExtentSpec, BedSpec, SnowlineSpec, DhdtSpec — see observations.py), each
    # carrying its own noise sigma, loss weight (constant or Schedule), and
    # optional acquisition-time override; acquisition times normally come from
    # variable-level attrs on the input files. None selects the standard six
    # products with library-default hyperparameters. GlacierProblem builds the
    # loaded Observation objects from these specs; products whose input file
    # is absent are skipped gracefully.
    observations: tuple = None

    # Global loss scale. May be a constant, or a Schedule(final=, ramp=) for
    # continuation during the initial inverse solve only. The steady-state
    # `final` is the value RTO / posterior / sensitivity see, so all tasks
    # target the same objective; only inverse.py honors the ramp. Per-product
    # weights follow the same contract but live on the observation specs.
    loss_scale:  LossWeight = 1e-4

    # Ice rheology
    rho_ice:  float = 917.0
    rho_water: float = 1000.0   # kg/m^3, freshwater (proglacial-lake termini)
    gravity:  float = 9.81
    n_glen:   int   = 3
    A_glen:   float = 1e-16    # Pa^-n s^-1
    eps_reg:  float = 1e-6
    H_reg:    float = 50.0

    # Stress balance approximation: "ssa" (membrane stresses only, the
    # historical physics of this repo) or "molho" (adds vertical shear; the
    # velocity splits into a depth-averaged part u and a deformational part
    # ud, with surface velocity u + ud/(n+1)). SSA is the exact ud == 0
    # restriction of MOLHO, so downstream code is scheme-agnostic: ud/vd are
    # simply zero fields under "ssa". MOLHO costs roughly 2x per solve.
    stress_scheme: str = "ssa"

    # Sliding
    beta_init:   float = 2.0
    sliding_m:   float = 1.0 / 3.0
    u_reg:       float = 1.0
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
    use_avalanche_model: bool = False

    # Avalanche-operator hyperparameters (only read when use_avalanche_model).
    # s_crit/w_trans set the deposition sigmoid (degrees), p the MFD flow
    # concentration, K the number of routing passes (Neumann truncation). At
    # K=12 the undeposited residual on St.-Elias-like terrain is ~5e-4 of the
    # slab mass and is deposited in place, so larger K buys very little; the
    # historical hard-coded value was 25.
    avalanche_s_crit: float = 30.0
    avalanche_w_trans: float = 8.0
    avalanche_p: float = 1.5
    avalanche_K: int = 12

    # avalanche_hoisted — READ THIS BEFORE ENABLING.
    #
    # When True, the avalanche redistribution R is applied ONCE per forward
    # call, to the iteration's precip field (via glare's AvalancheStep),
    # instead of inside every per-step SMB evaluation (~150 applications per
    # inversion iteration once checkpoint recomputes and adjoint rebuilds are
    # counted — ~20 s/iteration on a St. Elias-sized grid). This is exactly
    # equivalent — not an approximation — while BOTH of the following hold:
    #
    #   1. smb_model == "enthalpy". The enthalpy core consumes *total* precip
    #      and partitions rain/snow internally, so R acts on a field that the
    #      per-step temperature shift never touches. Under ETIM the snow/rain
    #      partition sits UPSTREAM of R and depends on the (anomaly-shifted)
    #      per-step t2m, so hoisting is invalid there — GlacierProblem raises.
    #
    #   2. The per-step precip variation is a SCALAR multiplier (the annual
    #      precip-anomaly ratio). R is linear, so R(c*P) = c*R(P). If the
    #      precip forcing ever varies per step by more than a scalar multiple
    #      (e.g. monthly anomaly *fields*, storm-track reweighting), hoisting
    #      becomes silently WRONG — no runtime check can catch it. Revisit
    #      this flag whenever the precip pipeline changes.
    avalanche_hoisted: bool = False

    # Solver settings
    forward_solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        relative_tolerance=1e-2, absolute_tolerance=10.0, report_norms=False,finest_steps=50))
    adjoint_solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        relative_tolerance=1e-2, absolute_tolerance=1e-5, report_norms=False,post_steps=50,finest_steps=50))

    # Bed GP conditioning (posterior-as-prior). See BedConditioningConfig.
    bed_conditioning: BedConditioningConfig = field(
        default_factory=BedConditioningConfig)

    # Input filenames (relative to base_dir/model_inputs/)
    gridded_filename:    str = "GLIDE_inputs.nc"
    flightline_filename: str = "flightlines.gpkg"
    anomaly_filename:    str = "temperature_anomaly.nc"
    precip_anomaly_filename: str = "precip_anomaly.nc"  # optional; multiplicative
    snowline_filename:   str = "gridded_snowline.nc"  # optional; ELA proxy
    debris_filename:   str = "gridded_debris.nc"  # optional; ELA proxy
    dhdt_filename:   str = "gridded_dhdt.nc"  # optional; surface elevation-change rate

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
    lr_z_bed:      float = 0.5
    lr_z_bed_mean: float = 0.5
    lr_z_log_beta: float = 0.05

    lr_z_pbias:    float = 0.001
    lr_z_tbias:    float = 0.001
    lr_z_log_mf:   float = 0.01
    lr_z_log_rf:   float = 0.01
    # Enthalpy-model scalars (Adam block, used in place of z_log_mf/z_log_rf
    # when smb_model == "enthalpy").
    lr_z_log_H_atm:   float = 0.05
    lr_z_logit_cloud: float = 0.05
    # Elevation-dependent precip depletion scalars (Adam block, only added to
    # the optimizer when precip_lapse_enabled). See the prior widths above.
    lr_z_tau:      float = 0.01
    lr_z_z0:       float = 0.01

    def at_iteration(self, iteration: int, level: int = 0, *,
                     schedule: bool = False) -> "GlacierConfig":
        """Return a copy with every schedulable loss weight resolved to a float
        at `(iteration, level)`.

        When `schedule` is False (the default, used by RTO / posterior /
        sensitivity) constants pass through and any Schedule collapses to its
        steady-state `final`, so every task targets the same objective. When
        `schedule` is True (the initial inverse solve) Schedule ramps and bare
        callables are evaluated as f(i, level) / f(i). Called per optimizer step
        by GlacierProblem.compute_loss. Per-observation weights follow the same
        contract but are resolved by Observation.weight_at — a domain config may
        set e.g. `SnowlineSpec(weight=Schedule(final=2e-4,
        ramp=lambda i, level: 0.0 if level > 0 else 2e-4))`.
        """
        overrides = {name: resolve_weight(getattr(self, name), iteration, level,
                                          schedule=schedule)
                     for name in SCHEDULABLE_WEIGHTS}
        return replace(self, **overrides)

    @property
    def output_dir(self) -> str:
        """Absolute path to this domain's active experiment directory."""
        return f"{self.base_dir}/{self.results_subdir}"

    @property
    def B_rate(self) -> float:
        """Computed rate factor used by IceDynamics rheology.B."""
        return self.A_glen ** (-1.0 / self.n_glen) / (self.rho_ice * self.gravity)
