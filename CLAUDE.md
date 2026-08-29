# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An experiment/orchestration workspace (not an installable package) for glacier inverse
modelling, forecasting, and data-collection planning over all Alaskan glaciers, by
mountain range. It is a thin application layer over four custom GPU libraries from
[github.com/glide-ism](https://github.com/glide-ism), tracked at `main`:

- `glide` — ice dynamics (`IceDynamics`, multigrid `mg`, `GlideStep`, `Field`/`VTIWriter`)
- `glare` — surface mass balance (`ImprovedTemperatureIndex`/`GlareStep` and the enthalpy
  core `EnthalpyModel`/`EnthalpyStep`)
- `ggapp` — Gaussian-process priors and whitening (`MaternPrior`, `GGaPPWhiten`/`GGaPPMap`)
- `gtic` — pulled transitively

The heavy lifting (forward solver, adjoint, priors) lives in those libs. This repo wires
them into a calibration/UQ pipeline. If you are developing the libs alongside this repo,
install them editable with the matching CUDA extra, e.g. `pip install -e ../glide[cuda12]`.

## Environment

GPU is mandatory — nearly every tensor is created with `device="cuda"` and the code mixes
`torch` and `cupy`. CuPy ships a separate wheel per CUDA major version; `requirements.txt`
defaults to `cupy-cuda12x` — edit that line for CUDA 11/13 before `pip install -r requirements.txt`.

## Commands

```bash
# 1. Download shared input bundles (common_data/ is gitignored, not checked in)
python download_common_data.py --extract          # --help for manifest/output options

# 2. Preprocess a domain's model inputs -> domains/<name>/model_inputs/
python preprocessing/make_all.py --domain-path domains/wrangell --year 2012

# 3. Sanity check the library end-to-end (builds wrangell, one forward+backward)
python smoke_test.py                              # PASS/FAIL per check, nonzero exit on failure

# 4. Run a task. Each driver picks its domain via a DOMAIN constant edited at the top.
python inverse.py        # deterministic MAP solve
python rto_sample.py     # randomize-then-optimize posterior sampling
python sensitivity.py    # information gain of ΔV predictions
python posterior.py      # empirical covariance from RTO samples
```

There is no test runner, linter, or build step — `smoke_test.py` is the closest thing to a
test. (`run_rto.sh` is stale: it calls a script that no longer exists; use `rto_sample.py`,
which loops internally.)

## Architecture

### The config-is-the-contract pattern

Everything physical is centralized in a single frozen `GlacierConfig`
(`glacier_inverse/config.py`): grid/multigrid levels, time stepping, Matérn prior
hyperparameters, rheology, solver tolerances, learning rates, input filenames, and the
tuple of **observation specs** (see below). Each domain under `domains/<name>/config.py`
constructs one `CONFIG = GlacierConfig(base_dir=...)` overriding only what differs.

This is deliberate: the four driver scripts share **the same physical model by
construction**. The split is strict — physical/prior hyperparameters live in the domain
config; per-task knobs (output paths, max iters, learning-rate schedules, warm-start path)
live in the driver. When changing behavior, decide which side it belongs to. Note that
learning rates are tightly coupled to prior hyperparameters (the natural step in whitened
coordinates is set by prior curvature), so changing a prior usually means retuning its `lr`.
Learning rates may also be `Schedule`s (see below), following the same `final`-is-the-contract
rule as the loss weights.

**Time-stamped observations.** Each observational product is an `Observation` subclass
(`glacier_inverse/observations.py`) carrying its data tensors, its acquisition time(s)
(`required_times`, calendar years), its noise/loss hyperparameters, its misfit, and an RTO
`randomized(**eps)` hook. Domain configs declare frozen **specs** (`SurfaceSpec`,
`VelocitySpec`, `ExtentSpec`, `BedSpec`, `SnowlineSpec`, `DhdtSpec` — the Brier terms are
one-sided by default (`ExtentSpec`/`SnowlineSpec(two_sided=False)` penalize only missing
ice/snow; `two_sided=True` is the full Brier score), and `VelocitySpec(mask_unobserved=)`
decides whether `v_mask == 0` mosaic pixels (no retrieved motion: all off-ice cells and
low-texture accumulation zones) are observations of ~0 velocity (False, historical) or
undefined and dropped (True) — plus the opt-in
prior-style `DivideFluxSpec` — a quadratic penalty on ice flux across RGI drainage
divides, the anti-basin-piracy term; its mask drops confluences where the velocity
mosaic shows real cross-boundary flow, and it dumps a `divide_mask` diagnostic for
validation, and the opt-in prior-style `BedSlopeSpec` — a penalty
`sum mask * |u . grad(B) / s_scale|^p` on flow-aligned bed slope, with `velocity=`
selecting the basal (`u - ud`, default)/surface/depth-averaged model velocity, plus
`p`, an `s0` deadband, and an `eps` smoothing of |.|; its
`BedSlopeObservation.flow_aligned_slope(state, bed, dx)` exposes the penalized field
for diagnostics) via
`GlacierConfig(observations=(...))`; `GlacierProblem` calls `spec.build(ctx)` to load the
data on the cropped grid (specs return `None` when the product file is absent — the term
simply doesn't exist). Acquisition times come from **variable-level NetCDF attrs**
(`time_nominal`/`time_start`/`time_end`) written by the preprocessing builders; a spec's
`time` override wins, and files without attrs fall back to `t_end` with a warning (dhdt
falls back to its legacy final-step rate). During every forward run the model emits a
`ModelState` snapshot at each required time (`scheduling.build_step_sequence` snaps the
uniform spinup grid onto observation times, variable dt), and each misfit compares against
`sim.at(t)` for its own epoch. dH/dt is a true two-snapshot rate
`(H(t1) − H(t0)) / (t1 − t0)` over the product window. The run horizon auto-extends to
`max(t_end, latest observation time)`; observation times ≤ `t_start` raise.

**Schedulable loss weights (continuation, inverse-only).** Per-observation weights
(`weight=` on each spec) and the global `loss_scale` may be a constant *or* a
`Schedule(final=, ramp=)`. This is a continuation device for the **initial MAP solve only**
(`inverse.py`), which needs a little hand-holding far from the optimum to avoid pathologies
(e.g. basin piracy before the snowline term is trustworthy). The consistency rule:
**`final` is the contract** — the steady-state weight every task targets — and the `ramp`
(`f(i)` or `f(i, level)`, which should asymptote to `final`) is an inverse-only decoration.
RTO/posterior/sensitivity always use `final`, so they sample/analyze the same objective the
MAP converged to; they never ramp because they warm-start from the MAP.

Mechanically, `Observation.weight_at(i, level, schedule=)` resolves each spec weight and
`GlacierConfig.at_iteration` resolves `loss_scale` (the only remaining entry in
`SCHEDULABLE_WEIGHTS`); `compute_loss(iteration=, level=, schedule=)` passes the resolved
float into each observation's `loss(...)`. The opt-in is a per-task driver knob:
`inverse.py` passes `schedule=True` with its per-level `i` and the multigrid `level`; every
other caller leaves the default `schedule=False` (gets `final`). A bare callable (no
`final`) still works as an inverse-only shorthand but **raises** if a non-scheduling task
hits it — it has no steady state to be consistent about. Note `inverse.py`'s `i` **resets
at each multigrid level**, and `level` runs coarsest (`max_level`) → finest (`min_level`,
usually 0), so `level == 0` is the finest grid. Example domain-config entries:

```python
from glacier_inverse import Schedule, SnowlineSpec
# defer the snowline term for the first 100 iters of each level; settle at 1e-5
SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i: 0.0 if i < 100 else 1e-5)),
# level-aware: snowline only on the finest grid during the solve
SnowlineSpec(weight=Schedule(final=1e-5, ramp=lambda i, level: 1e-5 if level == 0 else 0.0)),
```

**Schedulable learning rates (same contract).** Every `lr_z_*` field on `GlacierConfig`
(the tuple `SCHEDULABLE_LRS` in `config.py`) accepts the same `float | Schedule | callable`
as a loss weight. `GlacierConfig.learning_rates(i, level, schedule=)` resolves them to a
name→float dict (`at_iteration` resolves them too); `inverse.py` names each optimizer
param group after its config field and calls `refresh_learning_rates(i, level)` before
every step (`schedule=True`), printing the resolved set whenever it changes;
`rto_sample.py` builds its optimizers from `learning_rates(schedule=False)`, i.e. `final`.
A ramp value of `0.0` freezes that parameter for those iterations (optimizer state still
accumulates — SGD momentum / Adam moments — so the first live step is well-conditioned).
Because `i` resets per level, "hold the SMB scalars for the first 100 iterations" holds
them for the first 100 iterations *of each level* unless the ramp also inspects `level`:

```python
# freeze the melt/radiation factors until the geometry has settled
lr_z_log_mf=Schedule(final=0.01, ramp=lambda i: 0.0 if i < 100 else 0.01),
lr_z_log_rf=Schedule(final=0.01, ramp=lambda i: 0.0 if i < 100 else 0.01),
# only ever move pbias on the finest grid
lr_z_pbias=Schedule(final=1e-3, ramp=lambda i, level: 1e-3 if level == 0 else 0.0),
```

### The reusable core: `glacier_inverse/`

The library the drivers sit on top of:

- `problem.py` — `GlacierProblem` is the center of gravity. Its constructor builds the
  entire forward model from a config: `IceDynamics` + the SMB model selected by
  `config.smb_model` (`"temperature_index"` → `ImprovedTemperatureIndex`, `"enthalpy"` →
  glare's `EnthalpyModel`) + `GlacierPriors` + the observation list (from the config's
  specs) + the whitened parameter tensors. Drivers construct one and reuse it. Key types: `DomainData` (`problem.domain` —
  DEM, masks, glacier labels shared by all terms), `WhitenedParameters` (the tensors
  optimized over), `PhysicalParameters`. `problem.required_times` is the union of all
  observation epochs; `problem.get_observation(name)` fetches one product by its `name`
  ("srf", "vel", "extent", "bed", "snow", "dhdt", "divide").
- `observations.py` — the `Observation` base class + seven subclasses (data, times,
  hyperparameters, misfit, RTO noise hook) and their frozen spec dataclasses;
  `read_time_attrs` handles the file-attr → `t_end` fallback.
- `scheduling.py` — `build_step_sequence` (uniform spinup grid ∪ observation times,
  variable dt, horizon extension) and `merge_times`. Pure Python floats so snapshot dict
  keys match requested times exactly.
- `priors.py` — `GlacierPriors` (4 Matérn field priors + scalar SMB priors: log-normal
  mf/rf for the temperature-index model, log-normal H_atm and logit-normal clear-sky
  fraction f for the enthalpy model). Cheap to build without `IceDynamics`; `posterior.py` uses it
  directly to map samples back to physical space.
- `forward.py` — `simulate()` time-stepping loop (SMB → ice dynamics per step) over the
  snapped step sequence; emits a lazy-prolonging `ModelState` per requested time into
  `SimResult.states` (final state always recorded; `sim.at(t)` to fetch, compat properties
  delegate to the final state). Uses gradient checkpointing; the checkpointed smb fn does
  everything fine-grid-sized *inside* the segment: the precip multiplier (recomputed in
  backward instead of retaining ~50 full-grid copies) and the level restriction (avg_pool2d
  saves its input, so restricting outside would retain the fine smb per step). The fine smb
  is returned only for snapshot steps (`want_fine`), so `ModelState.smb_fine` is None on
  unrecorded steps. Snapshots are graph references — cheap; prolongation is lazy.
  `config.anomaly_integration` selects how the annual anomaly record is integrated over
  each step: `"mean_anomaly"` (default — one SMB call at the overlap-weighted interval-mean
  anomaly), `"annual"` (exact — one SMB call per overlapped calendar year, weighted by
  `year_overlap_weights`; averages the smb *fields* so the melt nonlinearity is respected),
  or `"end"` (legacy end-of-step sampling, kept for comparison; it aliases interannual
  variability and makes the forcing depend on the step partition). Multiple glare calls
  inside one checkpoint segment are safe because both glare adjoints re-derive their
  effective inputs (ETIM snowfall / enthalpy precip_eff) from the re-set raw inputs.
- `loss.py` — `LossTerms` (dict-backed, one data term per observation), prior terms, and
  shared helpers (`_huber`, the surge-marginal velocity likelihood — per-glacier
  marginal over the mosaic under-read factor η∈(0,1] with `Beta(alpha,1)` priors from
  `VelocitySpec.alpha_surge/alpha_nonsurge`; formed in the likelihood's own units (nats
  at `sigma`) and tempered by `loss_scale·weight·dx²` *afterwards*, so the weight
  expresses trust in the product while the α's set how much a glacier's mosaic may
  under-read the model — tempering inside the marginal would flatten it into a
  symmetric pull toward `E[η]/E[η²]·U_obs`). Identical for MAP
  (zero prior means) and RTO (nonzero whitened-space means via `PriorMeans`).
- `io.py` — VTI/PVD diagnostic writers and whitened-parameter `save`/`load`.
- `config.py`, `__init__.py` (`load_config(domain_dir)` imports a domain's `config.py` by path).

### Whitened coordinates

Optimization happens in **whitened** coordinates (`z_*` tensors), not physical units.
`GGaPPWhiten.apply` maps physical → whitened; `GGaPPMap.apply` maps back.
`GlacierProblem.physical_from(params)` does the reverse mapping for the whole parameter set.

**Bed GP conditioning (posterior-as-prior).** With
`GlacierConfig(bed_conditioning=BedConditioningConfig(enabled=True))` the bed prior is
conditioned on the bed data — flightline picks snapped to grid cells plus bed=DEM on every
ice-free/out-of-domain pixel — and `z_bed` parametrizes the **conditional** fluctuation
field: `bed = condition(Map(z_bed))` with `condition(u0) = u0 + (Q+D)⁻¹D(b − u0)`
(Matheron), solved matrix-free by prior-preconditioned flexible PCG in ggapp's
`ConditionedPrior`/`GGaPPCondition` (one extra solve per backward). **`bed_mean` stays out
of the conditioning map deliberately** — it acts purely through the re-whitened prior term
on the *unconditional* field (`compute_prior` uses `physical.bed_uncond`), exactly as in
the legacy parametrization; routing data cotangents through the mean model's near-singular
Helmholtz backward (ℓ=10 km ⇒ κ²≈1e-7) overflows float32, and the prior-only coupling
preserves the curvature structure the per-domain learning rates were tuned against.
Consequences: the `BedSpec` likelihood must carry `weight=0` (the data is in the map;
`GlacierProblem` warns on double counting); checkpoints carry a `bed_parametrization` tag
and convert on load (pass `priors=` to `load_whitened_params_into`; legacy→conditioned is
the identity on `z_bed` — the map corrects the warm-start bed onto the data, with the rms
shift logged — because the "exact" pre-image amplifies data-cell violations by
(σ_prior/σ_obs)² and is float32-hopeless). The
single shared whitened→bed map is `GlacierPriors.bed_from_whitened` (used by
`physical_from`, `posterior.py`, `sensitivity.collect_rto_samples`); RTO perturbs the
conditioning data per sample through `problem.set_bed_data_override`. The PCG is
preconditioned (default `pcg_preconditioner="shifted"`) by the inverse of the diagonally
shifted factor `(L+d)²`, `d = (τ/dx)√D`, held in a second multigrid hierarchy (ggapp's
`coefficients.shift` field; zero shift reproduces the plain operator exactly) — every
solve then takes O(10) iterations independent of data strength, and conditioning costs
under a second per iteration at St. Elias scale (`"prior"` falls back to the
C-preconditioner, iterations ~ σ_prior/σ_obs). Enabled for `st_elias`. Beware: this environment can segfault (HDF5) when
reopening a netCDF after other handles were GC'd — standalone consumers read via the
eager `_cropped_inputs(config, variables=[...])` load-and-close path, never lazy handles.
The optimized parameters are the four fields `z_bed`, `z_bed_mean`, `z_log_beta`, `z_pbias`
(precip bias) — plus, when `tbias_enabled`, a fifth field `z_tbias`: an **additive**
temperature bias in K (Matérn prior `tbias_prior`, not logarithmized — it may be negative)
added to t2m per step *inside* the checkpointed SMB fn alongside the scalar anomaly
(never pre-add it to t2m outside — that retains the biased (12,ny,nx) field plus a
full-rank grad-accumulation buffer for the whole run; inside the checkpoint the biased
t2m is recomputed in backward and only an (ny,nx) gradient leaves each segment), the
spatial time-constant complement of the uniform anomaly record (absorbs
lapse-correction/DEM error; t_base stays unbiased under the enthalpy backend; the z
tensor always exists and
is checkpointed as `"temperature_bias"`, but the Matérn hierarchy, the map, and its Adam
group exist only when enabled, so disabled domains are bit-identical; sensitivity.py
raises on tbias-enabled domains until its 5-block layout is extended) — plus the SMB
scalars of the active backend: `z_log_mf` (melt factor) and
`z_log_rf` (radiation factor) under `smb_model="temperature_index"`, or `z_log_H_atm`
(log of the lumped sensible/longwave transfer coefficient in W m⁻² K⁻¹, prior median
`mu_H_atm`) and `z_logit_cloud` (logit of the **clear-sky fraction** f = 1 − cloud
fraction, prior median `mu_cloud_factor`; direct `q_sw_insol = f·S₀` and diffuse
`q_sw_dif = (f·k_diffuse_clear + (1−f)·k_diffuse_cloud)·S₀` both derive from this one
scalar, `S₀ = q_sw_clear` = 1361 W m⁻² extraterrestrial — the clear-sky τ^airmass
attenuation lives in the direct potential, not in the base flux) under
`smb_model="enthalpy"`. The enthalpy balance is
`q = (1−α)(q_sw_bulk + q_sw_insol·I + q_sw_dif·I_dif) + q_lw0 + H_atm(T_air − T_s) + H_base(T_base − T_s)`,
with `I = monthly_solar_potential_mean` (direct beam: incidence × shadow × τ^m) and
`I_dif = monthly_diffuse_potential` (isotropic-sky view factor of the tilted,
horizon-limited surface × monthly-mean cos zenith; static geometry, no gradient, built by
`make_insolation.py` — inputs predating it load as zeros with a warning, i.e. no diffuse
term); at 60–63°N diffuse is ~half the monthly global shortwave, and it is what lights
shaded cirques and north-facing accumulation zones. The fixed (never inverted, never
checkpointed — they come from the config at every build) constants are
`q_sw_clear, k_diffuse_clear, k_diffuse_cloud, H_base0, q_sw_bulk, q_lw0, albedo_snow,
albedo_ice, M_albedo`. `q_lw0` is a
constant flux that is neither albedo-scaled nor ∝ ΔT — the offset part of net longwave /
latent exchange (clear-sky sky deficit, evaporation into sub-saturated air; negative cools,
default 0). It matters because without it a calibrated `H_atm` has to absorb the offset
(fits at ~5 instead of a first-principles ~15 W m⁻² K⁻¹), which flattens the ablation-area
balance gradient and the melt–temperature sensitivity; read `mu_H_atm` as the ΔT slope
*given* the offset. All scalar z-tensors always
exist (inactive ones sit at z = 0 = prior median, contribute zero prior loss, and are
saved/loaded for checkpoint compatibility); only the active pair joins the Adam block.
Fields use SGD, smooth/scalar params use Adam (see optimizer setup in
`inverse.py`/`rto_sample.py`). The enthalpy backend draws one seeded `(12, n_substeps)`
sub-monthly temperature-deviation realization at problem build (`enthalpy_seed`,
`enthalpy_n_substeps`) and passes it through `EnthalpyStep` on every call, so the
checkpointed objective is deterministic (an unseeded redraw would make the checkpoint's
backward recomputation linearize about a different weather realization). Both backends
apply the same debris attenuation (`1 − debris_factor·debris_fraction`); in the enthalpy
core it is the static `geometry.debris` field multiplying bare-ice melt fluxes (the pack
is untouched). Note glare's avalanche operator deposits with float `atomicAdd`, so
avalanche-enabled domains are repeatable only to ~1e-4 m/yr (both backends). Its
hyperparameters are config knobs (`avalanche_s_crit/w_trans/p/K`; K now defaults to 12 —
the K=25 historical value buys only a ~5e-4 residual-mass improvement). The
`avalanche_hoisted` flag (enthalpy-only; see its loud config docstring) applies the
redistribution once per forward call via glare's `AvalancheStep` instead of inside every
per-step SMB evaluation — exactly equivalent while per-step precip varies only by a
scalar multiplier, and worth ~20 s/iteration on St. Elias; `GlacierProblem` raises if it
is combined with ETIM, whose t2m-dependent partition sits upstream of the operator.

### Multigrid level schedule

Inversions run coarse-to-fine: from `config.max_level` (coarsest) down to `config.min_level`
(finest), spending `config.max_iters[level]` optimizer iterations per level, calling
`problem.model.set_top_level(level)` to switch. Grids are center-cropped so `ny`/`nx` are
multiples of `2**n_levels`.

### The four tasks

All read/write under `config.output_dir` = `base_dir/results_subdir` (default `inverse/`),
organized as `level_<n>/{vti/, inverse_soln.nc, torch_vars.p}`. Switch experiments by
editing `results_subdir` in the domain config, not the drivers.

1. **inverse.py** — deterministic MAP. Multigrid loop, writes per-level solution + whitened params.
2. **rto_sample.py** — RTO posterior sampler. Warm-starts from the MAP, then draws N samples,
   each with freshly randomized observations + prior means + a Bernoulli extent mask. Supports
   MC or Sobol/QMC noise (budget-limited; consumed scalars-first). Per-sample cosine LR decay.
   **Pending migration to the time-stamped-observation API**: it still calls the removed
   `Observations.randomized` and reads removed `config.sigma_*` fields. Migrate by randomizing
   per observation (`obs.randomized(eps_...=...)`, sigmas live on the objects; append any new
   QMC draws — e.g. the now-available dhdt perturbation — AFTER the existing draw order).
   The same migration should thread the enthalpy scalars through its five mf/rf touchpoints
   (`InitScheme`, noise draws, `PriorMeans`, init loop, Adam block — gated on
   `config.smb_model` like `inverse.py`), again appending new draws after the existing order.
3. **sensitivity.py** — loads RTO samples, projects forward (default +100 yr), uses
   ∂ΔV/∂(physical param) for a Stein-style information-gain metric. Optimizes in *physical*
   space (via `simulate_physical`), not whitened. Works with the new API via `volumes`
   checkpoints, except one `config.sigma_s` read in the Stein denominators
   (→ `problem.get_observation("srf").sigma`); pass `record_states_at=[]` for projection
   runs to skip observation snapshots. Its flattened parameter vector hardcodes the
   five-parameter (bed, pbias, log_beta, log_mf, log_rf) layout — extending it to the
   enthalpy scalars shifts that index arithmetic.
4. **posterior.py** — empirical posterior covariance (SVD factorization) from RTO samples.

### Preprocessing pipeline

`preprocessing/make_all.py` runs per-variable `make_*.py` builders in dependency order
(DEM → flightlines/velocity/snowline/debris/insolation/climate, plus independent
temperature & precip anomalies, then a merge) and writes `domains/<name>/model_inputs/`.
`make_insolation.py` emits the direct-beam potential (Fourier modes) *and* the diffuse-sky
potential; `--diffuse-only` adds the latter to an existing product (36 horizon ray-traces
rather than the hour-by-hour direct loop), followed by `make_merged.py`.
Inputs come from the shared `common_data/` bundle plus the domain's `local_data/outline.kml`.
Each builder is also importable/runnable on its own.

## Conventions

- Domains live under `domains/{chugach,delta,denali,juneau,st_elias,wrangell}`; `denali` has
  no `model_inputs/` yet. The active domain is selected by editing the `DOMAIN` constant at
  the top of each driver script (not a CLI arg).
- `model_inputs/`, `common_data/`, and all raster/VTI/NetCDF outputs (`*.nc *.tif *.vti
  *.pvd ...`) are gitignored. Only configs, library code, and outlines are tracked.
- Optional inputs (snowline, debris, precip anomaly) are skipped gracefully when their file
  is absent — the corresponding loss term is dropped. Don't assume they exist.
