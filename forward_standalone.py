"""
Standalone forward run of the MAP solution: the calibrated hindcast,
extended into a projection. Ported from ../greenland-forecast/forward_standalone.py.

Runs the inverse problem's composite forward (enthalpy SMB -> glide ice
dynamics, driven by the calibrated bed, basal traction, precip bias,
temperature bias and enthalpy parameter fields) with no autograd, no priors,
no observations: plain cupy state, an explicit time loop, a VTI writer.

The calibrated fields come from the inverse checkpoint through ONE export
step (the only use of the glacier_inverse priors / GP maps / bed
conditioning), cached in {results}/physical_fields.nc. Everything after that
reads only GLIDE_inputs.nc, the debris file, the anomaly records and that file.

    python forward_standalone.py                # export if needed, then run
    python forward_standalone.py --export       # only refresh the export
    python forward_standalone.py --gcm MRI-ESM2-0 --scenario ssp126

Time design. Up to the inverse's horizon (max(t_end, latest observation
epoch)) the run REPLAYS the inverse's forward exactly: the same step
sequence (uniform config.dt grid snapped onto the observation epochs, which
the export stores), the same 'mean_anomaly' forcing, the same initial
thickness. From the horizon to T_END it steps at PROJECTION_DT.

Forcing past the anomaly record. Every calendar year y contributes a
temperature shift (K, added to the monthly climatology + tbias) and a precip
multiplier. On the record (y <= last record year) these are the inverse's
alpha_t2m * a(y) - base_anomaly and its index precip ratio. Past it:

  * GCM set (--gcm): CMIP6 delta-method fields (preprocessing/
    make_cmip6_deltas.py), interpolated bilinearly onto the grid per year,
        shift(y, month, x) = anchor_T + dT_gcm(y, month, x)
        mult (y, month, x) = anchor_P * p_ratio_gcm(y, month, x)
    with anchor_T / anchor_P the mean record shift / multiplier over the
    deltas' reference window, so the GCM's anomaly is measured from the
    calibrated forcing of that window, not from the bare climatology.
  * GCM None: the PLACEHOLDER projection_shift(y) / projection_precip(y),
    the mean shift of the last PROJECTION_REF_YEARS record years
    (+ PROJECTION_TREND K/yr), a committed-loss scenario.

Surface-elevation feedback. The inverse's SMB sees the static observed DEM.
With ELEVATION_FEEDBACK the projection adds FEEDBACK_LAPSE * (S_model -
S_ref) to t2m, S_ref the model surface at FEEDBACK_T_REF (default: the
inverse horizon, so the calibrated hindcast is untouched). Precip and the
avalanche operator keep the static DEM.

Outputs ({config.output_dir}/forward_standalone/):
  glacier_volumes.nc  volume (m^3) and area (m^2, H > ICE_H_MIN) per RGI
                      glacier (the gridded rgi_label footprint, fine grid)
                      at t_start and every step end; the comparison product
  forward_soln.nc     final state on the run level
(forward_{GCM}_{SCENARIO}/ with --gcm)
  vti/                frames from VTI_T_MIN, every VTI_EVERY years
"""
import argparse
import math
from pathlib import Path

import cupy as cp
import numpy as np
import pyproj
import xarray as xr

from glide.model import IceDynamics
from glide.field import Field, GridEntity
from glide.io import VTIWriter
from glare.avalanche import AvalancheOperator
from glare.enthalpy import EnthalpyModel, SECONDS_PER_YEAR, generate_temp_deviations

from glacier_inverse import load_config   # config only: physics constants and paths
from glacier_inverse.forward import _hermite_nodes, year_overlap_weights
from glacier_inverse.problem import _apply_solver_settings
from glacier_inverse.scheduling import build_step_sequence   # the inverse's step design

# ----------------------------------------------------------------- settings
DOMAIN = "domains/denali"
LEVEL = 0                         # multigrid level to run on (0 = finest)
CHECKPOINT = None                 # default: {config.output_dir}/level_0/torch_vars.p
OUT_DIR = None                    # default: {config.output_dir}/forward_standalone (forward_{GCM}_{SCENARIO} with a GCM)
T_END = 2100.0                    # end of the projection
PROJECTION_DT = 1.0               # step length after the inverse horizon (yr)
ICE_H_MIN = 10.0                  # m; "ice" in the per-glacier area
REPORT_GLACIERS = ("RGI2000-v7.0-G-01-04282",)   # printed every step (Kahiltna)
# --- projection climate: CMIP6 deltas for GCM (None = the placeholder below)
GCM = None
SCENARIO = "ssp126"
CMIP_DELTA_DIR = Path(__file__).resolve().parent / "common_data" / "climate" / "cmip6" / "deltas"
# --- projection climate PLACEHOLDER (see projection_shift / projection_precip)
PROJECTION_REF_YEARS = 10         # hold the mean shift of the last N record years ...
PROJECTION_TREND = 0.0            # ... plus this many K/yr after the record
# --- surface-elevation feedback on t2m during the projection
ELEVATION_FEEDBACK = True
FEEDBACK_LAPSE = -6.0e-3          # K/m (scalar or 12 monthly values). Placeholder:
                                  # regress the along-surface gradient of the CARRA2
                                  # climatology over the ice, as greenland did
FEEDBACK_T_REF = None             # S_ref = model surface at this time; None = inverse horizon
# --- VTI frames (glide VTIWriter, LZ4, rounded, masked off-ice)
VTI_T_MIN = 2000.0                # frames only for step ends >= this year (None = all)
VTI_EVERY = 5.0                   # ... and only at multiples of this many years (None = every step)
VTI_COMPRESSOR = "lz4"
VTI_PRECISION = {"H": 0.01, "srf": 0.01, "dhdt": 1e-3, "smb": 1e-3, "U": 0.01, "U_s": 0.01,
                 "U_b": 0.01, "dT_fb": 1e-3, "bed": 0.01, "beta": 1e-3}
VTI_MASKED_FIELDS = ("U", "U_s", "U_b", "smb", "dhdt")

config = load_config(DOMAIN)

CHECKPOINT = CHECKPOINT or f"{config.output_dir}/level_0/torch_vars.p"


def default_out_dir() -> Path:
    name = "forward_standalone" if GCM is None else f"forward_{GCM}_{SCENARIO}"
    return Path(f"{config.output_dir}/{name}")


PHYSICAL_PATH = Path(config.output_dir) / "physical_fields.nc"
INPUTS = Path(config.base_dir) / "model_inputs"


# ----------------------------------------------------------- one-time export
def export_physical_fields(checkpoint: str, out_path: Path) -> None:
    """Map the whitened MAP checkpoint to physical fields on the (cropped)
    fine grid and write them to NetCDF, with the observation epochs the
    inverse snapped its steps onto. The only place the inverse library
    (priors, GP maps, bed conditioning) is used. Checks the exported bed /
    beta against the MAP's own inverse_soln.nc: the domain config may have
    moved since the MAP was written, and the whitened->physical map then
    no longer reproduces it."""
    import torch
    from glacier_inverse import GlacierProblem
    from glacier_inverse.io import load_whitened_params_into

    problem = GlacierProblem(config)
    load_whitened_params_into(problem.params, checkpoint, priors=problem.priors)
    with torch.no_grad():
        ph = problem.physical_from(problem.params)
        log_pbias = problem.effective_log_pbias(ph)
        H0 = problem._initial_thickness_from_geometry(ph.bed)
    gd = problem.gridded_data
    f32 = lambda t: t.detach().to(torch.float32).cpu().numpy()
    fields = {
        "bed": (f32(ph.bed), "calibrated bed (conditioned)", "m"),
        "beta": (f32(torch.exp(ph.log_beta)), "basal traction coefficient", "head per (m/yr)^m"),
        "log_pbias": (f32(log_pbias), "log precip multiplier", "-"),
        "tbias": (f32(ph.tbias) if ph.tbias is not None else np.zeros(ph.bed.shape, np.float32),
                  "additive air temperature bias", "K"),
        "H_atm": (f32(torch.exp(ph.log_H_atm)), "turbulent exchange coefficient", "W m-2 K-1"),
        "f_clear": (f32(torch.sigmoid(ph.logit_cloud)), "clear-sky fraction", "-"),
        "H_init": (f32(H0), "initial thickness from the observed surface and the calibrated bed", "m"),
    }
    ds = xr.Dataset(coords={"y": gd.y, "x": gd.x})
    ds["spatial_ref"] = gd["spatial_ref"]
    for k, (a, name, units) in fields.items():
        ds[k] = xr.DataArray(a, dims=("y", "x"), attrs={"long_name": name, "units": units})
    ds.attrs.update(checkpoint=str(checkpoint), smb_model=config.smb_model,
                    enthalpy_seed=int(config.enthalpy_seed),
                    required_times=np.asarray(problem.required_times, dtype=np.float64))

    soln = Path(checkpoint).with_name("inverse_soln.nc")
    if soln.exists():
        s = xr.load_dataset(soln)
        ice = gd.rgi_mask.values > 0
        for k in ("bed", "beta"):
            d = ds[k].values - s[k].values
            print(f"export check vs {soln.name}: {k} rms diff on ice {np.sqrt(np.mean(d[ice] ** 2)):.3g}, "
                  f"max {np.abs(d[ice]).max():.3g}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}")
    del problem
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ helpers
def crop_to_factor(ds: xr.Dataset, factor: int) -> xr.Dataset:
    """Same centred crop GlacierProblem applies so ny, nx divide 2^n_levels."""
    ny0, nx0 = ds.sizes["y"], ds.sizes["x"]
    ny, nx = (ny0 // factor) * factor, (nx0 // factor) * factor
    y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
    return ds.isel(y=slice(y0, y0 + ny), x=slice(x0, x0 + nx))


def restrict(a: cp.ndarray, n_times: int) -> cp.ndarray:
    """2x2 block mean, `n_times` times (the inverse's avg-pool restriction)."""
    for _ in range(n_times):
        ny, nx = a.shape
        a = a.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    return a


def prolong(a: cp.ndarray, n_times: int) -> cp.ndarray:
    """Injection back onto the fine grid."""
    f = 2 ** n_times
    return cp.repeat(cp.repeat(a, f, axis=0), f, axis=1) if f > 1 else a


def load_eager(path) -> xr.Dataset:
    # load and close: never hold a lazy handle on the inputs (CLAUDE.md, HDF5)
    with xr.open_dataset(path) as f:
        return f.load()


# ------------------------------------------------- projection climate (PLACEHOLDER)
def projection_shift(year: int, record_shift: dict) -> float:
    """Temperature shift (K) added to the monthly climatology (+ tbias) for a
    calendar year past the anomaly record. `record_shift` maps record years
    to the inverse's shift alpha_t2m * a(y) - base_anomaly, so a scenario
    delta should be expressed relative to the same reference.

    PLACEHOLDER: the mean shift over the last PROJECTION_REF_YEARS record
    years, plus PROJECTION_TREND K/yr after the record."""
    y_max = max(record_shift)
    ref = [record_shift[y] for y in range(y_max - PROJECTION_REF_YEARS + 1, y_max + 1)]
    return float(np.mean(ref)) + PROJECTION_TREND * (year - y_max)


def projection_precip(year: int) -> float:
    """Precip multiplier for a year past the record. PLACEHOLDER: 1."""
    return 1.0


class GcmDeltas:
    """CMIP6 delta-method forcing (preprocessing/make_cmip6_deltas.py) on the
    GCM grid, interpolated bilinearly onto the (cropped) fine grid per year
    and anchored to the record over the deltas' reference window."""

    def __init__(self, path: Path, gd: xr.Dataset, record_shift: dict, record_mult):
        if not path.exists():
            raise FileNotFoundError(f"{path}: run preprocessing/make_cmip6_deltas.py first")
        self.ds = xr.load_dataset(path)
        self.path = path
        r0, r1 = (int(v) for v in np.atleast_1d(self.ds.attrs["ref_years"]))
        ref = [y for y in range(r0, r1 + 1) if y in record_shift]
        if len(ref) < r1 - r0 + 1:
            raise ValueError(f"reference window {r0}-{r1} not covered by the anomaly record")
        self.anchor_T = float(np.mean([record_shift[y] for y in ref]))
        self.anchor_P = float(np.mean([record_mult(y) for y in ref]))
        self.ref = (r0, r1)
        # fractional GCM-grid indices of every fine cell
        X, Y = np.meshgrid(gd.x.values, gd.y.values)
        lon, lat = pyproj.Transformer.from_crs(pyproj.CRS(gd.spatial_ref.crs_wkt), 4326,
                                               always_xy=True).transform(X, Y)
        lon = lon % 360.0
        glat, glon = self.ds.lat.values, self.ds.lon.values
        if not (glat[0] < lat.min() and lat.max() < glat[-1] and glon[0] < lon.min() and lon.max() < glon[-1]):
            raise ValueError(f"domain outside the delta box of {path.name}")
        self.coords = cp.asarray(np.stack([np.interp(lat, glat, np.arange(len(glat))),
                                           np.interp(lon, glon, np.arange(len(glon)))]), dtype=cp.float32)
        self.cache = {}

    def _interp(self, a: np.ndarray) -> cp.ndarray:
        from cupyx.scipy.ndimage import map_coordinates
        a = cp.asarray(a, dtype=cp.float32)
        return cp.stack([map_coordinates(a[m], self.coords, order=1, mode="nearest") for m in range(12)])

    def year(self, y: int):
        if y not in self.cache:
            if y not in self.ds.year.values:
                raise ValueError(f"{self.path.name} has no year {y}")
            d = self.ds.sel(year=y)
            self.cache = {k: v for k, v in self.cache.items() if k >= y - 1}   # keep the window small
            self.cache[y] = (self.anchor_T + self._interp(d.dT.values),
                             self.anchor_P * self._interp(d.p_ratio.values))
        return self.cache[y]

    def describe(self) -> str:
        a = self.ds.attrs
        return (f"CMIP6 {a['source_id']} {a['member_id']} {a['experiment_id']} deltas vs "
                f"{self.ref[0]}-{self.ref[1]}, anchored at shift {self.anchor_T:+.2f} K, precip x{self.anchor_P:.3f}")


# -------------------------------------------------------------------- setup
class Run:
    """Everything the time loop needs; built by setup(), stepped by run()."""


def setup(level: int = None, out_dir=None) -> Run:
    if config.smb_model != "enthalpy":
        raise NotImplementedError("this driver mirrors the enthalpy SMB path only")
    if config.yearly_climate_filename is not None or config.climatology_only:
        raise NotImplementedError("yearly reanalysis forcing / climatology_only: see the greenland driver")
    if config.thermal is not None:
        raise NotImplementedError("thermal coupling: see the greenland driver")
    if config.ocean_forcing.enabled:
        raise NotImplementedError("ocean forcing / front pin (tidewater domains): see the greenland driver")
    if config.anomaly_integration != "mean_anomaly":
        raise NotImplementedError(f"anomaly_integration={config.anomaly_integration!r}: "
                                  "this driver mirrors 'mean_anomaly' only")

    ctx = Run()
    level = LEVEL if level is None else int(level)
    out_dir = Path(out_dir or OUT_DIR or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx.level, ctx.out_dir = level, out_dir
    factor = 2 ** config.n_levels

    ### Load data: gridded inputs (cropped exactly as the inverse), the
    ### calibrated fields, the anomaly records
    gd = crop_to_factor(load_eager(INPUTS / config.gridded_filename), factor)
    phys = xr.load_dataset(PHYSICAL_PATH)
    assert phys.sizes["y"] == gd.sizes["y"] and phys.sizes["x"] == gd.sizes["x"]
    ctx.required_times = [float(t) for t in np.atleast_1d(phys.attrs.get("required_times", []))]

    anom = xr.load_dataset(INPUTS / config.anomaly_filename)
    t_anom = {int(y): float(v) for y, v in zip(anom.time.values, anom.temp_anomaly.values)}
    year_max = max(t_anom)
    base_anomaly = 0.0 if config.base_anomaly_year is None else t_anom[int(config.base_anomaly_year)]
    # per-year shift on the record (forward._term_forcing: alpha * a - base)
    record_shift = {y: config.alpha_t2m * a - base_anomaly for y, a in t_anom.items()}

    # index precip multiplier, as forward.simulate applies it
    p_anom, base_precip = None, None
    alpha_p = float(config.alpha_precip)
    pa_path = INPUTS / config.precip_anomaly_filename
    if pa_path.exists() and alpha_p != 0.0:
        pa = xr.load_dataset(pa_path)
        p_anom = {int(y): float(v) for y, v in zip(pa.time.values, pa.precip_anomaly.values)}
        base_precip = p_anom[int(config.base_precip_year)]
    p_year_max = max(p_anom) if p_anom is not None else None

    def record_mult(y: int) -> float:
        if p_anom is None:
            return 1.0
        return 1.0 + alpha_p * (p_anom[min(y, p_year_max)] / base_precip - 1.0)

    gcm = None if GCM is None else GcmDeltas(CMIP_DELTA_DIR / f"{GCM}_{SCENARIO}.nc", gd,
                                             record_shift, record_mult)
    ctx.climate = gcm.describe() if gcm is not None else (
        f"placeholder: mean shift of last {PROJECTION_REF_YEARS} record years "
        f"+ {PROJECTION_TREND:g} K/yr; precip x1")
    print(f"projection climate past {year_max}: {ctx.climate}")

    def year_forcing(y: int):
        """(temperature shift K, precip multiplier) of calendar year y:
        floats, or (12, ny, nx) cupy fields from the GCM deltas."""
        if y <= year_max:
            return record_shift[y], record_mult(y)
        if gcm is not None:
            return gcm.year(y)
        return projection_shift(y, record_shift), projection_precip(y)

    # interannual-variance quadrature (forward._expand_interannual)
    if config.interannual_sigma:
        x, w = _hermite_nodes(int(config.interannual_nodes))
        nodes = (tuple(config.interannual_sigma * xi for xi in x), w)
    else:
        nodes = ((0.0,), (1.0,))

    ny, nx = gd.sizes["y"], gd.sizes["x"]
    dx = float(gd.x[1] - gd.x[0])
    x0, y0 = float(gd.x[0]), float(gd.y[0])
    crs = pyproj.CRS(gd.spatial_ref.crs_wkt)
    f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)

    ### Ice dynamics (GlacierProblem._build_ice_dynamics)
    model = IceDynamics(n_levels=config.n_levels, ny=ny, nx=nx, dx=dx,
                        x0=x0, y0=y0, crs=crs, stress_scheme=config.stress_scheme)
    mg = model.mg
    bed = f32(phys.bed.values)
    mg.geometry.thklim.set(config.thklim)
    mg.geometry.bed.set(bed)
    mg.geometry.depth.set(-bed)                   # signed: negative on dry land
    mg.geometry.sigmoid_c.set(config.sigmoid_c)
    mg.rheology.B.set(config.B_rate)
    mg.rheology.eps_reg.set(config.eps_reg)
    mg.rheology.n.set(float(config.n_glen))
    mg.rheology.H_reg.set(float(config.H_reg))
    mg.sliding.m.set(config.sliding_m)
    mg.sliding.u_reg.set(config.u_reg)
    mg.sliding.water_drag.set(config.water_drag)
    mg.sliding.u0.set(float(getattr(config, "sliding_u0", 0.0)))
    if config.sliding_N_scale_H:
        mg.sliding.N_scale_H.set(float(config.sliding_N_scale_H))
        mg.sliding.N_floor_H.set(float(config.sliding_N_floor_H))
    mg.calving.timescale.set(config.calving_timescale)
    mg.calving.q.set(config.calving_q)
    mg.calving.h0.set(config.calving_h0)
    mg.calving.H_c.set(config.calving_H_c)
    _apply_solver_settings(model.forward_solver, config.forward_solver)
    # beta_max caps the FINE field before restriction (simulate_physical's order)
    log_beta = cp.log(f32(phys.beta.values))
    if config.beta_max is not None:
        log_beta = cp.minimum(log_beta, cp.float32(math.log(float(config.beta_max))))

    ### SMB: glare's enthalpy model on the fine grid, calibrated parameters
    ### (GlacierProblem._build_enthalpy_smb_model + simulate_physical)
    spy = SECONDS_PER_YEAR
    smb_model = EnthalpyModel(ny=ny, nx=nx, nt=12, dx=dx, dt=1.0 / 12, x0=x0, y0=y0, crs=crs,
                              n_substeps=config.enthalpy_n_substeps,
                              materialize_state=config.enthalpy_materialize_state)
    g = smb_model.grid
    t2m = f32(gd.monthly_t2m.values)                                   # (12, ny, nx) degC
    tbias = f32(phys.tbias.values)
    g.geometry.srf.set(f32(gd.elevation.values))
    g.radiation.insol_mean.set(f32(gd.monthly_solar_potential_mean.values))
    g.radiation.insol_dif.set(f32(gd.monthly_diffuse_potential.values)
                              if "monthly_diffuse_potential" in gd else 0.0)
    g.geometry.t_base.set(cp.minimum(t2m.mean(axis=0), 0.0))         # not anomaly-shifted
    debris_path = INPUTS / config.debris_filename
    if debris_path.exists():
        deb = crop_to_factor(load_eager(debris_path), factor)
        g.geometry.debris.set(1.0 - config.debris_factor * f32(deb.debris_fraction.values))
    else:
        g.geometry.debris.set(1.0)
    f_clear = f32(phys.f_clear.values)
    g.thermodynamics.H_atm.set(f32(phys.H_atm.values) * spy)
    g.thermodynamics.H_base0.set(cp.float32(config.H_base0 * spy))
    g.radiation.q_sw_bulk.set(cp.float32(config.q_sw_bulk * spy))
    g.radiation.q_sw_insol.set(f_clear * config.q_sw_clear * spy)
    g.radiation.q_sw_dif.set((f_clear * config.k_diffuse_clear
                              + (1.0 - f_clear) * config.k_diffuse_cloud) * config.q_sw_clear * spy)
    g.radiation.q_lw0.set(cp.float32(config.q_lw0 * spy))
    g.radiation.albedo_snow.set(cp.float32(config.albedo_snow))
    g.radiation.albedo_ice.set(cp.float32(config.albedo_ice))
    g.radiation.M_albedo.set(cp.float32(config.M_albedo))

    precip = f32(gd.monthly_precip.values) * cp.exp(f32(phys.log_pbias.values))
    if config.use_avalanche_model:
        op = AvalancheOperator(g, s_crit=config.avalanche_s_crit, w_trans=config.avalanche_w_trans,
                               p=config.avalanche_p, K=config.avalanche_K)
        if config.avalanche_hoisted:
            # once, up front: R(c P) = c R(P) for the scalar per-step multiplier
            eff = cp.empty_like(precip)
            op.forward(cp.ascontiguousarray(precip), eff)
            precip = eff
        else:
            smb_model.avalanche = op
    # the inverse's fixed weather realization (same seed -> same forward)
    sigma_t2m = float(cp.asnumpy(g.temperature.sigma_t2m.value))
    temp_dev = generate_temp_deviations(12, config.enthalpy_n_substeps, sigma_t2m,
                                        np.random.default_rng(config.enthalpy_seed))
    domain_mask = cp.asarray(gd.domain_mask.values, dtype=bool)

    ctx.t2m_offset = None      # (12, ny, nx) elevation-feedback increment, set by run()

    def smb_eval(shift: float, mult: float) -> cp.ndarray:
        off = 0.0 if ctx.t2m_offset is None else ctx.t2m_offset
        g.temperature.t2m.set(t2m + shift + tbias + off)
        g.precipitation.precip.set(precip if isinstance(mult, float) and mult == 1.0 else precip * mult)
        smb_model.forward(temp_deviations=temp_dev)
        return g.state.smb.data.mean(axis=0)

    def compute_smb(t_prev: float, t_next: float) -> cp.ndarray:
        """Annual-mean SMB on the fine grid for the step (t_prev, t_next] as
        forward.simulate builds it under 'mean_anomaly': ONE evaluation at
        the overlap-weighted mean shift / precip multiplier of the step's
        years (+ the interannual quadrature); -10 m/yr outside the domain."""
        weights = year_overlap_weights(t_prev, t_next)
        forc = [(year_forcing(y), w) for y, w in weights]
        shift = sum(s * w for (s, _), w in forc)
        mult = sum(m * w for (_, m), w in forc)
        ctx.step_shift = float(shift.mean()) if hasattr(shift, "mean") else float(shift)
        smb = cp.zeros_like(t2m[0])
        for xi, wi in zip(*nodes):
            smb += wi * smb_eval(shift + xi, mult)
        smb[~domain_mask] = -10.0
        return smb

    ### Initial state on the run level: thickness from the observed surface
    ### and the calibrated bed, restricted like the inverse
    model.set_top_level(level)
    H = restrict(f32(phys.H_init.values), level)
    mg.state.H.set(H, start_level=level)
    mg.state.H_prev.set(H, start_level=level)
    mg.geometry.bed.set(restrict(bed, level), start_level=level)
    mg.sliding.beta.set(cp.exp(restrict(log_beta, level)), start_level=level)
    for f in (mg.state.u, mg.state.v, mg.state.ud, mg.state.vd, mg.state.mask):
        f.set(0.0, start_level=level)

    ### Per-glacier bookkeeping on the fine grid (rgi_label == k <-> rgi_id[k];
    ### unlabelled cells go to an extra bucket, index n_glac)
    rgi_ids = [str(s) for s in gd.rgi_id.values]
    n_glac = len(rgi_ids)
    lab = gd.rgi_label.values
    lab = np.where(np.isfinite(lab), lab, n_glac).astype(np.int64)
    ctx.labels = cp.asarray(lab.ravel())
    ctx.rgi_ids, ctx.n_glac, ctx.cell_area = rgi_ids, n_glac, dx * dx
    ctx.report = [(r, rgi_ids.index(r)) for r in REPORT_GLACIERS if r in rgi_ids]

    ### Writers on the run level
    lvl = mg[level]
    n_glen = float(config.n_glen)

    def cell_field(name, units, long_name):
        return Field(data=cp.zeros((lvl.ny, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.CELL,
                     dx=lvl.dx, grid=lvl, name=name, units=units, attrs={"long_name": long_name})

    srf = cell_field("srf", "m", "Surface elevation")
    dhdt = cell_field("dhdt", "m a^{-1}", "Thickness change rate")
    dT_fb = cell_field("dT_fb", "K", "Elevation-feedback temperature increment (annual mean)")
    u_s = Field(data=cp.zeros((lvl.ny, lvl.nx + 1), dtype=cp.float32), grid_entity=GridEntity.VERTICAL_FACET,
                dx=lvl.dx, grid=lvl, name="u_s", units="m a^{-1}", attrs={"long_name": "Surface velocity (x)"})
    v_s = Field(data=cp.zeros((lvl.ny + 1, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.HORIZONTAL_FACET,
                dx=lvl.dx, grid=lvl, name="v_s", units="m a^{-1}", attrs={"long_name": "Surface velocity (y)"})
    u_b = Field(data=cp.zeros((lvl.ny, lvl.nx + 1), dtype=cp.float32), grid_entity=GridEntity.VERTICAL_FACET,
                dx=lvl.dx, grid=lvl, name="u_b", units="m a^{-1}", attrs={"long_name": "Basal velocity (x)"})
    v_b = Field(data=cp.zeros((lvl.ny + 1, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.HORIZONTAL_FACET,
                dx=lvl.dx, grid=lvl, name="v_b", units="m a^{-1}", attrs={"long_name": "Basal velocity (y)"})
    rho_ratio = config.rho_ice / config.rho_water

    def update_derived(dt_step):
        # MOLHO: surface velocity u + ud/(n+1), basal velocity u - ud
        u_s.data[:, :] = lvl.state.u.data + lvl.state.ud.data / (n_glen + 1.0)
        v_s.data[:, :] = lvl.state.v.data + lvl.state.vd.data / (n_glen + 1.0)
        u_b.data[:, :] = lvl.state.u.data - lvl.state.ud.data
        v_b.data[:, :] = lvl.state.v.data - lvl.state.vd.data
        srf.data[:, :] = cp.maximum(lvl.state.H.data + lvl.geometry.bed.data,
                                    (1.0 - rho_ratio) * lvl.state.H.data)
        dhdt.data[:, :] = (lvl.state.H.data - lvl.state.H_prev.data) / dt_step

    vti_dir = out_dir / "vti"
    vti_dir.mkdir(parents=True, exist_ok=True)
    vti_writer = VTIWriter(str(vti_dir), base=config.vti_base_name, dx=lvl.dx,
                           compressor=VTI_COMPRESSOR, precision=VTI_PRECISION,
                           mask_field="mask", masked_fields=VTI_MASKED_FIELDS,
                           static_fields={"bed": lvl.geometry.bed, "beta": lvl.sliding.beta},
                           dynamic_fields={"H": lvl.state.H, "srf": srf, "dhdt": dhdt,
                                           "U": [lvl.state.u, lvl.state.v], "U_s": [u_s, v_s],
                                           "U_b": [u_b, v_b], "smb": lvl.forcing.smb,
                                           "mask": lvl.state.mask, "dT_fb": dT_fb})
    vti_writer.initialize(lvl)

    ctx.model, ctx.mg, ctx.lvl, ctx.gd, ctx.crs = model, mg, lvl, gd, crs
    ctx.compute_smb, ctx.update_derived = compute_smb, update_derived
    ctx.vti_writer, ctx.vti_dir = vti_writer, vti_dir
    ctx.srf, ctx.u_s, ctx.v_s, ctx.dhdt, ctx.dT_fb = srf, u_s, v_s, dhdt, dT_fb
    ctx.year_max = year_max
    return ctx


def dynamics_step(ctx: Run, t_prev: float, dt_step: float) -> None:
    """One glide step on the run level with the state handed over exactly as
    the inverse's GlideStep does: H_prev <- H, H <- H_prev, bed / beta reset
    from the run level down so every coarse level of the hierarchy is fresh."""
    mg, lvl, level = ctx.mg, ctx.lvl, ctx.level
    H_prev = lvl.state.H.data.copy()
    mg.state.H_prev.set(H_prev, start_level=level)
    mg.state.H.set(H_prev, start_level=level)
    mg.geometry.bed.set(lvl.geometry.bed.data.copy(), start_level=level)
    mg.sliding.beta.set(lvl.sliding.beta.data.copy(), start_level=level)
    ctx.model.forward(cp.float32(t_prev), cp.float32(dt_step), update_geometry=False)


def model_surface(ctx: Run) -> cp.ndarray:
    H = ctx.lvl.state.H.data
    return cp.maximum(H + ctx.lvl.geometry.bed.data, (1.0 - config.rho_ice / config.rho_water) * H)


def glacier_totals(ctx: Run):
    """Volume (m^3) and area with H > ICE_H_MIN (m^2) per RGI label, fine grid."""
    H = prolong(ctx.lvl.state.H.data, ctx.level).ravel()
    n = ctx.n_glac + 1
    V = cp.bincount(ctx.labels, weights=H, minlength=n) * ctx.cell_area
    A = cp.bincount(ctx.labels, weights=(H > ICE_H_MIN).astype(cp.float32), minlength=n) * ctx.cell_area
    return cp.asnumpy(V), cp.asnumpy(A)


def step_sequence(ctx: Run):
    """The inverse's steps up to its horizon, then PROJECTION_DT steps to T_END."""
    seq = build_step_sequence(t_start=float(config.t_start), t_end=float(config.t_end),
                              dt_max=float(config.dt), required_times=ctx.required_times,
                              dt_schedule=config.dt_schedule)
    ends = [t for t, _ in seq]
    horizon = ends[-1]
    if horizon > ctx.year_max + 1:
        print(f"WARNING: the inverse horizon {horizon:g} runs past the anomaly record "
              f"({ctx.year_max}); the replay there uses the projection placeholder, "
              "the inverse clamped to the last record year")
    if T_END > horizon:
        n = math.ceil((T_END - horizon) / PROJECTION_DT - 1e-9)
        ends += [horizon + (T_END - horizon) * (k + 1) / n for k in range(n)]
    return list(zip([float(config.t_start)] + ends[:-1], ends)), horizon


def run(ctx: Run) -> None:
    mg, lvl, level, out_dir = ctx.mg, ctx.lvl, ctx.level, ctx.out_dir
    steps, horizon = step_sequence(ctx)
    t_ref = horizon if FEEDBACK_T_REF is None else float(FEEDBACK_T_REF)
    lapse = cp.asarray(np.broadcast_to(np.atleast_1d(np.asarray(FEEDBACK_LAPSE, np.float32)), (12,)).copy())
    S_ref = None
    print(f"{len(steps)} steps {config.t_start:g} -> {T_END:g}; inverse horizon {horizon:g}, "
          f"anomaly record to {ctx.year_max}; elevation feedback "
          f"{'from ' + format(t_ref, 'g') if ELEVATION_FEEDBACK else 'off'}", flush=True)

    times, vols, areas, shifts = [float(config.t_start)], [], [], [np.nan]
    V, A = glacier_totals(ctx)
    vols.append(V); areas.append(A)
    for t_prev, t_next in steps:
        dt_step = t_next - t_prev
        if ELEVATION_FEEDBACK and t_prev >= t_ref - 1e-6:
            if S_ref is None:
                S_ref = model_surface(ctx).copy()
                print(f"  elevation feedback: reference surface captured at t={t_prev:g}", flush=True)
            dS = model_surface(ctx) - S_ref
            ctx.t2m_offset = lapse[:, None, None] * prolong(dS, level)[None]
            ctx.dT_fb.data[:, :] = float(lapse.mean()) * dS
        mg.forcing.smb.set(restrict(ctx.compute_smb(t_prev, t_next), level), start_level=level)
        dynamics_step(ctx, t_prev, dt_step)
        ctx.update_derived(dt_step)

        V, A = glacier_totals(ctx)
        times.append(t_next); vols.append(V); areas.append(A); shifts.append(ctx.step_shift)
        rep = "  ".join(f"{r[-5:]} {V[k] / 1e9:.2f} km3" for r, k in ctx.report)
        print(f"t {t_prev:.2f} -> {t_next:.2f} (dt {dt_step:.2f}): shift {ctx.step_shift:+.2f} K, "
              f"V_total {V[:-1].sum() / 1e9:.1f} km3  {rep}", flush=True)

        if ((VTI_T_MIN is None or t_next >= VTI_T_MIN - 1e-6)
                and (VTI_EVERY is None or abs(t_next / VTI_EVERY - round(t_next / VTI_EVERY)) < 1e-6)):
            ctx.vti_writer.append(lvl, time=float(t_next))
            ctx.vti_writer.write_pvd()

    ### Per-glacier time series (the comparison product)
    vol = xr.Dataset(
        {"volume": (("time", "glacier"), np.stack(vols)[:, :-1], {"units": "m3"}),
         "area": (("time", "glacier"), np.stack(areas)[:, :-1],
                  {"units": "m2", "long_name": f"area with H > {ICE_H_MIN:g} m"}),
         "volume_unlabelled": ("time", np.stack(vols)[:, -1], {"units": "m3",
                               "long_name": "ice outside every RGI footprint"}),
         "temperature_shift": ("time", np.asarray(shifts),
                               {"units": "K", "long_name": "step-mean t2m shift (excl. tbias, feedback)"})},
        coords={"time": np.asarray(times), "rgi_id": ("glacier", ctx.rgi_ids)})
    vol.attrs.update(domain=DOMAIN, level=level, checkpoint=str(CHECKPOINT), inverse_horizon=horizon,
                     anomaly_record_end=ctx.year_max, elevation_feedback=int(ELEVATION_FEEDBACK),
                     feedback_t_ref=t_ref, feedback_lapse=repr(FEEDBACK_LAPSE),
                     projection_climate=ctx.climate, gcm=GCM or "", scenario=SCENARIO if GCM else "",
                     note="volume over the gridded rgi_label footprint (RGI v7), fine grid; "
                          "time = step ends (inverse step design to the horizon)")
    vol.to_netcdf(out_dir / "glacier_volumes.nc")

    ### Final state (cell-centred fields on the run level)
    gd = ctx.gd
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)
    yc = restrict(f32(np.broadcast_to(gd.y.values[:, None], (ny, nx))), level)[:, 0]
    xc = restrict(f32(np.broadcast_to(gd.x.values[None, :], (ny, nx))), level)[0, :]
    out = xr.Dataset(coords={"y": cp.asnumpy(yc), "x": cp.asnumpy(xc)})
    for name, arr in [("H", lvl.state.H.data), ("srf", ctx.srf.data), ("dhdt", ctx.dhdt.data),
                      ("bed", lvl.geometry.bed.data), ("beta", lvl.sliding.beta.data),
                      ("smb", lvl.forcing.smb.data), ("mask", lvl.state.mask.data),
                      ("dT_fb", ctx.dT_fb.data),
                      ("u_s", 0.5 * (ctx.u_s.data[:, 1:] + ctx.u_s.data[:, :-1])),
                      ("v_s", 0.5 * (ctx.v_s.data[1:, :] + ctx.v_s.data[:-1, :]))]:
        out[name] = xr.DataArray(cp.asnumpy(arr), dims=("y", "x"))
    out.attrs.update(level=level, t_start=float(config.t_start), t_end=float(T_END),
                     inverse_horizon=horizon, n_steps=len(steps), checkpoint=str(CHECKPOINT),
                     crs_wkt=ctx.crs.to_wkt())
    out.to_netcdf(out_dir / "forward_soln.nc")
    print(f"wrote {out_dir / 'glacier_volumes.nc'}, {out_dir / 'forward_soln.nc'}; VTI series in {ctx.vti_dir}")


def main(export_only: bool = False) -> None:
    # re-export when the checkpoint is newer than the cached fields: an
    # inversion rerun into the same results_subdir otherwise replays the
    # PREVIOUS state without a word
    stale = (PHYSICAL_PATH.exists() and Path(CHECKPOINT).exists()
             and Path(CHECKPOINT).stat().st_mtime > PHYSICAL_PATH.stat().st_mtime)
    if stale:
        print(f"{PHYSICAL_PATH.name} is older than {CHECKPOINT}: re-exporting")
    if export_only or stale or not PHYSICAL_PATH.exists():
        export_physical_fields(CHECKPOINT, PHYSICAL_PATH)
        if export_only:
            return
    run(setup())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="only (re)write physical_fields.nc")
    ap.add_argument("--gcm", default=None, help="CMIP6 source_id with a deltas file (default: placeholder climate)")
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    GCM, SCENARIO = args.gcm, args.scenario
    OUT_DIR = Path(args.out_dir) if args.out_dir else None
    main(export_only=args.export)
