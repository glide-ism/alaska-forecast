"""
Delta domain configuration.

All numerical knobs that the four scripts share live here, so the physical
model is identical across inverse / rto / posterior / sensitivity by
construction. Per-task knobs (max iterations, output paths, warm-start path)
stay in the driver scripts.
"""

from pathlib import Path

from glacier_inverse.config import (
    BedConditioningConfig, FingerprintNuisance, GlacierConfig, MaternNoise,
    PriorHyperparams, Schedule,
)
from glacier_inverse.observations import (
    BedSpec, BedSlopeSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)
import numpy as np

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="denali",
    results_subdir="inverse_discrepency",
    smb_model = "enthalpy",
    # Constant (non-albedo-scaled) surface-flux offset: a-priori interior sky
    # longwave deficit + evaporative cooling. With the offset explicit, H_atm is
    # the pure dT slope, so its prior median moves to the first-principles
    # sensible+latent+longwave value and the prior is widened to honest ignorance.
    q_lw0=-40.0,
    mu_H_atm=15.0,
    # Legacy scalar sigma (checkpoint conversion only); the live pointwise
    # prior std is h_atm_prior.sigma below — kept at the same 0.01 pin.
    #sigma_log_H_atm=0.01,
    # H_atm / clear-sky fraction are (ny,nx) GP fields; sigma preserves the
    # scalar-era pointwise std (H_atm effectively pinned at the median),
    # l = the synoptic/orographic decorrelation scale.
    h_atm_prior=PriorHyperparams(sigma=0.2, l=80000.0, nu=1),
    cloud_prior=PriorHyperparams(sigma=0.25, l=80000.0, nu=1),
    # Rank-few model-error marginalization along the measured H_atm/f
    # sensitivity fingerprints (leading prior modes, whitened units): the
    # srf/dhdt data may at most double the prior precision on these modes
    # (s = 1 -> info floor 1/s^2 = one prior's worth), instead of the
    # measured 10^4-10^6x pinning. The fitted c-hat (printed at each
    # refresh) is the amount of each parameter's pattern attributed to
    # model error, in prior-std units. refresh=0: re-measure fingerprints
    # at each level start only.
    fingerprint_nuisance=FingerprintNuisance(
        params=("log_H_atm", "logit_cloud"), s=(1.0, 1.0),
        n_modes=4, refresh=0),
    # Shortwave: the inverted scalar f is the CLEAR-SKY fraction (1 - cloud
    # fraction); direct = f*S0*I (S0 = q_sw_clear = 1361, tau^airmass lives in
    # the direct potential I) and diffuse = (f*k_clr + (1-f)*k_cld)*S0*I_dif from
    # the same f, with I_dif = monthly_diffuse_potential (sky-view x cos zenith).
    # Interior-Alaska summer cloud fraction ~0.65 -> f ~ 0.35; at that median
    # June global over the RGI is ~190 W m-2 with a diffuse fraction ~0.55.
    # NB: checkpoints from the direct-only era hold z_logit_cloud ~ +6 (f = 0.87
    # under the old 0.6 prior) -- zero it on warm start rather than carry it.
    q_sw_clear=1361.0,
    mu_cloud_factor=0.35,
    sigma_logit_cloud=0.25,
    k_diffuse_clear=0.10,
    k_diffuse_cloud=0.30,
    anomaly_integration="mean_anomaly",
    stress_scheme='molho',
    grad_start_time=1712,
    #t_start=1712,
    observations=(
        # discrepancy: marginalized Kennedy-O'Hagan model-error component at
        # the synoptic scale — the surface term's information about the
        # domain/patch-scale LEVEL of the model-vs-DEM misfit is capped at
        # ~one observation of error sigma_D per l_D patch (the k->0 tail of
        # the 1-km Matern was extrapolation: a single-domain variogram has
        # n~1 samples at 80 km). Fine-scale weight (what constrains bed/beta)
        # is untouched. This is what stops the smooth SMB fields being pinned
        # through the domain-mean surface channel.
        SurfaceSpec(noise=MaternNoise(sigma=12.0, l=1000.0, nu=0.5, nugget=10.0,
                                      discrepancy=MaternNoise(sigma=20.0, l=80000.0, nu=1.0)),
                    weight=1.0, nu=3),
        VelocitySpec(noise=MaternNoise(sigma=12.0, l=3000.0, nugget=10.0,
        discrepancy=MaternNoise(sigma=20.0, l=80000.0, nu=1.0)
), weight=1.0,
                     surge_biased=True, nu=3, alpha_nonsurge=20),
        # eps_max bounds the coherent logit error (eps = eps_max*tanh(u/
        # eps_max)): below the bound the Gaussian model is unchanged, but a
        # full class flip (~3 logits at s_H=10) is unreachable, so missing
        # tongues (Ruth/Tokositna/Eldridge absorbed at |eps|~4 without it)
        # keep a permanent Brier floor and gradient. eps_max=1 ~ trusting
        # outlines to ~W_t/3 (~100 m of margin at a 300 m transition width);
        # check last["saturated_frac"] / eps pinned at the bound in
        # extent_logit_eps for real outline errors larger than that.
        ExtentSpec(weight=1.0, s_H=10.0, sigma_p=0.3,
                   logit_error=MaternNoise(sigma=0.3, l=1000.0),
                   nuisance_inner_steps=2, eps_max=1.0),
        BedSpec(weight=0.0e-6),
        SnowlineSpec(weight=Schedule(final=1.0, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1.0),
                     s_smb=0.5, sigma_p=0.3,
                     logit_error=MaternNoise(sigma=0.3, l=1000.0),
                     nuisance_inner_steps=2, eps_max=1.0),
        # dhdt discrepancy: sigma_D is in units of sigma_pix (= 0.5 x the
        # reported per-pixel error — the member is registered with unit
        # sigma and whitens r/sigma_pix), so sigma_D=1.0 trusts the
        # REGIONAL rate level to one sigma_pix (~0.15-0.25 m/yr physical)
        # per 80-km patch: the regional geodetic-bias / forcing-bias budget.
        # This is the domain-mean melt-rate channel through which the smooth
        # SMB fields (H_atm, f) were pinned; the experiment tests whether
        # that pinning was real.
        DhdtSpec(noise=MaternNoise(sigma=0.5, l=3000.0, nu=0.5, nugget=0.5,
                                   discrepancy=MaternNoise(sigma=1.0, l=80000.0, nu=1.0)),
                 weight=Schedule(final=1.0, ramp=lambda i, level: 0.0 if (i < 0 and level == 2) else 1.0)),
        BedSlopeSpec(weight=1e-5,s_scale=5.0),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
                         enabled=True,
                         sigma_picks=100,
                         sigma_dem=100,
                         pcg_rtol=1e-3,
                         pcg_rtol_adjoint=1e-2),
    sliding_m=1./3.,
    u_reg=1.0,
    beta_init=5.0,
    init_from_observed_geometry = True,
    use_avalanche_model = True,
    avalanche_hoisted=True,
    debris_factor=0.5,
    bed_prior = PriorHyperparams(sigma=500,    l=2000.0, nu=1),
    lr_z_bed=0.0125,
    alpha_t2m=2.5,
    dt=20.0,
    tbias_enabled=True,
    mu_log_beta = np.log(5.0),
    log_beta_prior = PriorHyperparams(sigma=1./3.,    l=1000.0, nu=1),
    lr_z_log_beta=0.05*9*9,
    max_level=2,
    max_iters=(50,50,500),
    lr_z_pbias=Schedule(final=0.05, ramp=lambda i,level:0.0 if (i<0 and level==2) else 0.05),
    lr_z_tbias=Schedule(final=1.0,ramp=lambda i,level:0.0 if (i<0 and level==2) else 1.0),
    # SGD-on-whitened-field lrs (the 0.05 finals were Adam-era scalar steps);
    # start conservative and retune on the level-2 trace.
    lr_z_log_H_atm=Schedule(final=1e-3,ramp=lambda i,level:0.0 if (i<0 and level==2) else 1e-3),
    lr_z_logit_cloud=Schedule(final=1e-1,ramp=lambda i,level:0.0 if (i<0 and level==2) else 1e-1),
)

