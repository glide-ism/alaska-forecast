"""
Loss assembly (data misfit + prior) and RTO mean perturbations.

The per-product data misfits live on the Observation subclasses in
`observations.py`; this module keeps the shared numerical helpers, the prior
terms, and the `LossTerms` container. The same loss is used by deterministic
MAP (zero prior means) and RTO sampling (non-zero whitened-space means).
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from ggapp.torch import GGaPPWhiten


def _huber(r, nu):
    """Huberized squared residual: sqrt(1 + (r/nu)^2) - 1 times nu^2."""
    return nu ** 2 * (torch.sqrt(1 + (r / nu) ** 2) - 1)


@dataclass
class PriorMeans:
    """Whitened-space prior means.

    For deterministic MAP, leave fields at None (they default to zero). For
    randomize-then-optimize, populate each field with N(0, I) draws shaped
    like the corresponding whitened parameter tensor.
    """
    z_bed:        Optional[torch.Tensor] = None
    z_bed_mean:   Optional[torch.Tensor] = None
    z_log_beta:   Optional[torch.Tensor] = None
    z_pbias:      Optional[torch.Tensor] = None
    # Temperature bias field: zero mean until the RTO migration wires it into
    # sample_like (its draw must be appended AFTER the existing draw order so
    # QMC dimension assignment stays stable).
    z_tbias:      Optional[torch.Tensor] = None
    z_log_mf:     Optional[torch.Tensor] = None
    z_log_rf:     Optional[torch.Tensor] = None
    # Precip-depletion scalars. Present so compute_prior can ask for their mean
    # uniformly; left out of sample_like (RTO does not yet perturb them), so they
    # default to a zero mean — i.e. no per-sample perturbation.
    z_tau:        Optional[torch.Tensor] = None
    z_z0:         Optional[torch.Tensor] = None
    # Enthalpy-model scalars: same status as tau/z0 — zero mean until the RTO
    # migration wires them into sample_like (new draws appended AFTER the
    # existing draw order).
    z_log_H_atm:   Optional[torch.Tensor] = None
    z_logit_cloud: Optional[torch.Tensor] = None

    @classmethod
    def zeros(cls) -> "PriorMeans":
        return cls()

    @classmethod
    def sample_like(cls, params) -> "PriorMeans":
        """Draw N(0, I) perturbations with the same shapes as `params`."""
        return cls(
            z_bed=torch.randn_like(params.z_bed),
            z_bed_mean=torch.randn_like(params.z_bed_mean),
            z_log_beta=torch.randn_like(params.z_log_beta),
            z_pbias=torch.randn_like(params.z_pbias),
            z_log_mf=torch.randn_like(params.z_log_mf),
            z_log_rf=torch.randn_like(params.z_log_rf),
        )

    def value(self, name: str, like: torch.Tensor) -> torch.Tensor:
        v = getattr(self, name)
        return torch.zeros_like(like) if v is None else v


# Display labels for the log banner, keyed by Observation.name. Unknown names
# fall back to the raw key so custom observations still log.
_LOG_LABELS = {
    "srf": "Srf", "vel": "U", "extent": "Ext",
    "bed": "Bed", "snow": "Snow", "dhdt": "dHdt", "divide": "Div",
    "bedslope": "BSlope",
}


@dataclass
class LossTerms:
    """All loss terms from one evaluation.

    `data_terms` is keyed by Observation.name — one entry per observation the
    domain actually has (absent products simply have no entry). The historical
    `J_srf` ... `J_dhdt` accessors read from it, returning 0 for a missing
    product, so existing analysis code keeps working.
    """
    data_terms: dict = field(default_factory=dict)
    J_prior_bed:      torch.Tensor = None
    J_prior_bed_mean: torch.Tensor = None
    J_prior_beta:     torch.Tensor = None
    J_prior_pbias:    torch.Tensor = None
    J_prior_tbias:    torch.Tensor = None
    J_prior_smb:      torch.Tensor = None

    def _term(self, name: str):
        if name in self.data_terms:
            return self.data_terms[name]
        return torch.zeros((), device=self.J_prior_bed.device) \
            if isinstance(self.J_prior_bed, torch.Tensor) else 0.0

    @property
    def J_srf(self): return self._term("srf")
    @property
    def J_vel(self): return self._term("vel")
    @property
    def J_extent(self): return self._term("extent")
    @property
    def J_bed(self): return self._term("bed")
    @property
    def J_snow(self): return self._term("snow")
    @property
    def J_dhdt(self): return self._term("dhdt")

    @property
    def J_data(self):
        return sum(self.data_terms.values())

    @property
    def J_prior(self):
        return (self.J_prior_bed + self.J_prior_bed_mean + self.J_prior_beta
                + self.J_prior_pbias + self.J_prior_tbias + self.J_prior_smb)

    @property
    def J(self):
        return self.J_data + self.J_prior

    def log(self, i: int) -> None:
        bar = "=" * 60
        print(bar)
        print(f"Iteration: {i}, Total Loss: {self.J.item():.2f}, "
              f"Data Loss: {float(self.J_data):.2f}, "
              f"Prior Loss: {self.J_prior.item():.2f}")
        print(", ".join(
            f"{_LOG_LABELS.get(name, name)} Loss: {float(term):.2f}"
            for name, term in self.data_terms.items()))
        print(f"Bed Prior: {float(self.J_prior_bed):.2f}, "
              f"Bed Mean Prior: {float(self.J_prior_bed_mean):.2f}, "
              f"Beta Prior: {float(self.J_prior_beta):.2f}, "
              f"Pbias Prior: {float(self.J_prior_pbias):.2f}, "
              f"Tbias Prior: {float(self.J_prior_tbias):.2f}")
        print(bar)


def marginal_velocity_log_likelihood(
    u_obs,          # (N, 2) observed velocity, flattened raster
    u_mod,          # (N, 2) modeled velocity
    sigma,          # (N,)   per-pixel noise scale
    labels,         # (N,)   long, glacier id per pixel, -1 = unlabeled
    eta_nodes,      # (K,)   quadrature nodes on (0, 1]
    log_w_eff,      # (K,)   log(w_k * P(eta_k)), precomputed once
    nu,             # pseudo-Huber threshold
    lamda,
    scale
):

    K = eta_nodes.shape[0]
    n_glaciers = (labels.max() + 1).item()
    labeled = labels >= 0

    # --- labeled pixels: marginalized likelihood ---
    u_obs_l = u_obs[labeled]                # (M, 2)
    u_mod_l = u_mod[labeled]                # (M, 2)
    sigma_l = sigma[labeled]                # (M,)
    lab     = labels[labeled]               # (M,)

    # residual at each quadrature node: (K, M, 2)
    # eta broadcasts as (K, 1, 1), u_obs_l as (1, M, 2)
    r = u_obs_l.unsqueeze(0) - eta_nodes[:, None, None] * u_mod_l.unsqueeze(0)

    # normalized residual magnitude squared: (K, M)
    r2 = (r / sigma_l[None, :, None]).square().sum(dim=-1)

    # pseudo-Huber log-likelihood per pixel per node: (K, M)
    phl = scale * lamda * (nu ** 2) * (torch.sqrt(1.0 + r2 / nu ** 2) - 1.0)

    # segment sum by glacier label: (K, M) -> (K, n_glaciers)
    per_glacier = torch.zeros(K, n_glaciers, device=u_obs.device, dtype=torch.float32)
    per_glacier.scatter_add_(1, lab.unsqueeze(0).expand(K, -1), phl)

    # add precomputed log effective weights, logsumexp over nodes
    marginal = torch.logsumexp(per_glacier + log_w_eff, dim=0)  # (n_glaciers,)
    ll_labeled = marginal.sum()

    # --- unlabeled pixels: standard likelihood at eta = 1 ---

    if (~labeled).any():
        r_ul = u_obs[~labeled] - u_mod[~labeled]
        r2_ul = (r_ul / sigma[~labeled, None]).square().sum(dim=-1)
        ll_unlabeled = scale * lamda * (nu ** 2) * (torch.sqrt(1.0 + r2_ul / nu ** 2) - 1.0)
        ll_labeled = ll_labeled + ll_unlabeled.sum()

    return ll_labeled


def compute_prior(
    *,
    config,
    priors,
    params,
    physical_bed: torch.Tensor,
    physical_bed_mean: torch.Tensor,
    log_rf: torch.Tensor,
    log_mf: torch.Tensor,
    prior_means: PriorMeans,
    physical_bed_uncond: Optional[torch.Tensor] = None,
) -> tuple:
    """Whitened-space Gaussian prior terms.

    physical_bed is the bed produced by the prior map; we recompute its
    whitened representation here (matching the original code) rather than
    threading it through, because GGaPPWhiten/GGaPPMap is not exactly
    self-inverse and the original used this form.
    """
    scale = config.loss_scale

    # The bed_mean hierarchy: re-whiten the UNCONDITIONAL bed fluctuation
    # about the mean field. Under bed conditioning, `physical_bed_uncond`
    # (= Map(z_bed), before the kriging correction) must be used here — the
    # correction is data-driven and must not be penalized by the prior, and
    # the mean stays out of the conditioning map so its curvature (and its
    # tuned learning rate) is identical in both parametrizations.
    base_bed = physical_bed if physical_bed_uncond is None else physical_bed_uncond
    z_bed_recomputed = GGaPPWhiten.apply(priors.bed_model, base_bed - physical_bed_mean)
    J_prior_bed = scale * ((z_bed_recomputed - prior_means.value("z_bed", z_bed_recomputed)) ** 2).sum()
    J_prior_bed_mean = scale * ((params.z_bed_mean - prior_means.value("z_bed_mean", params.z_bed_mean)) ** 2).sum()
    J_prior_beta = scale * ((params.z_log_beta - prior_means.value("z_log_beta", params.z_log_beta)) ** 2).sum()
    J_prior_pbias = scale * ((params.z_pbias - prior_means.value("z_pbias", params.z_pbias)) ** 2).sum()
    # Temperature bias: whitened-space term, so it needs no Matern model and
    # is computed unconditionally — exactly zero while the term is disabled
    # (z_tbias stays at 0 and out of the optimizer).
    J_prior_tbias = scale * ((params.z_tbias - prior_means.value("z_tbias", params.z_tbias)) ** 2).sum()
    # Standard-normal whitened priors on every scalar, including the precip-
    # depletion tau/z0 and the enthalpy-model H_atm/cloud pair (each inert when
    # its model/term is disabled: those z stay at 0).
    J_prior_smb = scale * ((params.z_log_rf - prior_means.value("z_log_rf", params.z_log_rf)) ** 2
                   + (params.z_log_mf - prior_means.value("z_log_mf", params.z_log_mf)) ** 2
                   + (params.z_tau - prior_means.value("z_tau", params.z_tau)) ** 2
                   + (params.z_z0 - prior_means.value("z_z0", params.z_z0)) ** 2
                   + (params.z_log_H_atm - prior_means.value("z_log_H_atm", params.z_log_H_atm)) ** 2
                   + (params.z_logit_cloud - prior_means.value("z_logit_cloud", params.z_logit_cloud)) ** 2)

    return (J_prior_bed, J_prior_bed_mean, J_prior_beta, J_prior_pbias,
            J_prior_tbias, J_prior_smb)
