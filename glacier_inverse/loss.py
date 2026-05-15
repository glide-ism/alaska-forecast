"""
Loss terms (data misfit + prior) and RTO mean perturbations.

The same loss is used by deterministic MAP (zero prior means) and RTO
sampling (non-zero prior means drawn from N(0, I) in whitened coordinates).
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.functional import grid_sample

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
    z_log_mf:     Optional[torch.Tensor] = None
    z_log_rf:     Optional[torch.Tensor] = None

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


@dataclass
class LossTerms:
    J_srf:            torch.Tensor
    J_vel:            torch.Tensor
    J_extent:         torch.Tensor
    J_bed:            torch.Tensor
    J_prior_bed:      torch.Tensor
    J_prior_bed_mean: torch.Tensor
    J_prior_beta:     torch.Tensor
    J_prior_pbias:    torch.Tensor
    J_prior_smb:      torch.Tensor

    @property
    def J_data(self):
        return self.J_srf + self.J_vel + self.J_extent + self.J_bed

    @property
    def J_prior(self):
        return (self.J_prior_bed + self.J_prior_bed_mean + self.J_prior_beta
                + self.J_prior_pbias + self.J_prior_smb)

    @property
    def J(self):
        return self.J_data + self.J_prior

    def log(self, i: int) -> None:
        bar = "=" * 60
        print(bar)
        print(f"Iteration: {i}, Total Loss: {self.J.item():.2f}, "
              f"Data Loss: {self.J_data.item():.2f}, "
              f"Prior Loss: {self.J_prior.item():.2f}")
        print(f"Srf Loss: {self.J_srf.item():.2f}, "
              f"U Loss: {self.J_vel.item():.2f}, "
              f"Ext Loss: {self.J_extent.item():.2f}, "
              f"Bed Loss: {self.J_bed.item():.2f}")
        print(f"Bed Prior: {float(self.J_prior_bed):.2f}, "
              f"Bed Mean Prior: {float(self.J_prior_bed_mean):.2f}, "
              f"Beta Prior: {float(self.J_prior_beta):.2f}, "
              f"Pbias Prior: {float(self.J_prior_pbias):.2f}")
        print(bar)


def compute_data_misfit(
    *,
    config,
    sim_result,
    bed_fine: torch.Tensor,
    observations,
    mask: torch.Tensor,
    dx: float,
) -> tuple:
    """(J_srf, J_vel, J_extent, J_bed) on the fine grid."""
    S = sim_result.S_fine
    u_fine = sim_result.u_fine
    v_fine = sim_result.v_fine
    H_fine = sim_result.H_fine

    scale = config.loss_scale * dx ** 2

    # Surface elevation misfit.
    r_s = (S - observations.S_obs) / config.sigma_s
    J_srf = scale * config.lambda_s * _huber(r_s, config.nu_s).sum()

    # Velocity misfit (averaged onto cell centers).
    u_pred = (u_fine[:, 1:] + u_fine[:, :-1]) / 2.0
    v_pred = (v_fine[1:, :] + v_fine[:-1, :]) / 2.0
    r_u2 = ((u_pred - observations.u_obs) ** 2 + (v_pred - observations.v_obs) ** 2) / config.sigma_u ** 2
    J_vel = scale * config.lambda_u * config.nu_u ** 2 * (torch.sqrt(1 + r_u2 / config.nu_u ** 2) - 1).sum()

    # Glacier extent (Bernoulli log-likelihood with mask).
    p_extent = (2.0 / (1 + torch.exp(-H_fine / config.s_H)) - 1).clip(min=0.001, max=0.999)
    J_extent = -config.loss_scale * config.lambda_e * dx ** 2 * (mask * torch.log(p_extent)).sum()

    # Bed: flightline samples + grid bed-equals-surface where no ice (1-mask).
    bed_at_flightlines = grid_sample(
        bed_fine[None, None, :, :],
        observations.bed_normed_coords[None, None, :, :],
        mode="bilinear",
    ).squeeze()
    r_bed_fl = (bed_at_flightlines - observations.bed_obs) / config.sigma_bed
    r_bed_grid = (bed_fine - observations.S_obs) / config.sigma_s * (1 - mask)
    J_bed = scale * config.lambda_bed * (_huber(r_bed_fl, config.nu_bed).sum()
                                          + _huber(r_bed_grid, config.nu_bed).sum())

    return J_srf, J_vel, J_extent, J_bed


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
) -> tuple:
    """Whitened-space Gaussian prior terms.

    physical_bed is the bed produced by the prior map; we recompute its
    whitened representation here (matching the original code) rather than
    threading it through, because GGaPPWhiten/GGaPPMap is not exactly
    self-inverse and the original used this form.
    """
    scale = config.loss_scale

    z_bed_recomputed = GGaPPWhiten.apply(priors.bed_model, physical_bed - physical_bed_mean)
    J_prior_bed = scale * ((z_bed_recomputed - prior_means.value("z_bed", z_bed_recomputed)) ** 2).sum()
    J_prior_bed_mean = scale * ((params.z_bed_mean - prior_means.value("z_bed_mean", params.z_bed_mean)) ** 2).sum()
    J_prior_beta = scale * ((params.z_log_beta - prior_means.value("z_log_beta", params.z_log_beta)) ** 2).sum()
    J_prior_pbias = scale * ((params.z_pbias - prior_means.value("z_pbias", params.z_pbias)) ** 2).sum()

    z_log_rf_now = (log_rf - priors.mu_log_rf) / priors.sigma_log_rf
    z_log_mf_now = (log_mf - priors.mu_log_mf) / priors.sigma_log_mf
    J_prior_smb = ((z_log_rf_now - prior_means.value("z_log_rf", z_log_rf_now)) ** 2
                   + (z_log_mf_now - prior_means.value("z_log_mf", z_log_mf_now)) ** 2)

    return J_prior_bed, J_prior_bed_mean, J_prior_beta, J_prior_pbias, J_prior_smb
