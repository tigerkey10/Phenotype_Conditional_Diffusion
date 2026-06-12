"""Gaussian diffusion (DDPM): forward process, training loss, classifier-free
guidance, and deterministic DDIM sampling (plus SDEdit editing).

Forward process: q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I), T = 1000 steps,
linear OR cosine beta schedule (`schedule`). The denoiser predicts either the clean
signal x0 or the noise eps (`pred_type`); x0-prediction is the project default
because eps-prediction is unstable on this sparse modality (see README §5.3).

Key methods:
  p_losses     -- training loss; optional dropout-gate BCE + nonzero-weighted MSE.
  ddim_sample  -- generate from noise, with per-gene x0 clipping for stability.
  ddim_edit    -- SDEdit: noise a real cell partway, denoise toward a target class
                  (per-cell counterfactual / perturbation editing, README §10-11).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def make_beta_schedule(name: str, T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if name == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64).float()
    if name == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
        abar = f / f[0]
        betas = 1 - (abar[1:] / abar[:-1])
        return betas.clamp(1e-8, 0.999).float()
    raise ValueError(f"unknown schedule {name}")


class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
        pred_type: str = "x0",
        device: Optional[torch.device] = None,
    ):
        self.T = timesteps
        self.pred_type = pred_type  # "eps" (predict noise) | "x0" (predict signal)
        self.device = device or torch.device("cpu")
        betas = make_beta_schedule(schedule, timesteps, beta_start, beta_end).to(self.device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_acp[t][:, None] * x0 + self.sqrt_one_minus_acp[t][:, None] * noise

    def p_losses(self, model, x0: torch.Tensor, c: torch.Tensor, cfg_dropout: float,
                 mask: Optional[torch.Tensor] = None, gate_weight: float = 0.0,
                 nonzero_weight: float = 1.0):
        """Diffusion magnitude loss, optionally with a hurdle gate head.

        mask: (B, G) binary indicator of expressed (nonzero) genes in x0.
        gate_weight: BCE weight for the on/off head (0 disables the gate).
        nonzero_weight: per-entry weight on expressed genes in the magnitude MSE,
            so the 93%-zero background does not dominate nonzero fidelity.
        Returns (total_loss, mag_loss, gate_loss) for logging.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Classifier-free guidance: randomly replace label with the null token.
        if cfg_dropout > 0:
            drop = torch.rand(b, device=x0.device) < cfg_dropout
            c = c.clone()
            c[drop] = model.null_idx

        use_gate = gate_weight > 0 and mask is not None and getattr(model, "gate_head", False)
        if use_gate:
            pred, gate_logits = model.forward_both(x_t, t, c)
        else:
            pred = model(x_t, t, c)

        target = noise if self.pred_type == "eps" else x0
        if nonzero_weight != 1.0 and mask is not None:
            w = 1.0 + (nonzero_weight - 1.0) * mask
            mag_loss = (w * (pred - target) ** 2).sum() / w.sum()
        else:
            mag_loss = F.mse_loss(pred, target)

        gate_loss = torch.zeros((), device=x0.device)
        if use_gate:
            gate_loss = F.binary_cross_entropy_with_logits(gate_logits, mask)
        total = mag_loss + gate_weight * gate_loss
        return total, mag_loss.detach(), gate_loss.detach()

    @torch.no_grad()
    def _guided_out(self, model, x_t, t, c, w: float) -> torch.Tensor:
        """Guided model output in its native space (eps or x0). CFG is linear in
        either space because eps and x0 are affine in each other given x_t."""
        if w == 0:
            return model(x_t, t, c)
        null = torch.full_like(c, model.null_idx)
        out_c = model(x_t, t, c)
        out_u = model(x_t, t, null)
        return (1.0 + w) * out_c - w * out_u

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        n: int,
        n_genes: int,
        c: torch.Tensor,
        steps: int = 50,
        w: float = 0.0,
        eta: float = 0.0,
        x0_clip=None,
    ) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sampling with CFG guidance scale w.

        x0_clip: optional (lo, hi) tensors. The predicted x0 is divided by
        sqrt(abar_t), which is ~6e-3 at the first step, so any error in the noise
        prediction is amplified ~150x and the trajectory diverges. Clipping the
        predicted x0 to the observed per-gene data range each step keeps sampling
        stable (standard practice in DDPM/DDIM image samplers).
        """
        model.eval()
        device = self.device
        x = torch.randn(n, n_genes, device=device)
        lo, hi = (None, None) if x0_clip is None else x0_clip

        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        for i in range(steps):
            t = ts[i]
            t_b = torch.full((n,), int(t), device=device, dtype=torch.long)
            out = self._guided_out(model, x, t_b, c, w)

            acp_t = self.alphas_cumprod[t]
            if self.pred_type == "eps":
                eps = out
                x0_pred = (x - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)
            else:  # x0-prediction: model output is the clean signal directly
                x0_pred = out
            if lo is not None:
                x0_pred = torch.clamp(x0_pred, lo, hi)
            # Recover the noise estimate consistent with (possibly clipped) x0.
            eps = (x - torch.sqrt(acp_t) * x0_pred) / torch.sqrt(1 - acp_t)

            if i < steps - 1:
                t_prev = ts[i + 1]
                acp_prev = self.alphas_cumprod[t_prev]
                sigma = eta * torch.sqrt(
                    (1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)
                )
                dir_xt = torch.sqrt(torch.clamp(1 - acp_prev - sigma**2, min=0.0)) * eps
                noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(acp_prev) * x0_pred + dir_xt + noise
            else:
                x = x0_pred
        return x

    @torch.no_grad()
    def ddim_edit(self, model, x0_real, c_target, strength: float,
                  steps: int = 50, w: float = 0.0, x0_clip=None) -> torch.Tensor:
        """SDEdit-style per-cell counterfactual: noise a real cell to a fraction
        `strength` of the trajectory, then reverse-denoise toward c_target. Low
        strength preserves the source cell; high strength approaches a fresh
        c_target sample. x0_real is in standardized space, shape (B, n_genes).
        """
        model.eval()
        device = self.device
        b, n_genes = x0_real.shape
        lo, hi = (None, None) if x0_clip is None else x0_clip

        t_start = int(max(1, min(self.T - 1, round(strength * (self.T - 1)))))
        noise = torch.randn_like(x0_real)
        t0 = torch.full((b,), t_start, device=device, dtype=torch.long)
        x = self.q_sample(x0_real, t0, noise)        # forward-noise the real cell

        ts = torch.linspace(t_start, 0, steps, dtype=torch.long, device=device)
        for i in range(steps):
            t = ts[i]
            t_b = torch.full((b,), int(t), device=device, dtype=torch.long)
            out = self._guided_out(model, x, t_b, c_target, w)
            acp_t = self.alphas_cumprod[t]
            x0_pred = out if self.pred_type != "eps" else \
                (x - torch.sqrt(1 - acp_t) * out) / torch.sqrt(acp_t)
            if lo is not None:
                x0_pred = torch.clamp(x0_pred, lo, hi)
            eps = (x - torch.sqrt(acp_t) * x0_pred) / torch.sqrt(1 - acp_t)
            if i < steps - 1:
                acp_prev = self.alphas_cumprod[ts[i + 1]]
                x = torch.sqrt(acp_prev) * x0_pred + torch.sqrt(1 - acp_prev) * eps
            else:
                x = x0_pred
        return x
