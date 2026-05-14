

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

class ReliabilityAwareCrossModalResidualGating(nn.Module):


    def __init__(
        self,
        d_model: int = 256,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        entropy_temp: float = 1.0,
        min_gate: float = 0.0,

        bottleneck_dim: int = None,
    ):

        super().__init__()
        self.d_model = d_model
        self.entropy_temp = entropy_temp
        self.min_gate = min_gate

        if bottleneck_dim is not None:
            hidden_dim = bottleneck_dim
        else:
            hidden_dim = d_model * mlp_ratio

        self.reliability_net = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

        self.residual_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_attention_entropy(
        self,
        attn_weights: torch.Tensor,
        lang_mask: torch.Tensor = None,
    ) -> torch.Tensor:


        attn = attn_weights.transpose(1, 2)

        B, M, L = attn.shape

        attn = attn.clamp(min=0.0)

        if lang_mask is not None:

            valid_mask = (~lang_mask).float().unsqueeze(1)
            attn = attn * valid_mask

            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn = attn / attn_sum

        eps = 1e-8

        entropy = -(attn * torch.log(attn.clamp(min=eps))).sum(dim=-1)

        if lang_mask is not None:
            L_valid = (~lang_mask).float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            L_valid = torch.tensor(float(L), device=attn.device)
        max_entropy = torch.log(L_valid + eps)
        entropy_norm = entropy / max_entropy.clamp(min=eps)
        entropy_norm = entropy_norm.clamp(0.0, 1.0)

        reliability_input = (1.0 - entropy_norm).unsqueeze(-1)
        reliability = torch.sigmoid(self.reliability_net(reliability_input))

        return reliability

    def forward(
        self,
        query: torch.Tensor,
        prompt: torch.Tensor,
        attn_weights: torch.Tensor,
        lang_mask: torch.Tensor = None,
        topk_mask: torch.Tensor = None,
    ) -> torch.Tensor:


        reliability = self.compute_attention_entropy(
            attn_weights, lang_mask
        )

        if topk_mask is not None:
            if self.min_gate > 0:
                soft_mask = topk_mask * (1.0 - self.min_gate) + self.min_gate
                reliability = reliability * soft_mask.unsqueeze(-1)
            else:
                reliability = reliability * topk_mask.unsqueeze(-1)

        prompt_transformed = self.residual_mlp(prompt)

        query_enhanced = query + reliability * prompt_transformed

        query_enhanced = self.norm(query_enhanced)

        return query_enhanced

class ResidualBridgedDifferentiableTopK(nn.Module):


    def __init__(
        self,
        top_k: int = 64,
        ste_scale: float = 1.0,
        residual_weight_init: float = 0.1,

        topk_ratio: float = None,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        tau_anneal_steps: int = 10000,

        k_anneal_start: int = None,
        k_anneal_steps: int = 0,
    ):

        super().__init__()
        self.top_k = top_k
        self.topk_ratio = topk_ratio
        self.ste_scale = ste_scale
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_anneal_steps = tau_anneal_steps
        self.k_anneal_start = k_anneal_start
        self.k_anneal_steps = k_anneal_steps

        self.residual_weight = nn.Parameter(torch.tensor(residual_weight_init))

        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    @property
    def current_tau(self) -> float:

        progress = min(self._step.item() / max(self.tau_anneal_steps, 1), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    @property
    def current_k(self) -> int:

        if self.k_anneal_start is None or self.k_anneal_steps <= 0:
            return self.top_k
        progress = min(self._step.item() / max(self.k_anneal_steps, 1), 1.0)
        k_float = self.k_anneal_start + (self.top_k - self.k_anneal_start) * progress
        return max(self.top_k, int(round(k_float)))

    def forward(
        self,
        scores: torch.Tensor,
        num_queries: int = None,
    ) -> torch.Tensor:

        B, M = scores.shape

        if num_queries is not None:
            K = num_queries
        elif self.topk_ratio is not None:
            K = max(1, int(M * self.topk_ratio))
        else:
            K = self.current_k
        K = min(K, M)

        if self.training:
            self._step += 1

        tau = self.current_tau

        _, topk_indices = torch.topk(scores, K, dim=-1)

        mask_hard = torch.zeros_like(scores)
        mask_hard.scatter_(1, topk_indices, 1.0)

        if not self.training:

            return mask_hard

        mask_soft = F.softmax(scores / tau, dim=-1) * M

        mask_soft = mask_soft / (mask_soft.max(dim=-1, keepdim=True)[0].clamp(min=1e-8))
        mask_soft = mask_soft.clamp(0.0, 1.0)

        mask = mask_hard + (mask_soft - mask_soft.detach())

        return mask

class DecoderEmbeddedVariationalQueryDenoising(nn.Module):


    def __init__(
        self,
        d_model: int = 256,
        latent_dim: int = 128,
        kl_weight: float = 0.01,
        dropout: float = 0.1,

        d_latent: int = None,
        scale_init: float = 0.01,
    ):

        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent if d_latent is not None else latent_dim
        self.kl_weight = kl_weight
        self.dropout_rate = dropout

        d_lat = self.d_latent

        self.mu_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_lat),
        )

        self.log_var_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_lat),
        )

        self.decoder_proj = nn.Linear(d_lat, d_model)

        self.alpha = nn.Parameter(torch.tensor(scale_init))

        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if hasattr(self.log_var_net[-1], 'bias') and self.log_var_net[-1].bias is not None:
            nn.init.constant_(self.log_var_net[-1].bias, -2.0)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        if self.training:

            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:

            z = mu
        return z

    def compute_kl_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:


        kl = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0)

        kl_loss = kl.sum(dim=-1).mean()
        return kl_loss

    def forward(
        self,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        mu = self.mu_net(query)
        log_var = self.log_var_net(query)

        log_var = log_var.clamp(-10.0, 2.0)

        z = self.reparameterize(mu, log_var)

        z_proj = self.decoder_proj(z)
        query_denoised = query + self.alpha * z_proj

        query_denoised = self.norm(query_denoised)

        if self.training:
            kl_loss = self.kl_weight * self.compute_kl_loss(mu, log_var)
        else:
            kl_loss = torch.tensor(0.0, device=query.device)

        return query_denoised, kl_loss
