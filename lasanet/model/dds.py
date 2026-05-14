

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False,
                   dim: int = -1, eps: float = 1e-10) -> torch.Tensor:


    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype)
    )
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:

        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret

def gumbel_topk(logits: torch.Tensor, k: int, tau: float = 1.0,
                hard: bool = True, dim: int = -1) -> tuple:

    B, N = logits.shape
    device = logits.device

    selected_masks = []

    cumulative_mask = torch.zeros(B, N, device=device)

    for i in range(k):

        masked_logits = logits - cumulative_mask * 1e9

        soft_selection = gumbel_softmax(masked_logits, tau=tau, hard=hard, dim=dim)
        selected_masks.append(soft_selection)

        if hard:
            cumulative_mask = cumulative_mask + soft_selection.detach()
        else:
            cumulative_mask = cumulative_mask + soft_selection

    selected_mask = torch.stack(selected_masks, dim=1)

    selected_indices = selected_mask.argmax(dim=-1)

    return selected_mask, selected_indices

class DifferentiableDiverseSelection(nn.Module):


    def __init__(
        self,
        num_proposal: int = 128,
        pc_dim: int = 256,
        lang_dim: int = 256,
        d_model: int = 256,
        tau_start: float = 1.0,
        tau_end: float = 0.5,
        tau_anneal_steps: int = 10000,
        diversity_weight: float = 0.1,
        use_diversity_loss: bool = True,

        diversity_margin: float = 0.1,
        diversity_focal_gamma: float = 2.0,
        spatial_sigma: float = 0.5,
    ):

        super().__init__()

        self.num_proposal = num_proposal
        self.d_model = d_model
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_anneal_steps = tau_anneal_steps
        self.diversity_weight = diversity_weight
        self.use_diversity_loss = use_diversity_loss

        self.diversity_margin = diversity_margin
        self.diversity_focal_gamma = diversity_focal_gamma
        self.spatial_sigma = spatial_sigma

        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))

        self.query_proj = nn.Linear(pc_dim, d_model, bias=False)

        self.key_proj = nn.Linear(lang_dim, d_model, bias=False)

        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        if use_diversity_loss:
            self.diversity_proj = nn.Linear(pc_dim, d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_tau(self) -> float:

        if self.training:

            progress = min(self.current_step.item() / self.tau_anneal_steps, 1.0)
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress
            return tau
        else:

            return self.tau_end

    def compute_relevance_scores(
        self,
        pc_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        pc_mask: torch.Tensor = None,
        lang_mask: torch.Tensor = None
    ) -> torch.Tensor:

        B, N, _ = pc_feat.shape
        L = lang_feat.shape[1]

        Q = self.query_proj(pc_feat)
        K = self.key_proj(lang_feat)

        scale = math.sqrt(self.d_model)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scale

        if lang_mask is not None:

            attn_scores = attn_scores.masked_fill(
                lang_mask.unsqueeze(1), float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        lang_context = torch.bmm(attn_weights, K)

        fused = Q + lang_context

        scores = self.score_mlp(fused).squeeze(-1)

        if pc_mask is not None:
            scores = scores.masked_fill(pc_mask, float('-inf'))

        return scores

    def compute_diversity_loss(
        self,
        selected_features: torch.Tensor,
        selected_coords: torch.Tensor = None
    ) -> torch.Tensor:

        B, K, D = selected_features.shape

        if K <= 1:
            return torch.tensor(0.0, device=selected_features.device)

        norm_features = F.normalize(selected_features, p=2, dim=-1)

        sim_matrix = torch.bmm(norm_features, norm_features.transpose(1, 2))

        diag_mask = torch.eye(K, device=selected_features.device).bool()
        sim_matrix = sim_matrix.masked_fill(diag_mask.unsqueeze(0), 0.0)

        margin = self.diversity_margin

        sim_penalty = torch.clamp(sim_matrix - margin, min=0.0)

        gamma = self.diversity_focal_gamma

        max_penalty = 1.0 - margin + 1e-8
        normalized_penalty = sim_penalty / max_penalty
        focal_weight = normalized_penalty.pow(gamma)

        weighted_penalty = focal_weight * sim_penalty

        if selected_coords is not None:

            dist_matrix = torch.cdist(selected_coords, selected_coords, p=2)

            spatial_weight = torch.exp(-dist_matrix / (self.spatial_sigma + 1e-8))

            spatial_weight = spatial_weight.masked_fill(diag_mask.unsqueeze(0), 0.0)

            weighted_penalty = weighted_penalty * spatial_weight

        num_pairs = K * (K - 1)
        loss = weighted_penalty.sum() / (B * num_pairs + 1e-8)

        return loss

    def forward(
        self,
        pc_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        pc_mask: torch.Tensor = None,
        lang_mask: torch.Tensor = None,
        gate: torch.Tensor = None,
        return_diversity_loss: bool = False,
        pc_coords: torch.Tensor = None
    ) -> tuple:

        B, N, _ = pc_feat.shape

        if self.training:
            self.current_step += 1

        ref_scores = self.compute_relevance_scores(
            pc_feat, lang_feat, pc_mask, lang_mask
        )

        if gate is not None:
            ref_scores = ref_scores * gate

        if N <= self.num_proposal:
            sample_inds = torch.arange(N, device=pc_feat.device).unsqueeze(0).repeat(B, 1).int()
            if return_diversity_loss:
                return sample_inds, ref_scores, torch.tensor(0.0, device=pc_feat.device)
            return sample_inds, ref_scores

        tau = self.get_tau()

        if self.training:

            selected_mask, sample_inds = gumbel_topk(
                ref_scores, k=self.num_proposal, tau=tau, hard=True
            )

        else:

            sample_inds = torch.topk(ref_scores, self.num_proposal, dim=-1)[1]

        sample_inds = sample_inds.int()

        if return_diversity_loss and self.use_diversity_loss and self.training:

            sample_inds_long = sample_inds.long()
            selected_features = pc_feat.gather(
                dim=1,
                index=sample_inds_long.unsqueeze(-1).expand(-1, -1, pc_feat.shape[-1])
            )

            selected_features = self.diversity_proj(selected_features)

            selected_coords = None
            if pc_coords is not None:
                selected_coords = pc_coords.gather(
                    dim=1,
                    index=sample_inds_long.unsqueeze(-1).expand(-1, -1, pc_coords.shape[-1])
                )

            diversity_loss = self.compute_diversity_loss(selected_features, selected_coords)

            return sample_inds, ref_scores, diversity_loss * self.diversity_weight

        if return_diversity_loss:
            return sample_inds, ref_scores, torch.tensor(0.0, device=pc_feat.device)

        return sample_inds, ref_scores

class DDS(DifferentiableDiverseSelection):

    pass

if __name__ == '__main__':
    """测试代码"""

    B, N, L = 2, 512, 50
    pc_dim, lang_dim, d_model = 256, 256, 256

    pc_feat = torch.randn(B, N, pc_dim)
    pc_coords = torch.randn(B, N, 3)
    lang_feat = torch.randn(B, L, lang_dim)
    pc_mask = None
    lang_mask = torch.zeros(B, L, dtype=torch.bool)
    lang_mask[:, 40:] = True

    dds = DDS(
        num_proposal=128,
        pc_dim=pc_dim,
        lang_dim=lang_dim,
        d_model=d_model,
        tau_start=1.0,
        tau_end=0.5,
        diversity_weight=0.1,
        diversity_margin=0.1,
        diversity_focal_gamma=2.0,
        spatial_sigma=0.5
    )

    dds.train()
    sample_inds, ref_scores, div_loss = dds(
        pc_feat, lang_feat, pc_mask, lang_mask,
        return_diversity_loss=True
    )
    print(f"[无空间] sample_inds shape: {sample_inds.shape}")
    print(f"[无空间] diversity_loss: {div_loss.item():.4f}")

    sample_inds, ref_scores, div_loss_spatial = dds(
        pc_feat, lang_feat, pc_mask, lang_mask,
        return_diversity_loss=True,
        pc_coords=pc_coords
    )
    print(f"[空间感知] diversity_loss: {div_loss_spatial.item():.4f}")

    dds.eval()
    sample_inds, ref_scores = dds(pc_feat, lang_feat, pc_mask, lang_mask)
    print(f"[Eval] sample_inds shape: {sample_inds.shape}")
