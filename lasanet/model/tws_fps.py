

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SemanticScorePredictor(nn.Module):


    def __init__(
        self,
        pc_dim: int = 256,
        lang_dim: int = 768,
        d_model: int = 256,
        dropout: float = 0.1
    ):

        super().__init__()

        self.d_model = d_model

        self.pc_proj = nn.Sequential(
            nn.Linear(pc_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lang_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.score_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        pc_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        lang_mask: torch.Tensor = None
    ) -> torch.Tensor:


        squeeze_batch = False
        if pc_feat.dim() == 2:
            squeeze_batch = True
            pc_feat = pc_feat.unsqueeze(0)
            lang_feat = lang_feat.unsqueeze(0)
            if lang_mask is not None:
                lang_mask = lang_mask.unsqueeze(0)

        B, N, _ = pc_feat.shape
        L = lang_feat.shape[1]

        pc_proj = self.pc_proj(pc_feat)
        lang_proj = self.lang_proj(lang_feat)

        lang_attn_logits = self.lang_attn(lang_proj).squeeze(-1)
        if lang_mask is not None:
            lang_attn_logits = lang_attn_logits.masked_fill(lang_mask, float('-inf'))
        lang_attn_weights = F.softmax(lang_attn_logits, dim=-1)

        global_lang = torch.bmm(
            lang_attn_weights.unsqueeze(1), lang_proj
        ).squeeze(1)

        global_lang_expanded = global_lang.unsqueeze(1).expand(-1, N, -1)

        fused = torch.cat([pc_proj, global_lang_expanded], dim=-1)

        scores = self.score_head(fused).squeeze(-1)
        scores = torch.sigmoid(scores)

        if squeeze_batch:
            scores = scores.squeeze(0)

        return scores

class TextWeightedSemanticFPS(nn.Module):


    def __init__(
        self,
        pc_dim: int = 256,
        lang_dim: int = 768,
        d_model: int = 256,
        semantic_gamma: float = 0.5,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):

        super().__init__()

        self.semantic_gamma = semantic_gamma
        self.temperature = temperature

        self.score_predictor = SemanticScorePredictor(
            pc_dim=pc_dim,
            lang_dim=lang_dim,
            d_model=d_model,
            dropout=dropout
        )

    def compute_weighted_distance(
        self,
        coords: torch.Tensor,
        semantic_scores: torch.Tensor
    ) -> torch.Tensor:

        B, N, _ = coords.shape

        coords_sq = (coords ** 2).sum(dim=-1, keepdim=True)
        dist_sq = coords_sq + coords_sq.transpose(1, 2) - 2 * torch.bmm(coords, coords.transpose(1, 2))
        dist_sq = dist_sq.clamp(min=0)
        dist = torch.sqrt(dist_sq + 1e-8)

        semantic_weight = torch.exp(self.semantic_gamma * semantic_scores)
        weighted_dist = dist * semantic_weight.unsqueeze(1)

        return weighted_dist

    def soft_fps(
        self,
        coords: torch.Tensor,
        semantic_scores: torch.Tensor,
        num_samples: int
    ) -> tuple:

        B, N, _ = coords.shape
        device = coords.device

        weighted_dist = self.compute_weighted_distance(coords, semantic_scores)

        first_idx = semantic_scores.argmax(dim=-1)

        batch_indices = torch.arange(B, device=device)

        min_distances = weighted_dist[batch_indices, first_idx]

        sampled_indices = [first_idx]
        soft_masks = []

        first_mask = F.one_hot(first_idx, num_classes=N).float()
        soft_masks.append(first_mask)

        selected_mask = first_mask.clone()

        for _ in range(num_samples - 1):

            masked_min_dist = min_distances - selected_mask * 1e9

            if self.training:

                logits = masked_min_dist / self.temperature
                soft_selection = F.softmax(logits, dim=-1)

                hard_idx = soft_selection.argmax(dim=-1)
                hard_selection = F.one_hot(hard_idx, num_classes=N).float()
                selection = hard_selection - soft_selection.detach() + soft_selection
            else:

                hard_idx = masked_min_dist.argmax(dim=-1)
                selection = F.one_hot(hard_idx, num_classes=N).float()

            sampled_indices.append(selection.argmax(dim=-1))
            soft_masks.append(selection)

            selected_mask = selected_mask + selection

            new_distances = weighted_dist[batch_indices, sampled_indices[-1]]
            min_distances = torch.min(min_distances, new_distances)

        sampled_indices = torch.stack(sampled_indices, dim=1)
        soft_masks = torch.stack(soft_masks, dim=1)

        return sampled_indices, soft_masks

    def forward(
        self,
        coords: torch.Tensor,
        pc_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        num_samples: int,
        lang_mask: torch.Tensor = None
    ) -> tuple:

        B, N, _ = coords.shape

        semantic_scores = self.score_predictor(pc_feat, lang_feat, lang_mask)

        if N <= num_samples:
            indices = torch.arange(N, device=coords.device).unsqueeze(0).expand(B, -1)
            return indices, semantic_scores, None

        sampled_indices, soft_masks = self.soft_fps(coords, semantic_scores, num_samples)

        return sampled_indices, semantic_scores, soft_masks

class TWSFPS(TextWeightedSemanticFPS):

    pass

def semantic_fps_functional(
    coords: torch.Tensor,
    semantic_scores: torch.Tensor,
    num_samples: int,
    gamma: float = 0.5
) -> torch.Tensor:

    B, N, _ = coords.shape
    device = coords.device

    coords_sq = (coords ** 2).sum(dim=-1, keepdim=True)
    dist_sq = coords_sq + coords_sq.transpose(1, 2) - 2 * torch.bmm(coords, coords.transpose(1, 2))
    dist = torch.sqrt(dist_sq.clamp(min=0) + 1e-8)

    semantic_weight = torch.exp(gamma * semantic_scores)
    weighted_dist = dist * semantic_weight.unsqueeze(1)

    sampled_indices = []
    batch_indices = torch.arange(B, device=device)

    first_idx = semantic_scores.argmax(dim=-1)
    sampled_indices.append(first_idx)

    min_distances = weighted_dist[batch_indices, first_idx]
    selected = torch.zeros(B, N, device=device, dtype=torch.bool)
    selected[batch_indices, first_idx] = True

    for _ in range(num_samples - 1):
        masked_dist = min_distances.clone()
        masked_dist[selected] = -float('inf')

        next_idx = masked_dist.argmax(dim=-1)
        sampled_indices.append(next_idx)

        selected[batch_indices, next_idx] = True
        new_dist = weighted_dist[batch_indices, next_idx]
        min_distances = torch.min(min_distances, new_dist)

    return torch.stack(sampled_indices, dim=1)

if __name__ == '__main__':
    """测试代码"""
    B, N, L = 2, 500, 50
    pc_dim, lang_dim = 256, 768
    num_samples = 128

    coords = torch.randn(B, N, 3)
    pc_feat = torch.randn(B, N, pc_dim)
    lang_feat = torch.randn(B, L, lang_dim)
    lang_mask = torch.zeros(B, L, dtype=torch.bool)
    lang_mask[:, 40:] = True

    tws_fps = TWSFPS(
        pc_dim=pc_dim,
        lang_dim=lang_dim,
        semantic_gamma=0.5,
        temperature=1.0
    )

    tws_fps.train()
    indices, scores, soft_masks = tws_fps(
        coords, pc_feat, lang_feat, num_samples, lang_mask
    )

    print(f"sampled_indices shape: {indices.shape}")
    print(f"semantic_scores shape: {scores.shape}")
    print(f"soft_masks shape: {soft_masks.shape}")

    indices_func = semantic_fps_functional(coords, scores, num_samples)
    print(f"[Functional] sampled_indices shape: {indices_func.shape}")
