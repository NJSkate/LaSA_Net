

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .tws_fps import TextWeightedSemanticFPS, TWSFPS
from .dds import DDS

class SemanticAdaptiveQueryGeneration(nn.Module):


    def __init__(
        self,

        d_model: int = 256,
        pc_dim: int = None,
        lang_dim: int = 768,
        proj_lang_dim: int = 256,
        dropout: float = 0.1,

        use_tws_fps: bool = True,
        tws_fps_cfg: Dict = None,

        use_dds: bool = False,
        dds_cfg: Dict = None,
        num_proposal: int = 128,
        fps_num: int = 512
    ):

        super().__init__()

        if pc_dim is None:
            pc_dim = d_model

        self.d_model = d_model
        self.pc_dim = pc_dim
        self.lang_dim = lang_dim
        self.proj_lang_dim = proj_lang_dim
        self.num_proposal = num_proposal
        self.fps_num = fps_num

        self.use_tws_fps = use_tws_fps
        self.use_dds = use_dds

        if tws_fps_cfg is None:
            tws_fps_cfg = {}
        if dds_cfg is None:
            dds_cfg = {}

        if use_tws_fps:
            self.tws_fps = TWSFPS(
                pc_dim=pc_dim,
                lang_dim=lang_dim,
                d_model=d_model,
                semantic_gamma=tws_fps_cfg.get('semantic_gamma', 0.5),
                temperature=tws_fps_cfg.get('temperature', 1.0),
                dropout=dropout
            )
        else:
            self.tws_fps = None

        if use_dds:
            self.dds_module = DDS(
                num_proposal=num_proposal,
                pc_dim=d_model,
                lang_dim=proj_lang_dim,
                d_model=d_model,
                tau_start=dds_cfg.get('tau_start', 1.0),
                tau_end=dds_cfg.get('tau_end', 0.5),
                tau_anneal_steps=dds_cfg.get('tau_anneal_steps', 10000),
                diversity_weight=dds_cfg.get('diversity_weight', 0.1),
                use_diversity_loss=dds_cfg.get('use_diversity_loss', True),

                diversity_margin=dds_cfg.get('diversity_margin', 0.1),
                diversity_focal_gamma=dds_cfg.get('diversity_focal_gamma', 2.0),
                spatial_sigma=dds_cfg.get('spatial_sigma', 0.5),
            )
        else:
            self.dds_module = None

        if pc_dim != d_model:
            self.pc_proj = nn.Sequential(
                nn.Linear(pc_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )
        else:
            self.pc_proj = nn.Identity()

        if lang_dim != proj_lang_dim:
            self.lang_proj = nn.Sequential(
                nn.Linear(lang_dim, proj_lang_dim),
                nn.LayerNorm(proj_lang_dim),
                nn.ReLU()
            )
        else:
            self.lang_proj = nn.Identity()

    def forward(
        self,
        sp_feats: torch.Tensor,
        sp_coords: torch.Tensor,
        lang_feats: torch.Tensor,
        lang_mask: torch.Tensor,
        batch_offsets: torch.Tensor = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:

        B = sp_feats.shape[0] if sp_feats.dim() == 3 else len(batch_offsets) - 1
        device = sp_feats.device

        if sp_feats.dim() == 2 and batch_offsets is not None:
            sp_feats, sp_coords, sp_mask = self._to_batch_format(
                sp_feats, sp_coords, batch_offsets
            )
        else:
            sp_mask = None

        N = sp_feats.shape[1]

        actual_fps_num = min(N, self.fps_num)

        lang_feats_proj = self.lang_proj(lang_feats)

        if self.use_tws_fps and self.tws_fps is not None:

            fps_indices, semantic_scores, soft_masks = self.tws_fps(
                sp_coords, sp_feats, lang_feats, actual_fps_num, lang_mask
            )
        else:

            fps_indices = self._standard_fps(sp_coords, actual_fps_num)
            semantic_scores = None
            soft_masks = None

        if actual_fps_num < self.fps_num:

            pad_num = self.fps_num - actual_fps_num

            repeat_times = (pad_num // actual_fps_num) + 1
            fps_indices_padded = fps_indices.repeat(1, repeat_times + 1)[:, :self.fps_num]
            fps_indices = fps_indices_padded

        fps_indices_long = fps_indices.long()
        seed_feats = torch.gather(
            sp_feats, dim=1,
            index=fps_indices_long.unsqueeze(-1).expand(-1, -1, sp_feats.shape[-1])
        )

        seed_coords = torch.gather(
            sp_coords, dim=1,
            index=fps_indices_long.unsqueeze(-1).expand(-1, -1, 3)
        )

        seed_feats = self.pc_proj(seed_feats)

        diversity_loss = None
        if self.use_dds and self.dds_module is not None:

            _, _, dds_div_loss = self.dds_module(
                seed_feats,
                lang_feats_proj,
                None,
                lang_mask,
                return_diversity_loss=True,
                pc_coords=seed_coords
            )

            diversity_loss = dds_div_loss

        output = {
            'fps_indices': fps_indices,
            'seed_feats': seed_feats,
            'seed_coords': seed_coords,
            'diversity_loss': diversity_loss,
        }

        if return_intermediate:
            output['semantic_scores'] = semantic_scores
            output['soft_fps_masks'] = soft_masks
            output['lang_feats_updated'] = lang_feats_proj

        return output

    def _standard_fps(self, coords: torch.Tensor, num_samples: int) -> torch.Tensor:

        try:
            from pointnet2.pointnet2_utils import FurthestPointSampling
            return FurthestPointSampling.apply(coords, num_samples)
        except ImportError:

            return self._fps_python(coords, num_samples)

    def _fps_python(self, coords: torch.Tensor, num_samples: int) -> torch.Tensor:

        B, N, _ = coords.shape
        device = coords.device

        indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)

        for b in range(B):

            distances = torch.full((N,), float('inf'), device=device)
            farthest = 0

            for i in range(num_samples):
                indices[b, i] = farthest
                centroid = coords[b, farthest:farthest+1]
                dist = ((coords[b] - centroid) ** 2).sum(dim=-1)
                distances = torch.min(distances, dist)
                farthest = distances.argmax().item()

        return indices

    def _to_batch_format(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        batch_offsets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[i+1] - batch_offsets[i] for i in range(B))
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()

        D = feats.shape[-1]
        device = feats.device

        batch_feats = torch.zeros(B, max_len, D, device=device)
        batch_coords = torch.zeros(B, max_len, 3, device=device)
        batch_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for i in range(B):
            start = batch_offsets[i]
            end = batch_offsets[i + 1]
            length = end - start

            batch_feats[i, :length] = feats[start:end]
            batch_coords[i, :length] = coords[start:end]
            batch_mask[i, :length] = False

        return batch_feats, batch_coords, batch_mask

class SAQG(SemanticAdaptiveQueryGeneration):

    pass

def build_saqg(cfg: Dict) -> SAQG:

    saqg_cfg = cfg.get('saqg', {})

    return SAQG(
        d_model=cfg.get('d_model', 256),
        pc_dim=cfg.get('pc_dim', None),
        lang_dim=cfg.get('lang_dim', 768),
        proj_lang_dim=cfg.get('proj_lang_dim', 256),
        dropout=cfg.get('dropout', 0.1),
        use_tws_fps=saqg_cfg.get('use_tws_fps', True),
        tws_fps_cfg=saqg_cfg.get('tws_fps', {}),
        use_dds=saqg_cfg.get('use_dds', False),
        dds_cfg=saqg_cfg.get('dds', {}),
        num_proposal=cfg.get('num_proposal', 128),
        fps_num=cfg.get('fps_num', 512)
    )

if __name__ == '__main__':
    """测试代码"""
    B, N, L = 2, 1000, 50
    d_model, lang_dim = 256, 768
    fps_num = 512

    sp_feats = torch.randn(B, N, d_model)
    sp_coords = torch.randn(B, N, 3)
    lang_feats = torch.randn(B, L, lang_dim)
    lang_mask = torch.zeros(B, L, dtype=torch.bool)
    lang_mask[:, 40:] = True

    saqg = SAQG(
        d_model=d_model,
        lang_dim=lang_dim,
        proj_lang_dim=256,
        dropout=0.1,
        use_tws_fps=True,
        tws_fps_cfg={'semantic_gamma': 0.5, 'temperature': 1.0},
        use_dds=True,
        dds_cfg={
            'tau_start': 1.0,
            'tau_end': 0.5,
            'tau_anneal_steps': 10000,
            'diversity_weight': 0.1,
            'use_diversity_loss': True,
        },
        num_proposal=128,
        fps_num=fps_num
    )

    saqg.train()
    output = saqg(sp_feats, sp_coords, lang_feats, lang_mask, return_intermediate=True)

    print("=== SAQG Output (TWS-FPS + DDS regularizer) ===")
    print(f"fps_indices: {output['fps_indices'].shape}")
    print(f"seed_feats: {output['seed_feats'].shape}")
    print(f"seed_coords: {output['seed_coords'].shape}")
    print(f"diversity_loss: {output['diversity_loss']}")

    if output.get('semantic_scores') is not None:
        print(f"semantic_scores: {output['semantic_scores'].shape}")

    loss = output['seed_feats'].sum()
    if output['diversity_loss'] is not None:
        loss = loss + output['diversity_loss']
    loss.backward()
    print("Gradient check passed!")
