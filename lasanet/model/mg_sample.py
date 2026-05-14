

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionMap(nn.Module):


    def __init__(self, q_dim, k_dim, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(q_dim, d_model, bias=False)
        self.key_proj = nn.Linear(k_dim, d_model, bias=False)

    def forward(self, query, key, query_mask=None, key_padding_mask=None):

        Q = self.query_proj(query)
        K = self.key_proj(key)
        scores = torch.einsum('bnd,bld->bnl', Q, K) / math.sqrt(self.d_model)
        if query_mask is not None:
            scores = scores.masked_fill(query_mask.unsqueeze(2), float('-inf'))
        attn_map = F.softmax(scores, dim=1)
        return attn_map

class MGSamplingModule(nn.Module):


    def __init__(self, num_proposal, pc_dim, lang_dim, d_model):
        super().__init__()
        self.num_proposal = num_proposal

        self.camap = CrossAttentionMap(pc_dim, lang_dim, d_model)

        self.sent_proj = nn.Linear(pc_dim, d_model, bias=False)

        self.alpha_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, pc_feat, lang_feat, pc_mask, lang_mask, gate=None):

        if pc_mask is not None:
            pc_mask = pc_mask.bool()

        attn_map = self.camap(
            pc_feat, lang_feat, pc_mask, lang_mask.bool()
        )
        word_scores = (attn_map * (~lang_mask.unsqueeze(1)).float()).sum(-1)

        lang_valid = (~lang_mask.bool()).float().unsqueeze(-1)
        sent_emb = (lang_feat * lang_valid).sum(dim=1) / lang_valid.sum(dim=1).clamp(min=1)

        pc_proj = self.sent_proj(pc_feat)

        pc_norm = F.normalize(pc_proj, p=2, dim=-1)
        sent_norm = F.normalize(sent_emb, p=2, dim=-1)
        sent_scores = torch.einsum('bnd,bd->bn', pc_norm, sent_norm)
        sent_scores = F.relu(sent_scores)

        alpha = torch.sigmoid(self.alpha_raw)
        ref_scores = word_scores + alpha * sent_scores

        if gate is not None:
            ref_scores = ref_scores * gate

        if ref_scores.shape[1] < self.num_proposal:
            sample_inds = torch.arange(
                ref_scores.shape[1], device=ref_scores.device
            ).int().unsqueeze(0).repeat(ref_scores.shape[0], 1)
        else:
            sample_inds = torch.topk(ref_scores, self.num_proposal)[1].int()

        return sample_inds, ref_scores

if __name__ == '__main__':
    B, N, L = 2, 256, 50
    pc_dim, lang_dim, d_model = 256, 256, 256

    pc_feat = torch.randn(B, N, pc_dim).cuda()
    lang_feat = torch.randn(B, L, lang_dim).cuda()
    lang_mask = torch.zeros(B, L, dtype=torch.bool).cuda()
    lang_mask[:, 40:] = True

    module = MGSamplingModule(
        num_proposal=128, pc_dim=pc_dim, lang_dim=lang_dim, d_model=d_model
    ).cuda()

    sample_inds, ref_scores = module(pc_feat, lang_feat, None, lang_mask)
    print(f"sample_inds: {sample_inds.shape}")
    print(f"ref_scores:  {ref_scores.shape}")
    print(f"ref_scores range: [{ref_scores.min().item():.4f}, {ref_scores.max().item():.4f}]")
    print(f"alpha: {torch.sigmoid(module.alpha_raw).item():.6f}")
    print(f"Parameters: {sum(p.numel() for p in module.parameters()):,}")
