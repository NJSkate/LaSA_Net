import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools
import numpy as np

from lasanet.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou

from .dec import DEC
from transformers import RobertaModel
from pointnet2.pointnet2_utils import FurthestPointSampling

try:
    from .saqg import SAQG, build_saqg
    SAQG_AVAILABLE = True
except ImportError:
    SAQG_AVAILABLE = False
    SAQG = None

class LaSANet(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        sampling_module=None,
        dec=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        infer_mode='pos',
        fps_num=512,
        fix_module=[],
        task_type='res',

        use_saqg=False,
        saqg=None,
    ):
        super().__init__()
        self.task_type = task_type

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.use_saqg = use_saqg and SAQG_AVAILABLE
        self.saqg_module = None
        if self.use_saqg and saqg is not None:

            saqg_cfg = {
                'd_model': dec.get('d_model', 256) if dec else 256,
                'pc_dim': media,
                'lang_dim': 768,
                'proj_lang_dim': dec.get('d_model', 256) if dec else 256,
                'dropout': dec.get('dropout', 0.1) if dec else 0.1,
                'num_proposal': sampling_module.get('num_proposal', 128) if sampling_module else 128,
                'fps_num': fps_num,
                'saqg': saqg
            }
            self.saqg_module = build_saqg(saqg_cfg)

        self.decoder_param = dec
        self.fps_num = fps_num
        self.text_encoder = RobertaModel.from_pretrained('./backbones/roberta-base')

        self.sampling_module = sampling_module
        self.dec = DEC(**dec, sampling_module=sampling_module, in_channel=media)

        self.criterion = Criterion(**criterion)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.infer_mode = infer_mode
        self.init_weights()
        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(LaSANet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():

                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, **kwargs):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        lang_mask_for_saqg = ~lang_masks.bool() if lang_masks is not None else None

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d, saqg_diversity_loss = self.expand_and_fps(
            sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d,
            mode='tr', lang_feats=lang_feats, lang_mask=lang_mask_for_saqg
        )

        out = self.dec(
            sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d,
        )

        if saqg_diversity_loss is not None:
            out['diversity_loss'] = saqg_diversity_loss

        loss, loss_dict = self.criterion(out, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels, dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets)

        return loss, loss_dict

    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, **kwargs):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        lang_mask_for_saqg = ~lang_masks.bool() if lang_masks is not None else None

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d, _ = self.expand_and_fps(
            sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d,
            mode='pred', lang_feats=lang_feats, lang_mask=lang_mask_for_saqg
        )

        out = self.dec(
            sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d,
        )
        ret = self.predict_by_feat(scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp)
        if meta_datas[0] is not None:
            ret['meta_datas'] = meta_datas
            if view_dependents[0] is not None:
                ret['view_dependents'] = view_dependents
        return ret

    def predict_by_feat(self, scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp):

        spious, pious, sp_r_ious, p_r_ious, pred_pmasks, scan_idss = [], [], [], [], [], []
        seed_inds, sample_inds, indis = [], [], []
        nt_labels = []
        b = len(object_idss)
        for i in range(b):
            gt_pmask = gt_pmasks[i]
            gt_spmask = gt_spmasks[i]
            pred_spmask = out['masks'][i].squeeze()

            pred_indis = out['indis'][i]

            if self.task_type == 'gres':
                indicate = pred_indis.argmax(-1) == 1
                top = indicate.nonzero(as_tuple=False).squeeze(-1)

                is_nt = False
                if len(top)==0:
                    is_nt = True
                    piou = torch.tensor(0.0, device=pred_spmask.device)
                    spiou = torch.tensor(0.0, device=pred_spmask.device)
                    pred_spmask = torch.zeros_like(pred_spmask[0], device=pred_spmask.device)
                    pred_pmask = pred_spmask[superpoints]

                else:
                    top_mask = pred_spmask[top]
                    pred_spmask = top_mask.max(0)[0]
                    spiou = get_iou(pred_spmask, gt_spmask)
                    pred_pmask = pred_spmask[superpoints]
                    piou = get_iou(pred_pmask, gt_pmask)
                    if not is_nt:
                        pred_pmask_binay = pred_pmask.sigmoid() > 0.5
                        is_nt = True if pred_pmask_binay.sum() < 50 else False
                nt_labels.append(is_nt)
            elif self.task_type == 'res':
                softmax_indis = F.softmax(pred_indis, dim=-1)
                indicate = softmax_indis[:,1]

                top = indicate.argmax()

                pred_spmask = pred_spmask[top]
                spiou = get_iou(pred_spmask, gt_spmask)
                pred_pmask = pred_spmask[superpoints]
                piou = get_iou(pred_pmask, gt_pmask)
            else:
                raise NotImplementedError

            spious.append(spiou.cpu())
            pious.append(piou.cpu())
            pred_pmasks.append(pred_pmask.sigmoid().cpu())
            scan_idss.append(scan_ids[0])
            seed_inds.append(fps_seed_sp[i].cpu())
            sample_inds.append(out['sample_inds'][i].cpu())
            indis.append(indicate.cpu())
        gt_pmasks = [gt_pmask.cpu() for gt_pmask in gt_pmasks]
        return dict(scan_id=scan_idss, object_ids=object_idss, ann_id=ann_ids, piou=pious, spiou=spious, gt_pmask=gt_pmasks, pred_pmask=pred_pmasks,
                    sp_r_iou=sp_r_ious, p_r_iou=p_r_ious, nt_label = nt_labels, seed_ind=seed_inds, sample_ind=sample_inds, indi=indis)

    def extract_feat(self, x, superpoints, v2p_map):


        x = self.input_conv(x)
        x, mutil_x = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]

        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)
        return x

    def expand_and_fps(self, sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode=None, lang_feats=None, lang_mask=None):

        sp_ins_labels_expand = []
        has_2d = feats_2d is not None

        saqg_diversity_losses = []

        batch_offsets_expand = batch_offsets[0:1]
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            if i==0:
                sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
                if has_2d:
                    feats_2d_expand = feats_2d[s:e].repeat(scenes_len[i],1)

                if self.use_saqg and self.saqg_module is not None and lang_feats is not None:

                    sp_feats_batch = sp_feats[s:e].unsqueeze(0)
                    sp_coords_batch = sp_coords_float[s:e].unsqueeze(0)

                    lang_feat_i = lang_feats[0:1] if lang_feats.dim() == 3 else lang_feats.unsqueeze(0)
                    lang_mask_i = lang_mask[0:1] if lang_mask is not None and lang_mask.dim() == 2 else lang_mask

                    saqg_output = self.saqg_module(
                        sp_feats_batch, sp_coords_batch, lang_feat_i, lang_mask_i
                    )
                    fps_seed_sp = saqg_output['fps_indices']

                    if saqg_output.get('diversity_loss') is not None:
                        saqg_diversity_losses.append(saqg_output['diversity_loss'])
                else:
                    fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)
            else:
                sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
                if has_2d:
                    feats_2d_expand = torch.cat((feats_2d_expand, feats_2d[s:e].repeat(scenes_len[i],1)),dim=0)

                if self.use_saqg and self.saqg_module is not None and lang_feats is not None:
                    sp_feats_batch = sp_feats[s:e].unsqueeze(0)
                    sp_coords_batch = sp_coords_float[s:e].unsqueeze(0)

                    lang_idx = min(i, lang_feats.shape[0] - 1)
                    lang_feat_i = lang_feats[lang_idx:lang_idx+1]
                    lang_mask_i = lang_mask[lang_idx:lang_idx+1] if lang_mask is not None else None

                    saqg_output = self.saqg_module(
                        sp_feats_batch, sp_coords_batch, lang_feat_i, lang_mask_i
                    )
                    fps_seed_sp = saqg_output['fps_indices']

                    if saqg_output.get('diversity_loss') is not None:
                        saqg_diversity_losses.append(saqg_output['diversity_loss'])
                else:
                    fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = torch.cat((fps_seed_sp_expand, fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)),dim=0)
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                if sp_ins_labels:
                    sp_ins_labels_expand.append(sp_ins_labels[i])

        saqg_diversity_loss = None
        if saqg_diversity_losses:
            saqg_diversity_loss = torch.stack(saqg_diversity_losses).mean()

        return sp_feats_expand, sp_coords_float_expand, fps_seed_sp_expand, batch_offsets_expand, sp_ins_labels_expand, feats_2d_expand if has_2d else None, saqg_diversity_loss
