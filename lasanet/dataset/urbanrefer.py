

import glob
import json
import math
import os
import os.path as osp

import numpy as np
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union
from tqdm import tqdm
from gorilla import is_main_process
import sng_parser
from torch_scatter import scatter_max, scatter_mean
from transformers import RobertaTokenizerFast

MAX_NUM_OBJ = 132

class UrbanReferDataset(Dataset):


    CLASSES = ('other', 'road', 'tree', 'building', 'transmission_line',
               'pole', 'car', 'pedestrian', 'low_vegetation')

    def __init__(self,
                 data_root,
                 prefix,
                 suffix='.pth',
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 mode=4,
                 with_elastic=True,
                 aug=False,
                 use_xyz=True,
                 logger=None,
                 max_des_len=78,
                 lang_num_max=16,
                 src_sample=-1,
                 dataset='urbanrefer',
                 feat_2d_type='dinov3-16plus',
                 **kwargs
                 ):
        self.data_root = data_root
        self.dataset = dataset
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.aug = aug
        self.use_xyz = use_xyz
        self.logger = logger
        self.max_des_len = max_des_len
        self.src_sample = src_sample
        self.feat_2d_type = feat_2d_type

        self.feat_2d_paths = {
            'dinov3-16plus': os.path.join(data_root, 'dino-featv3-h-outdoor'),
        }
        if is_main_process() and logger:
            logger.info(f'使用2D特征: {feat_2d_type} (路径: {self.feat_2d_paths[feat_2d_type]})')

        np.random.seed(1999)

        self.sp_filenames = self.get_sp_filenames()

        refer_path = os.path.join(self.data_root, self.dataset, 'refer', f'refer_{prefix}.json')

        if os.path.exists(refer_path):
            with open(refer_path, 'r', encoding='utf-8') as f:
                self.scanrefer = json.load(f)
            if is_main_process() and logger:
                logger.info(f'Load {prefix} {self.dataset}: {len(self.scanrefer)} samples')
        else:
            urbanrefer_json = os.path.join(self.data_root, self.dataset, 'UrbanRefer_new.json')
            if os.path.exists(urbanrefer_json):
                with open(urbanrefer_json, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                self.scanrefer = self._filter_by_available_scenes(all_data)
                if is_main_process() and logger:
                    logger.info(f'Load {prefix} {self.dataset} (from full): {len(self.scanrefer)} samples')
            else:
                raise FileNotFoundError(f"未找到引用表达式文件: {refer_path} 或 {urbanrefer_json}")

        self.scanrefer.sort(key=lambda x: x['scene_id'])
        scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        scanrefer_new = []
        scanrefer_new_scene = []
        scene_id = ""
        for data in self.scanrefer:
            if data["scene_id"] in scene_list:
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_new_scene) > 0:
                        scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                if len(scanrefer_new_scene) >= lang_num_max:
                    scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                scanrefer_new_scene.append(data)
        if len(scanrefer_new_scene) > 0:
            scanrefer_new.append(scanrefer_new_scene)

        self.scene_inputs = scanrefer_new

        self.scene_graphs = self.load_scene_graphs()

        self.type2class = {
            'other': 0, 'road': 1, 'tree': 2, 'building': 3,
            'transmission line': 4, 'pole': 5, 'car': 6,
            'pedestrian': 7, 'low vegetation': 8
        }
        self.class2type = {v: k for k, v in self.type2class.items()}

        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        try:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                './backbones/roberta-base/', local_files_only=True
            )
        except Exception:
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    def _filter_by_available_scenes(self, all_data):
        available_scenes = set()
        for fn in self.sp_filenames:
            scene_name = os.path.splitext(os.path.basename(fn))[0].replace('_refer', '')
            available_scenes.add(scene_name)
        return [item for item in all_data if item['scene_id'] in available_scenes]

    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.data_root, self.dataset, self.prefix, '*_refer.pth'))
        if len(filenames) == 0:
            filenames = glob.glob(osp.join(self.data_root, self.dataset, 'points', '*_refer.pth'))
        if len(filenames) == 0:
            print(f"警告: 未找到超点文件，请先运行预处理脚本")
        filenames = sorted(filenames)
        return filenames

    def load_scene_graphs(self):
        scene_graphs = {}
        scene_graphs_path = os.path.join(
            self.data_root, self.dataset, f'scene_graphs_{self.prefix}.json'
        )
        if os.path.exists(scene_graphs_path):
            scene_graphs = json.load(open(scene_graphs_path, 'r', encoding='utf-8'))
        else:
            print(f'Begin {self.prefix} text decoupling (scene graphs)...')
            for data in tqdm(self.scanrefer):
                scene_id = str(data['scene_id'])
                ann_id = str(data['ann_id'])
                if scene_id not in scene_graphs:
                    scene_graphs[scene_id] = {}
                if ann_id not in scene_graphs[scene_id]:
                    scene_graphs[scene_id][ann_id] = {}
                if 'description' in data:
                    caption = data['description']
                elif 'token' in data:
                    caption = ' '.join(data['token']) if isinstance(data['token'], list) else data['token']
                else:
                    caption = "This is an object."
                scene_graphs[scene_id][ann_id] = Scene_graph_parse(caption, scanrefer=False)
            print(f'Saving {self.prefix} text decoupling (scene graphs)...')
            scene_graphs_dir = os.path.dirname(scene_graphs_path)
            if scene_graphs_dir and not os.path.exists(scene_graphs_dir):
                os.makedirs(scene_graphs_dir, exist_ok=True)
            with open(scene_graphs_path, 'w', encoding='utf-8') as f:
                json.dump(scene_graphs, f, indent=2, ensure_ascii=False)
            print(f'Done {self.prefix} text decoupling (scene graphs).')
        return scene_graphs

    def load(self, filename):
        if self.with_label:
            xyz, rgb, sem_label, inst_label, superpoint = torch.load(filename)
            return xyz, rgb, superpoint, sem_label, inst_label
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label

    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label):
        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        valid_idxs = self.sample_rdn(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        unique, superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, unique

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        if self.src_sample > 0:
            valid_idxs = np.random.choice(
                xyz.shape[0], self.src_sample,
                replace=xyz.shape[0] < self.src_sample)
        else:
            valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        unique, superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, unique

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])
        return np.matmul(xyz, m)

    def sample_rdn(self, xyz):
        if xyz.shape[0] > self.voxel_cfg.max_npoint:
            valid_idxs = np.random.choice(
                xyz.shape[0], self.voxel_cfg.max_npoint,
                replace=xyz.shape[0] < self.voxel_cfg.max_npoint)
            return valid_idxs
        else:
            return np.ones(xyz.shape[0], dtype=bool)

    def elastic(self, xyz, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3
        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        max_grid_dim = 128
        if bb.max() > max_grid_dim:
            factor = int(np.ceil(bb.max() / max_grid_dim))
            gran = gran * factor
            bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])
        return xyz + g(xyz) * mag

    def get_ref_mask(self, coord_float, instance_label, superpoint, object_ids):
        ref_lbl = torch.zeros_like(instance_label)
        for object_id in object_ids:
            ref_lbl[instance_label == object_id] = 1
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask

    def get_sp_label(self, superpoint, instance_label, semantic_label):
        sp_ins_label = torch.zeros(len(superpoint.unique()))
        sp_ins_label[:] = 199
        sem2ins = {}
        for instance in instance_label.unique():
            ref_lbl = instance_label == instance
            gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
            gt_spmask = gt_spmask > 0.5
            sp_ins_label[gt_spmask] = instance
        for semantic in semantic_label.unique().tolist():
            ref_lbl = semantic_label == semantic
            sem2ins[semantic] = instance_label[ref_lbl].unique()
        return sp_ins_label, sem2ins

    def __len__(self):
        return len(self.scene_inputs)

    def __getitem__(self, index):
        ann_ids, object_idss, gt_pmasks, gt_spmasks = [], [], [], []
        sp_ref_masks, lang_utterances, dense_maps = [], [], []
        meta_datas, view_dependents = [], []

        scene_input = self.scene_inputs[index]

        for i in range(len(scene_input)):
            data = scene_input[i]
            scan_id = data['scene_id']

            if i == 0:
                sp_filename = None
                for fn in self.sp_filenames:
                    if scan_id in fn:
                        sp_filename = fn
                        break
                if sp_filename is None:
                    raise FileNotFoundError(f"未找到场景 {scan_id} 的超点文件")

                scene = self.load(sp_filename)
                scene = self.transform_train(*scene) if self.training else self.transform_test(*scene)

                xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, unique_sp = scene

                coord = torch.from_numpy(xyz).long()
                coord_float = torch.from_numpy(xyz_middle).float()
                feat = torch.from_numpy(rgb).float()
                superpoint = torch.from_numpy(superpoint)
                semantic_label = torch.from_numpy(semantic_label).long()
                instance_label = torch.from_numpy(instance_label).long()
                sp_ins_label, _ = self.get_sp_label(superpoint, instance_label, semantic_label)

                feat_2d_path = os.path.join(self.feat_2d_paths[self.feat_2d_type], scan_id + '.pth')
                feat_2d = torch.load(feat_2d_path, map_location='cpu')[torch.from_numpy(unique_sp).long()]

            if 'object_ids' in data:
                object_ids = data['object_ids']
            else:
                object_ids = [int(data['object_id'])]
            ann_id = int(data['ann_id'])

            if 'eval_type' in data:
                meta_data = {k: data[k] for k in ('eval_type', 'spatial', 'color',
                             'texture', 'shape') if k in data}
            else:
                meta_data = None
            view_dependent = None

            anno = self.scene_graphs[scan_id][str(ann_id)]
            auxi_box = None

            tokens_positive, positive_map, modify_positive_map, pron_positive_map,\
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map_by_parse(anno, auxi_box)

            dense_map = {
                'tokens_positive': torch.from_numpy(tokens_positive).float(),
                'positive_map': torch.from_numpy(positive_map).float(),
                'modify_positive_map': torch.from_numpy(modify_positive_map).float(),
                'pron_positive_map': torch.from_numpy(pron_positive_map).float(),
                'other_entity_map': torch.from_numpy(other_entity_map).float(),
                'auxi_entity_positive_map': torch.from_numpy(auxi_entity_positive_map).float(),
                'rel_positive_map': torch.from_numpy(rel_positive_map).float(),
            }

            lang_utterance = ' '.join(anno['utterance'].replace(',', ' ,').split()) + ' . not mentioned'

            point_ref_mask = np.zeros(instance_label.shape[0])
            for object_id in object_ids:
                point_ref_mask[instance_label == object_id] = 1

            sp_ref_mask = torch_scatter.scatter_mean(
                torch.from_numpy(point_ref_mask).float(), superpoint, dim=-1
            )
            gt_pmask, gt_spmask = self.get_ref_mask(coord_float, instance_label, superpoint, object_ids)

            lang_utterances.append(lang_utterance)
            ann_ids.append(ann_id)
            object_idss.append(object_ids)
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            meta_datas.append(meta_data)
            view_dependents.append(view_dependent)
            dense_maps.append(dense_map)

        return (ann_ids, scan_id, coord, coord_float, feat, superpoint, object_idss,
                gt_pmasks, gt_spmasks, sp_ref_masks, lang_utterances, sp_ins_label,
                meta_datas, dense_maps, view_dependents, feat_2d)

    def collate_fn(self, batch):
        ann_ids, scan_ids, coords, coords_float, feats, superpoints = [], [], [], [], [], []
        object_idss, gt_pmasks, gt_spmasks, sp_ref_masks = [], [], [], []
        lang_tokenss, lang_masks, lang_utterances = [], [], []
        sp_ins_labels, meta_datas, dense_mapss = [], [], []
        batch_offsets = [0]
        scenes_len = []
        superpoint_bias = 0
        view_dependents = []
        feats_2d = []

        for i, data in enumerate(batch):
            (ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_ids,
             gt_pmask, gt_spmask, sp_ref_mask, lang_utterance, sp_ins_label,
             meta_data, dense_maps, view_dependent, feat_2d_item) = data

            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            scenes_len.append(len(ann_id))
            batch_offsets.append(superpoint_bias)

            ann_ids.extend(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            sp_ins_labels.append(sp_ins_label)

            object_idss.extend(object_ids)

            gt_pmasks.extend(gt_pmask)
            gt_spmasks.extend(gt_spmask)
            sp_ref_masks.extend(sp_ref_mask)

            meta_datas.extend(meta_data)
            view_dependents.extend(view_dependent)

            feats_2d.append(feat_2d_item)

            lang_utterances.extend(lang_utterance)
            dense_mapss.extend(dense_maps)

        token_dict = self.tokenizer.batch_encode_plus(
            lang_utterances, padding="longest", return_tensors="pt"
        )
        lang_tokenss = token_dict['input_ids']
        lang_masks = token_dict['attention_mask']

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
        scenes_len = torch.tensor(scenes_len, dtype=torch.int)
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)
        feats = torch.cat(feats, 0)
        superpoints = torch.cat(superpoints, 0).long()
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)

        feats_2d = torch.cat(feats_2d, dim=0)

        spatial_shape = np.clip(
            (coords.max(0)[0][1:] + 1).numpy(),
            self.voxel_cfg.spatial_shape[0],
            None
        )
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
            coords, len(batch), self.mode
        )

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_idss': object_idss,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            'lang_tokenss': lang_tokenss,
            'lang_masks': lang_masks,
            'coords_float': coords_float,
            'sp_ins_labels': sp_ins_labels,
            'scenes_len': scenes_len,
            'meta_datas': meta_datas,
            'dense_maps': dense_mapss,
            'view_dependents': view_dependents,
            'feats_2d': feats_2d,
        }

    def _get_token_positive_map_by_parse(self, anno, auxi_box):
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        graph_node = anno["graph_node"]
        graph_edge = anno["graph_edge"]

        target_char_span = np.zeros((MAX_NUM_OBJ, 2))
        modify_char_span = np.zeros((MAX_NUM_OBJ, 2))
        pron_char_span = np.zeros((MAX_NUM_OBJ, 2))
        rel_char_span = np.zeros((MAX_NUM_OBJ, 2))

        assert graph_node[0]['node_id'] == 0
        main_entity_target = graph_node[0]['target_char_span']
        main_entity_modify = graph_node[0]['mod_char_span']
        main_entity_pron = graph_node[0]['pron_char_span']
        main_entity_rel = graph_node[0]['rel_char_span']

        other_target_char_span = np.zeros((MAX_NUM_OBJ, 2))
        other_entity_target = []
        if len(graph_node) > 1:
            for node in graph_node:
                if node["node_id"] != 0 and node["node_type"] == "Object":
                    for span in node['target_char_span']:
                        other_entity_target.append(span)

        num_t, num_m, num_p, num_o, num_r = 0, 0, 0, 0, 0
        for target in main_entity_target:
            if len(target) == 2:
                target_char_span[num_t] = target
                num_t += 1
        for modify in main_entity_modify:
            if len(modify) == 2:
                modify_char_span[num_m] = modify
                num_m += 1
        for pron in main_entity_pron:
            if len(pron) == 2:
                pron_char_span[num_p] = pron
                num_p += 1
        for other in other_entity_target:
            if len(other) == 2:
                other_target_char_span[num_o] = other
                num_o += 1
        for rel in main_entity_rel:
            if len(rel) == 2:
                rel_char_span[num_r] = rel
                num_r += 1

        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )

        target_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))

        gt_map_t = get_positive_map(tokenized, target_char_span[:num_t])
        gt_map_m = get_positive_map(tokenized, modify_char_span[:num_m])
        gt_map_p = get_positive_map(tokenized, pron_char_span[:num_p])
        gt_map_o = get_positive_map(tokenized, other_target_char_span[:num_o])
        gt_map_r = get_positive_map(tokenized, rel_char_span[:num_r])

        gt_map_t = gt_map_t.sum(axis=0)
        gt_map_m = gt_map_m.sum(axis=0)
        gt_map_p = gt_map_p.sum(axis=0)
        gt_map_o = gt_map_o.sum(axis=0)
        gt_map_r = gt_map_r.sum(axis=0)

        target_positive_map[:1] = gt_map_t
        modify_positive_map[:1] = gt_map_m
        pron_positive_map[:1] = gt_map_p
        other_entity_positive_map[:1] = gt_map_o
        rel_positive_map[:1] = gt_map_r

        auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        if auxi_box is not None:
            auxi_entity = anno["auxi_entity"]['target_char_span']
            num_a = 0
            auxi_entity_char_span = np.zeros((MAX_NUM_OBJ, 2))
            for a, auxi in enumerate(auxi_entity):
                auxi_entity_char_span[a] = auxi
                num_a = a + 1
            gt_map_a = get_positive_map(tokenized, auxi_entity_char_span[:num_a])
            gt_map_a = gt_map_a.sum(axis=0)
            auxi_entity_positive_map[:1] = gt_map_a

        return (target_char_span, target_positive_map, modify_positive_map,
                pron_positive_map, other_entity_positive_map, auxi_entity_positive_map,
                rel_positive_map)

def get_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except Exception:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except Exception:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)
    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()

def Scene_graph_parse(caption, scanrefer=False):
    caption = ' '.join(caption.replace(',', ' , ').split())
    caption = ' '.join(caption.replace("'m", "am").split())
    caption = ' '.join(caption.replace("'s", "is").split())
    caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
    caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
    caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
    caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
    caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
    caption = ' '.join(caption.replace("2-door", "2 - door").split())
    caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
    caption = ' '.join(caption.replace("computer/", "computer /").split())
    caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
    caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
    caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
    caption = ' '.join(caption.replace("theses", "these").split())

    if not scanrefer:
        caption = ' '.join(caption.replace('.', ' .').split())
        caption = ' '.join(caption.replace(';', ' ; ').split())
        caption = ' '.join(caption.replace('-', ' ').split())
        caption = ' '.join(caption.replace('"', ' ').split())
        caption = ' '.join(caption.replace('?', ' ').split())
        caption = ' '.join(caption.replace("*", " ").split())
        caption = ' '.join(caption.replace(':', ' ').split())
        caption = ' '.join(caption.replace('$', ' ').split())
        caption = ' '.join(caption.replace("#", " ").split())
        caption = ' '.join(caption.replace("/", " / ").split())
        caption = ' '.join(caption.replace("you're", "you are").split())
        caption = ' '.join(caption.replace("isn't", "is not").split())
        caption = ' '.join(caption.replace("thats", "that is").split())
        caption = ' '.join(caption.replace("theres", "there is").split())
        caption = ' '.join(caption.replace("doesn't", "does not").split())
        caption = ' '.join(caption.replace("doesnt", "does not").split())
        caption = ' '.join(caption.replace("itis", "it is").split())
        caption = ' '.join(caption.replace("left-hand", "left - hand").split())
        caption = ' '.join(caption.replace("[", " [ ").split())
        caption = ' '.join(caption.replace("]", " ] ").split())
        caption = ' '.join(caption.replace("(", " ( ").split())
        caption = ' '.join(caption.replace(")", " ) ").split())
        caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
        caption = ' '.join(caption.replace(";s", "is").split())
        caption = ' '.join(caption.replace("tha=e", "the").split())
        caption = ' '.join(caption.replace("it's", "it is").split())
        caption = ' '.join(caption.replace("'s", " is").split())
        caption = ' '.join(caption.replace("isnt", "is not").split())
        caption = ' '.join(caption.replace("Don't", "Do not").split())
        caption = ' '.join(caption.replace("arent", "are not").split())
        caption = ' '.join(caption.replace("cant", "can not").split())
        caption = ' '.join(caption.replace("you're", "you are").split())
        caption = ' '.join(caption.replace('!', ' !').split())
        caption = ' '.join(caption.replace('id the', ' , the').split())
        caption = ' '.join(caption.replace('youre', 'you are').split())
        caption = ' '.join(caption.replace("'", ' ').split())
        caption = ' '.join(caption.replace("``", ' ').split())

    caption = caption.strip()
    if not caption or len(caption) == 0:
        caption = "This is an object."
    if len(caption) > 0 and caption[0] == "'":
        caption = caption[1:]
    if len(caption) > 0 and caption[-1] == "'":
        caption = caption[:-1]
    if not caption or len(caption) == 0:
        caption = "This is an object."

    graph_node, graph_edge = sng_parser.parse(caption)
    if (len(graph_node) < 1) or\
        (len(graph_node) > 0 and graph_node[0]["node_id"] != 0):
        caption = "This is an object . " + caption
        graph_node, graph_edge = sng_parser.parse(caption)

    auxi_entity = None
    for node in graph_node:
        if (node["node_id"] != 0) and (node["node_type"] == "Object"):
            auxi_entity = node
            break

    return {
        "graph_node": graph_node,
        "graph_edge": graph_edge,
        "auxi_entity": auxi_entity,
        "utterance": caption
    }
