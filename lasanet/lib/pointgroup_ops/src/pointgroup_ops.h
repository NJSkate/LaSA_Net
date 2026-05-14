#ifndef POINTGROUP_H
#define POINTGROUP_H
#include "datatype/datatype.h"

#include "bfs_cluster/bfs_cluster.h"
#include "roipool/roipool.h"
#include "get_iou/get_iou.h"
#include "sec_mean/sec_mean.h"

void voxelize_idx_3d( at::Tensor coords,  at::Tensor output_coords,
                   at::Tensor input_map,  at::Tensor output_map, Int batchSize, Int mode);

void voxelize_fp_feat( at::Tensor feats,
               at::Tensor output_feats,
               at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane);

void voxelize_bp_feat( at::Tensor d_output_feats,  at::Tensor d_feats,  at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane);

void point_recover_fp_feat( at::Tensor feats,  at::Tensor output_feats,  at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);

void point_recover_bp_feat( at::Tensor d_output_feats,  at::Tensor d_feats,   at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);


#endif
