





#ifndef VOXELIZE_H
#define VOXELIZE_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype/datatype.h"


template <Int dimension>
void voxelize_idx( at::Tensor coords,  at::Tensor output_coords,
                   at::Tensor input_map,  at::Tensor output_map, Int batchSize, Int mode);

template <Int dimension>
void voxelize_outputmap(long *coords, long *output_coords, Int *output_map, Int *rule, Int nOutputRows, Int maxActive);

template <Int dimension>
Int voxelize_inputmap(SparseGrids<dimension> &SGs, Int *input_map, RuleBook &rules, Int &nActive, long *coords, Int nInputRows, Int nInputColumns, Int batchSize, Int mode);


template <typename T>
void voxelize_fp( at::Tensor feats,
               at::Tensor output_feats,
               at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane);

template <typename T>
void voxelize_fp_cuda(Int nOutputRows, Int maxActive, Int nPlanes, T *feats, T *output_feats, Int *rules, bool average);



template <typename T>
void voxelize_bp( at::Tensor d_output_feats,  at::Tensor d_feats,  at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane);

template <typename T>
void voxelize_bp_cuda(Int nOutputRows, Int maxActive, Int nPlanes, T *d_output_feats, T *d_feats, Int *rules, bool average);



template <typename T>
void point_recover_fp( at::Tensor feats,  at::Tensor output_feats,  at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);


template <typename T>
void point_recover_bp( at::Tensor d_output_feats,  at::Tensor d_feats,   at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);


#endif
