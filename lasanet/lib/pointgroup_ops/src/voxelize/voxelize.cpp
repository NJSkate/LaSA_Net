





#include "voxelize.h"


template <Int dimension>
void voxelize_idx( at::Tensor coords,  at::Tensor output_coords,
                   at::Tensor input_map,  at::Tensor output_map, Int batchSize, Int mode){
    assert(coords.ndimension() == 2);
    assert(coords.size(1) >= dimension and coords.size(1) <= dimension + 1);

    RuleBook voxelizeRuleBook;
    SparseGrids<dimension> inputSGs;
    Int nActive = 0;

    Int maxActive = voxelize_inputmap<dimension>(inputSGs, input_map.data<Int>(), voxelizeRuleBook, nActive, coords.data<long>(), coords.size(0), coords.size(1), batchSize, mode);

    output_map.resize_({nActive, maxActive + 1});
    output_map.zero_();

    output_coords.resize_({nActive, coords.size(1)});
    output_coords.zero_();

    Int *oM = output_map.data<Int>();
    long *oC = output_coords.data<long>();
    voxelize_outputmap<dimension>(coords.data<long>(), oC, oM, &voxelizeRuleBook[1][0], nActive, maxActive);
}


template <Int dimension>
void voxelize_outputmap(long *coords, long *output_coords, Int *output_map, Int *rule, Int nOutputRows, Int maxActive){
    for(Int i = 0; i < nOutputRows; i++){
        for(Int j = 0; j <= maxActive; j++)
            output_map[j] = rule[j];
        Int inputIdx = rule[1];
        rule += (1 + maxActive);
        output_map += (1 + maxActive);

        long *coord = coords + inputIdx * (dimension + 1);
        long *output_coord  = output_coords + i * (dimension + 1);
        for(Int j = 0; j <= dimension; j++){
            output_coord[j] = coord[j];
        }
    }
}








template <Int dimension>
Int voxelize_inputmap(SparseGrids<dimension> &SGs, Int *input_map, RuleBook &rules, Int &nActive, long *coords, Int nInputRows, Int nInputColumns, Int batchSize, Int mode){
    assert(nActive == 0);
    assert(rules.size() == 0);
    assert(SGs.size() == 0);

    SGs.resize(batchSize);
    Point<dimension> p;

    std::vector<std::vector<Int>> outputRows;
    if(nInputColumns == dimension){
        SGs.resize(1);
        auto &sg = SGs[0];
        for(Int i = 0; i < nInputRows; i++){
            for(Int j = 0; j < dimension; j++)
                p[j] = coords[j];
            coords += dimension;
            auto iter = sg.mp.find(p);
            if (iter == sg.mp.end()){
                sg.mp[p] = nActive++;
                outputRows.resize(nActive);
            }
            outputRows[sg.mp[p]].push_back(i);

            input_map[i] = sg.mp[p];
        }
    }
    else{
        Int batchIdx;
        for(Int i = 0; i < nInputRows; i++){
            batchIdx = coords[0];
            for(Int j = 0; j < dimension; j++)
                p[j] = coords[j + 1];
            coords += (dimension + 1);
            if(batchIdx + 1 >= (Int)SGs.size()){
                SGs.resize(batchIdx + 1);
            }
            auto &sg = SGs[batchIdx];
            auto iter = sg.mp.find(p);
            if(iter == sg.mp.end()){
                sg.mp[p] = nActive++;
                outputRows.resize(nActive);
            }
            outputRows[sg.mp[p]].push_back(i);

            input_map[i] = sg.mp[p];
        }
    }







    rules.resize(2);
    rules[0].push_back(mode);
    rules[0].push_back(1);
    rules[0].push_back(nInputRows);
    rules[0].push_back(outputRows.size());
    auto &rule = rules[1];
    if(mode == 0){
        assert(nInputRows == (Int)outputRows.size());
        for(Int i = 0; i < nActive; i++){
            rule.push_back(1);
            assert((Int)outputRows[i].size() == 1);
            rule.push_back(outputRows[i][0]);
        }
    }
    if(mode == 1){
        for(Int i = 0; i < nActive; i++){
            rule.push_back(1);
            rule.push_back(outputRows[i].front());
        }
    }
    if(mode == 2){
        for(Int i = 0; i < nActive; i++){
            rule.push_back(1);
            rule.push_back(outputRows[i].back());
        }
    }
    Int maxActive = 1;
    if(mode == 3 or mode == 4){
        for(auto &row: outputRows)
            maxActive = std::max(maxActive, (Int)row.size());
        rules[0][1] = maxActive;
        for(auto &row: outputRows){
            rule.push_back(row.size());
            for(auto &r: row)
                rule.push_back(r);
            rule.resize((rule.size() + maxActive) / (maxActive + 1) * (maxActive + 1));
        }
    }
    return maxActive;
}



template <typename T>
void voxelize_fp( at::Tensor feats,
               at::Tensor output_feats,
               at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane){

    auto iF = feats.data<T>();
    auto oF = output_feats.data<T>();

    Int *rules = output_map.data<Int>();

    voxelize_fp_cuda<T>(nActive, maxActive, nPlane, iF, oF, rules, mode==4);
}

template <typename T>
void voxelize_bp( at::Tensor d_output_feats,  at::Tensor d_feats,  at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane){
    auto d_oF = d_output_feats.data<T>();
    auto d_iF = d_feats.data<T>();

    Int *rules = output_map.data<Int>();

    voxelize_bp_cuda<T>(nActive, maxActive, nPlane, d_oF, d_iF, rules, mode==4);
}


template <typename T>
void point_recover_fp( at::Tensor feats,  at::Tensor output_feats,  at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    auto iF = feats.data<T>();
    auto oF = output_feats.data<T>();

    Int *rules = idx_map.data<Int>();

    voxelize_bp_cuda<T>(nActive, maxActive, nPlane, iF, oF, rules, false);
}


template <typename T>
void point_recover_bp( at::Tensor d_output_feats,  at::Tensor d_feats,   at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    auto d_oF = d_output_feats.data<T>();
    auto d_iF = d_feats.data<T>();

    Int *rules = idx_map.data<Int>();

    voxelize_fp_cuda<T>(nActive, maxActive, nPlane, d_oF, d_iF, rules, false);
}
