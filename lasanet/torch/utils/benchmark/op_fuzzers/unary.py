import numpy as np
import torch

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor

_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))

class UnaryOpFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False):
        super().__init__(
            parameters=[

                FuzzedParameter("dim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                [
                    FuzzedParameter(
                        name=f"x_step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    ) for i in range(3)
                ],

                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024**3,
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
