"""
The testing package contains testing-specific utilities.
"""

import torch
import random
import math
import cmath
from typing import List, Optional, Tuple, Union
import operator

FileCheck = torch._C.FileCheck

__all__ = [
    "FileCheck",
    "get_all_device_types",
    "make_non_contiguous",
]

def is_integral(dtype: torch.dtype) -> bool:
    return dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

def is_quantized(dtype: torch.dtype) -> bool:
    return dtype in (torch.quint8, torch.qint8, torch.qint32, torch.quint4x2)

def _unravel_index(flat_index, shape):
    flat_index = operator.index(flat_index)
    res = []

    if shape == torch.Size([]):
        return 0

    for size in shape[::-1]:
        res.append(flat_index % size)
        flat_index = flat_index // size

    if len(res) == 1:
        return res[0]

    return tuple(res[::-1])

_compare_return_type = Tuple[bool, Optional[str]]

def _compare_tensors_internal(a: torch.Tensor, b: torch.Tensor, *, rtol, atol, equal_nan) -> _compare_return_type:
    debug_msg : Optional[str]

    if (
        (is_integral(a.dtype) and rtol == 0 and atol < 1)
        or a.dtype is torch.bool
        or is_quantized(a.dtype)
    ):
        if (a == b).all().item():
            return (True, None)

        identity_mask = a != b
        a_flat = a.to(torch.long).flatten()
        b_flat = b.to(torch.long).flatten()
        count_non_identical = torch.sum(identity_mask, dtype=torch.long)
        diff = torch.abs(a_flat - b_flat)
        greatest_diff_index = torch.argmax(diff)
        debug_msg = ("Found {0} different element(s) (out of {1}), with the greatest "
                     "difference of {2} ({3} vs. {4}) occuring at index "
                     "{5}.".format(count_non_identical.item(),
                                   a.numel(),
                                   diff[greatest_diff_index],
                                   a_flat[greatest_diff_index],
                                   b_flat[greatest_diff_index],
                                   _unravel_index(greatest_diff_index, a.shape)))
        return (False, debug_msg)

    if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
        return (True, None)

    a_flat = a.to(torch.float64 if not a.dtype.is_complex else torch.complex128).flatten()
    b_flat = b.to(torch.float64 if not a.dtype.is_complex else torch.complex128).flatten()
    diff = torch.abs(a_flat - b_flat)

    close = torch.isclose(a_flat, b_flat, rtol, atol, equal_nan)
    diff[close] = 0
    nans = torch.isnan(diff)
    num_nans = nans.sum()

    outside_range = (diff > (atol + rtol * torch.abs(b_flat))) | (diff == math.inf)
    count_outside_range = torch.sum(outside_range, dtype=torch.long)
    greatest_diff_index = torch.argmax(diff)
    debug_msg = ("With rtol={0} and atol={1}, found {2} element(s) (out of {3}) whose "
                 "difference(s) exceeded the margin of error (including {4} nan comparisons). "
                 "The greatest difference was {5} ({6} vs. {7}), which "
                 "occurred at index {8}.".format(rtol, atol,
                                                 count_outside_range + num_nans,
                                                 a.numel(),
                                                 num_nans,
                                                 diff[greatest_diff_index],
                                                 a_flat[greatest_diff_index],
                                                 b_flat[greatest_diff_index],
                                                 _unravel_index(greatest_diff_index, a.shape)))
    return (False, debug_msg)

def _compare_scalars_internal(a, b, *, rtol: float, atol: float, equal_nan: Union[str, bool]) -> _compare_return_type:
    def _helper(a, b, s) -> _compare_return_type:

        if a == b or ((equal_nan in {"relaxed", True}) and a != a and b != b):
            return (True, None)

        if not (equal_nan in {"relaxed", True}) and (a != a or b != b):
            msg = ("Found {0} and {1} while comparing" + s + "and either one "
                   "is nan and the other isn't, or both are nan and "
                   "equal_nan is False").format(a, b)
            return (False, msg)

        diff = abs(a - b)
        allowed_diff = atol + rtol * abs(b)
        result = diff <= allowed_diff

        if ((cmath.isinf(a) or cmath.isinf(b)) and a != b):
            result = False

        msg = None
        if not result:
            if rtol == 0 and atol == 0:
                msg = f"{a} != {b}"
            else:
                msg = (
                    f"Comparing{s}{a} and {b} gives a "
                    f"difference of {diff}, but the allowed difference "
                    f"with rtol={rtol} and atol={atol} is "
                    f"only {allowed_diff}!"
                )
        return result, msg

    return _helper(a, b, " ")

def make_non_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() <= 1:
        return tensor.clone()
    osize = list(tensor.size())

    for _ in range(2):
        dim = random.randint(0, len(osize) - 1)
        add = random.randint(4, 15)
        osize[dim] = osize[dim] + add

    input = tensor.new(torch.Size(osize + [random.randint(2, 3)]))
    input = input.select(len(input.size()) - 1, random.randint(0, 1))

    for i in range(len(osize)):
        if input.size(i) != tensor.size(i):
            bounds = random.randint(1, input.size(i) - tensor.size(i))
            input = input.narrow(i, bounds, tensor.size(i))

    input.copy_(tensor)

    return input.data

def get_all_device_types() -> List[str]:
    return ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
