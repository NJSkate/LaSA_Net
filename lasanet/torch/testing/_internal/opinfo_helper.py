import collections
import warnings
from functools import partial

import torch
from torch.testing._internal.common_cuda import (TEST_CUDA)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    all_types_and_complex,
    all_types_and_half,
    all_types,
    complex_types,
    floating_and_complex_types,
    floating_types_and_half,
    floating_types,
    integral_types,
    floating_types_and,
    floating_and_complex_types_and,
    integral_types_and,
    all_types_and,
    _dispatch_dtypes,
)

COMPLETE_DTYPES_DISPATCH = (
    all_types,
    all_types_and_complex,
    all_types_and_half,
    floating_types,
    floating_and_complex_types,
    floating_types_and_half,
    integral_types,
    complex_types,
)

EXTENSIBLE_DTYPE_DISPATCH = (
    all_types_and_complex_and,
    floating_types_and,
    floating_and_complex_types_and,
    integral_types_and,
    all_types_and,
)

DEVICES = ['cpu'] + (['cuda'] if TEST_CUDA else [])

class _dynamic_dispatch_dtypes(_dispatch_dtypes):

    pass

def get_supported_dtypes(op, sample_inputs_fn, device_type):

    assert device_type in ['cpu', 'cuda']
    if not TEST_CUDA and device_type == 'cuda':
        warnings.warn("WARNING: CUDA is not available, empty_dtypes dispatch will be returned!")
        return _dynamic_dispatch_dtypes(())

    supported_dtypes = set()
    for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
        try:
            samples = sample_inputs_fn(op, device_type, dtype, False)
        except RuntimeError:

            warnings.warn(f"WARNING: Unable to generate sample for device:{device_type} and dtype:{dtype}")
            continue

        supported = True
        for sample in samples:
            try:
                op(sample.input, *sample.args, **sample.kwargs)
            except RuntimeError as re:

                supported = False
                break

        if supported:
            supported_dtypes.add(dtype)

    return _dynamic_dispatch_dtypes(supported_dtypes)

def dtypes_dispatch_hint(dtypes):

    return_type = collections.namedtuple('return_type', 'dispatch_fn dispatch_fn_str')

    if len(dtypes) == 0:
        return return_type((), str(tuple()))

    set_dtypes = set(dtypes)
    for dispatch in COMPLETE_DTYPES_DISPATCH:

        if set(dispatch()) == set_dtypes:
            return return_type(dispatch, dispatch.__name__ + "()")

    chosen_dispatch = None
    chosen_dispatch_score = 0.
    for dispatch in EXTENSIBLE_DTYPE_DISPATCH:
        dispatch_dtypes = set(dispatch())
        if not dispatch_dtypes.issubset(set_dtypes):
            continue

        score = len(dispatch_dtypes)
        if score > chosen_dispatch_score:
            chosen_dispatch_score = score
            chosen_dispatch = dispatch

    if chosen_dispatch is None:
        return return_type((), str(dtypes))

    return return_type(partial(dispatch, *tuple(set(dtypes) - set(dispatch()))),
                       dispatch.__name__ + str(tuple(set(dtypes) - set(dispatch()))))

def is_dynamic_dtype_set(op):

    return op.dynamic_dtypes

def str_format_dynamic_dtype(op):
    fmt_str = """
        OpInfo({name},
               dtypes={dtypesIfCPU},
               dtypesIfCUDA={dtypesIfCUDA},
        )
        """.format(name=op.name,
                   dtypesIfCPU=dtypes_dispatch_hint(op.dtypesIfCPU).dispatch_fn_str,
                   dtypesIfCUDA=dtypes_dispatch_hint(op.dtypesIfCUDA).dispatch_fn_str)

    return fmt_str
