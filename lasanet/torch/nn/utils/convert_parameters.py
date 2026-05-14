import torch
from typing import Iterable, Optional

def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """

    param_device = None

    vec = []
    for param in parameters:

        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    param_device = None

    pointer = 0
    for param in parameters:

        param_device = _check_param_device(param, param_device)

        num_param = param.numel()

        param.data = vec[pointer:pointer + num_param].view_as(param).data

        pointer += num_param

def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device
