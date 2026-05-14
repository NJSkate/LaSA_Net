import torch
import contextlib
from typing import Iterator

@contextlib.contextmanager
def enable_python_mode(cls) -> Iterator[None]:
    if not hasattr(cls, '__torch_dispatch__'):
        raise ValueError('The class passed to enable_python_mode '
                         'must have a __torch_dispatch__ classmethod')
    if not isinstance(cls, type) or not issubclass(cls, (torch.Tensor,)):
        raise ValueError('The argument passed to enable_python_mode '
                         'must be the type of a Tensor subclass')
    torch._C._enter_python_mode(cls)
    try:
        yield
    finally:
        torch._C._exit_python_mode()
