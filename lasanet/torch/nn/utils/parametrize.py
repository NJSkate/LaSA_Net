import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor

import collections
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence

_cache_enabled = 0
_cache: Dict[Tuple[int, str], Optional[Tensor]] = {}

@contextmanager
def cached():
    r"""Context manager that enables the caching system within parametrizations
    registered with :func:`register_parametrization`.

    The value of the parametrized objects is computed and cached the first time
    they are required when this context manager is active. The cached values are
    discarded when leaving the context manager.

    This is useful when using a parametrized parameter more than once in the forward pass.
    An example of this is when parametrizing the recurrent kernel of an RNN or when
    sharing weights.

    The simplest way to activate the cache is by wrapping the forward pass of the neural network

    .. code-block:: python

        import torch.nn.utils.parametrize as P
        ...
        with P.cached():
            output = model(inputs)

    in training and evaluation. One may also wrap the parts of the modules that use
    several times the parametrized tensors. For example, the loop of an RNN with a
    parametrized recurrent kernel:

    .. code-block:: python

        with P.cached():
            for x in xs:
                out_rnn = self.rnn_cell(x, out_rnn)
    """
    global _cache
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache = {}

def _register_parameter_or_buffer(module, name, X):
    if isinstance(X, Parameter):
        module.register_parameter(name, X)
    else:
        module.register_buffer(name, X)

class ParametrizationList(ModuleList):
    r"""A sequential container that holds and manages the ``original`` or ``original0``, ``original1``, ...
    parameters or buffers of a parametrized :class:`torch.nn.Module`.

    It is the type of ``module.parametrizations[tensor_name]`` when ``module[tensor_name]``
    has been parametrized with :func:`register_parametrization`.

    If the first registered parmetrization has a ``right_inverse`` that returns one tensor or
    does not have a ``right_inverse`` (in which case we assume that ``right_inverse`` is the identity),
    it will hold the tensor under the name ``original``.
    If it has a ``right_inverse`` that returns more than one tensor, these will be registered as
    ``original0``, ``original1``, ...

    .. warning::
        This class is used internally by :func:`register_parametrization`. It is documented
        here for completeness. It shall not be instantiated by the user.

    Args:
        modules (sequence): sequence of modules representing the parametrizations
        original (Parameter or Tensor): parameter or buffer that is parametrized
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.
    """
    original: Tensor
    unsafe: bool

    def __init__(
        self, modules: Sequence[Module], original: Union[Tensor, Parameter], unsafe: bool = False
    ) -> None:

        if len(modules) == 0:
            raise ValueError("ParametrizationList requires one or more modules.")

        super().__init__(modules)
        self.unsafe = unsafe

        original_shape = original.shape
        original_dtype = original.dtype

        with torch.no_grad():
            new = original
            for module in reversed(self):
                if hasattr(module, "right_inverse"):
                    try:
                        new = module.right_inverse(new)
                    except NotImplementedError:
                        pass

        if not isinstance(new, Tensor) and not isinstance(new, collections.abc.Sequence):
            raise ValueError("'right_inverse' must return a Tensor or a Sequence of tensors (list, tuple...). "
                             f"Got {type(new).__name__}")

        self.is_tensor = isinstance(new, Tensor)
        self.ntensors = 1 if self.is_tensor else len(new)

        if self.is_tensor:
            if original.dtype != new.dtype:
                raise ValueError(
                    "When `right_inverse` outputs one tensor, it may not change the dtype.\n"
                    f"original.dtype: {original.dtype}\n"
                    f"right_inverse(original).dtype: {new.dtype}"
                )

            with torch.no_grad():
                original.set_(new)
            _register_parameter_or_buffer(self, "original", original)
        else:
            for i, originali in enumerate(new):
                if not isinstance(originali, Tensor):
                    raise ValueError("'right_inverse' must return a Tensor or a Sequence of tensors "
                                     "(list, tuple...). "
                                     f"Got element {i} of the sequence with type {type(originali).__name__}.")

                if isinstance(original, Parameter):
                    originali = Parameter(originali)
                originali.requires_grad_(original.requires_grad)
                _register_parameter_or_buffer(self, f"original{i}", originali)

        if not self.unsafe:

            Z = self()
            if not isinstance(Z, Tensor):
                raise ValueError(
                    f"A parametrization must return a tensor. Got {type(Z).__name__}."
                )
            if Z.dtype != original_dtype:
                raise ValueError(
                    "Registering a parametrization may not change the dtype of the tensor, unless `unsafe` flag is enabled.\n"
                    f"unparametrized dtype: {original_dtype}\n"
                    f"parametrized dtype: {Z.dtype}"
                )
            if Z.shape != original_shape:
                raise ValueError(
                    "Registering a parametrization may not change the shape of the tensor, unless `unsafe` flag is enabled.\n"
                    f"unparametrized shape: {original_shape}\n"
                    f"parametrized shape: {Z.shape}"
                )

    def right_inverse(self, value: Tensor) -> None:
        r"""Calls the methods ``right_inverse`` (see :func:`register_parametrization`)
        of the parametrizations in the inverse order they were registered in.
        Then, it stores the result in ``self.original`` if ``right_inverse`` outputs one tensor
        or in ``self.original0``, ``self.original1``, ... if it outputs several.

        Args:
            value (Tensor): Value to which initialize the module
        """

        with torch.no_grad():

            for module in reversed(self):
                if hasattr(module, "right_inverse"):
                    value = module.right_inverse(value)
                else:
                    raise RuntimeError(f"parametrization {type(module).__name__} does not implement "
                                       "right_inverse.")
            if self.is_tensor:

                if not isinstance(value, Tensor):
                    raise ValueError(
                        f"`right_inverse` should return a tensor. Got {type(value).__name__}"
                    )
                if value.dtype != self.original.dtype:
                    raise ValueError(
                        f"The tensor returned by `right_inverse` has dtype {value.dtype} "
                        f"while `original` has dtype {self.original.dtype}"
                    )

                self.original.set_(value)
            else:
                if not isinstance(value, collections.abc.Sequence):
                    raise ValueError(
                        "'right_inverse' must return a sequence of tensors. "
                        f"Got {type(value).__name__}."
                    )
                if len(value) != self.ntensors:
                    raise ValueError(
                        "'right_inverse' must return a sequence of tensors of length "
                        f"{self.ntensors}. Got a sequence of lenght {len(value)}."
                    )
                for i, tensor in enumerate(value):
                    original_i = getattr(self, f"original{i}")
                    if not isinstance(tensor, Tensor):
                        raise ValueError(
                            f"`right_inverse` must return a sequence of tensors. "
                            f"Got element {i} of type {type(tensor).__name__}"
                        )
                    if original_i.dtype != tensor.dtype:
                        raise ValueError(
                            f"Tensor {i} returned by `right_inverse` has dtype {tensor.dtype} "
                            f"while `original{i}` has dtype {original_i.dtype}"
                        )
                    original_i.set_(tensor)

    def forward(self) -> Tensor:

        if self.is_tensor:
            x = self[0](self.original)
        else:
            originals = (getattr(self, f"original{i}") for i in range(self.ntensors))
            x = self[0](*originals)

        curr_idx = 1
        while hasattr(self, str(curr_idx)):
            x = self[curr_idx](x)
            curr_idx += 1
        return x

def _inject_new_class(module: Module) -> None:
    r"""Sets up a module to be parametrized.

    This works by substituting the class of the module by a class
    that extends it to be able to inject a property

    Args:
        module (nn.Module): module into which to inject the property
    """
    cls = module.__class__

    def getstate(self):
        raise RuntimeError(
            "Serialization of parametrized modules is only "
            "supported through state_dict(). See:\n"
            "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
            "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
        )

    param_cls = type(
        f"Parametrized{cls.__name__}",
        (cls,),
        {
            "__getstate__": getstate,
        },
    )

    module.__class__ = param_cls

def _inject_property(module: Module, tensor_name: str) -> None:
    r"""Injects a property into module[tensor_name].

    It assumes that the class in the module has already been modified from its
    original one using _inject_new_class and that the tensor under :attr:`tensor_name`
    has already been moved out

    Args:
        module (nn.Module): module into which to inject the property
        tensor_name (str): name of the name of the property to create
    """

    assert not hasattr(module, tensor_name)

    @torch.jit.unused
    def get_cached_parametrization(parametrization) -> Tensor:
        global _cache
        key = (id(module), tensor_name)
        tensor = _cache.get(key)
        if tensor is None:
            tensor = parametrization()
            _cache[key] = tensor
        return tensor

    def get_parametrized(self) -> Tensor:
        parametrization = self.parametrizations[tensor_name]
        if _cache_enabled:
            if torch.jit.is_scripting():

                raise RuntimeError('Caching is not implemented for scripting. '
                                   'Either disable caching or avoid scripting.')
            elif torch._C._get_tracing_state() is not None:

                raise RuntimeError('Cannot trace a model while caching parametrizations.')
            else:
                return get_cached_parametrization(parametrization)
        else:

            return parametrization()

    def set_original(self, value: Tensor) -> None:
        self.parametrizations[tensor_name].right_inverse(value)

    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))

def register_parametrization(
    module: Module, tensor_name: str, parametrization: Module, *, unsafe: bool = False,
) -> Module:
    r"""Adds a parametrization to a tensor in a module.

    Assume that ``tensor_name="weight"`` for simplicity. When accessing ``module.weight``,
    the module will return the parametrized version ``parametrization(module.weight)``.
    If the original tensor requires a gradient, the backward pass will differentiate
    through :attr:`parametrization`, and the optimizer will update the tensor accordingly.

    The first time that a module registers a parametrization, this function will add an attribute
    ``parametrizations`` to the module of type :class:`~ParametrizationList`.

    The list of parametrizations on the tensor ``weight`` will be accessible under
    ``module.parametrizations.weight``.

    The original tensor will be accessible under
    ``module.parametrizations.weight.original``.

    Parametrizations may be concatenated by registering several parametrizations
    on the same attribute.

    The training mode of a registered parametrization is updated on registration
    to match the training mode of the host module

    Parametrized parameters and buffers have an inbuilt caching system that can be activated
    using the context manager :func:`cached`.

    A :attr:`parametrization` may optionally implement a method with signature

    .. code-block:: python

        def right_inverse(self, X: Tensor) -> Union[Tensor, Sequence[Tensor]]

    This method is called on the unparametrized tensor when the first parametrization
    is registered to compute the initial value of the original tensor.
    If this method is not implemented, the original tensor will be just the unparametrized tensor.

    If all the parametrizations registered on a tensor implement `right_inverse` it is possible
    to initialize a parametrized tensor by assigning to it, as shown in the example below.

    It is possible for the first parametrization to depend on several inputs.
    This may be implemented returning a tuple of tensors from ``right_inverse``
    (see the example implementation of a ``RankOne`` parametrization below).

    In this case, the unconstrained tensors are also located under ``module.parametrizations.weight``
    with names ``original0``, ``original1``,...

    .. note::

        If unsafe=False (default) both the forward and right_inverse methods will be called
        once to perform a number of consistency checks.
        If unsafe=True, then right_inverse will be called if the tensor is not parametrized,
        and nothing will be called otherwise.

    .. note::

        In most situations, ``right_inverse`` will be a function such that
        ``forward(right_inverse(X)) == X`` (see
        `right inverse <https://en.wikipedia.org/wiki/Inverse_function#Right_inverses>`_).
        Sometimes, when the parametrization is not surjective, it may be reasonable
        to relax this.

    .. warning::

        If a parametrization depends on several inputs, :func:`~register_parametrization`
        will register a number of new parameters. If such parametrization is registered
        after the optimizer is created, these new parameters will need to be added manually
        to the optimizer. See :meth:`torch.Optimizer.add_param_group`.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (str): name of the parameter or buffer on which to register
            the parametrization
        parametrization (nn.Module): the parametrization to register
    Keyword args:
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.

    Raises:
        ValueError: if the module does not have a parameter or a buffer named :attr:`tensor_name`

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.utils.parametrize as P
        >>>
        >>> class Symmetric(nn.Module):
        >>>     def forward(self, X):
        >>>         return X.triu() + X.triu(1).T  # Return a symmetric matrix
        >>>
        >>>     def right_inverse(self, A):
        >>>         return A.triu()
        >>>
        >>> m = nn.Linear(5, 5)
        >>> P.register_parametrization(m, "weight", Symmetric())
        >>> print(torch.allclose(m.weight, m.weight.T))  # m.weight is now symmetric
        True
        >>> A = torch.rand(5, 5)
        >>> A = A + A.T   # A is now symmetric
        >>> m.weight = A  # Initialize the weight to be the symmetric matrix A
        >>> print(torch.allclose(m.weight, A))
        True

        >>> class RankOne(nn.Module):
        >>>     def forward(self, x, y):
        >>>         # Form a rank 1 matrix multiplying two vectors
        >>>         return x.unsqueeze(-1) @ y.unsqueeze(-2)
        >>>
        >>>     def right_inverse(self, Z):
        >>>         # Project Z onto the rank 1 matrices
        >>>         U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        >>>         # Return rescaled singular vectors
        >>>         s0_sqrt = S[0].sqrt().unsqueeze(-1)
        >>>         return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt
        >>>
        >>> linear_rank_one = P.register_parametrization(nn.Linear(4, 4), "weight", RankOne())
        >>> print(torch.linalg.matrix_rank(linear_rank_one.weight).item())
        1

    """
    parametrization.train(module.training)
    if is_parametrized(module, tensor_name):

        if not unsafe:
            Y = getattr(module, tensor_name)
            X = parametrization(Y)
            if not isinstance(X, Tensor):
                raise ValueError(
                    f"A parametrization must return a tensor. Got {type(X).__name__}."
                )
            if X.dtype != Y.dtype:
                raise ValueError(
                    "Registering a parametrization may not change the dtype of the tensor, unless the `unsafe` flag is enabled.\n"
                    f"module.{tensor_name}.dtype: {Y.dtype}\n"
                    f"parametrization(module.{tensor_name}).dtype: {X.dtype}"
                )
            if X.shape != Y.shape:
                raise ValueError(
                    "Registering a parametrization may not change the shape of the tensor, unless the `unsafe` flag is enabled.\n"
                    f"module.{tensor_name}.shape: {Y.shape}\n"
                    f"parametrization(module.{tensor_name}).shape: {X.shape}"
                )
            if hasattr(parametrization, "right_inverse"):
                try:
                    Z = parametrization.right_inverse(X)
                except NotImplementedError:
                    pass
                else:
                    if not isinstance(Z, Tensor):
                        raise ValueError(
                            f"parametrization.right_inverse must return a tensor. Got: {type(Z).__name__}"
                        )
                    if Z.dtype != Y.dtype:
                        raise ValueError(
                            "The tensor returned by parametrization.right_inverse must have the same dtype "
                            f"as module.{tensor_name}, unless the `unsafe` flag is enabled.\n"
                            f"module.{tensor_name}.dtype: {Y.dtype}\n"
                            f"returned dtype: {Z.dtype}"
                        )
                    if Z.shape != Y.shape:
                        raise ValueError(
                            "The tensor returned by parametrization.right_inverse must have the same shape "
                            f"as module.{tensor_name}, unless the `unsafe` flag is enabled.\n"
                            f"module.{tensor_name}.shape: {Y.shape}\n"
                            f"returned shape: {Z.shape}"
                        )

        assert isinstance(module.parametrizations, ModuleDict)
        module.parametrizations[tensor_name].append(parametrization)

        module.parametrizations[tensor_name].unsafe |= unsafe
    elif tensor_name in module._buffers or tensor_name in module._parameters:

        original = getattr(module, tensor_name)

        parametrizations = ParametrizationList([parametrization], original, unsafe=unsafe)

        delattr(module, tensor_name)

        if not is_parametrized(module):

            _inject_new_class(module)

            module.parametrizations = ModuleDict()

        _inject_property(module, tensor_name)

        assert isinstance(module.parametrizations, ModuleDict)
        module.parametrizations[tensor_name] = parametrizations
    else:
        raise ValueError(
            f"Module '{module}' does not have a parameter, a buffer, or a "
            f"parametrized element with name '{tensor_name}'"
        )
    return module

def is_parametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    r"""Returns ``True`` if module has an active parametrization.

    If the argument :attr:`tensor_name` is specified, returns ``True`` if
    ``module[tensor_name]`` is parametrized.

    Args:
        module (nn.Module): module to query
        name (str, optional): attribute in the module to query
            Default: ``None``
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, ModuleDict):
        return False
    if tensor_name is None:

        return len(parametrizations) > 0
    else:
        return tensor_name in parametrizations

def remove_parametrizations(
    module: Module, tensor_name: str, leave_parametrized: bool = True
) -> Module:
    r"""Removes the parametrizations on a tensor in a module.

    - If ``leave_parametrized=True``, ``module[tensor_name]`` will be set to
      its current output. In this case, the parametrization shall not change the ``dtype``
      of the tensor.
    - If ``leave_parametrized=False``, ``module[tensor_name]`` will be set to
      the unparametrised tensor in ``module.parametrizations[tensor_name].original``.
      This is only possible when the parametrization depends on just one tensor.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute :attr:`tensor_name` parametrized.
            Default: ``True``

    Returns:
        Module: module

    Raises:
        ValueError: if ``module[tensor_name]`` is not parametrized
        ValueError: if ``leave_parametrized=False`` and the parametrization depends on several tensors
    """

    if not is_parametrized(module, tensor_name):
        raise ValueError(f"Module {module} does not have a parametrization on {tensor_name}")

    assert isinstance(module.parametrizations, ModuleDict)
    parametrizations = module.parametrizations[tensor_name]
    if parametrizations.is_tensor:
        original = parametrizations.original
        if leave_parametrized:
            with torch.no_grad():
                t = getattr(module, tensor_name)

            with torch.no_grad():
                original.set_(t)
    else:
        if leave_parametrized:

            t = getattr(module, tensor_name)

            original = Parameter(t) if t.requires_grad else t
        else:
            raise ValueError("Cannot leave unparametrized (`leave_parametrized=False`) a tensor "
                             "that is parametrized in terms of a sequence of tensors.")

    delattr(module.__class__, tensor_name)

    del module.parametrizations[tensor_name]

    _register_parameter_or_buffer(module, tensor_name, original)

    if not is_parametrized(module):
        delattr(module, "parametrizations")

        orig_cls = module.__class__.__bases__[0]
        module.__class__ = orig_cls
    return module
