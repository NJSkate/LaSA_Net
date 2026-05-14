r"""
Pruning methods
"""
import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple

import torch

class BasePruningMethod(ABC):
    r"""Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """
    _tensor_name: str

    def __init__(self):
        pass

    def __call__(self, module, inputs):
        r"""Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using
        :meth:`apply_mask`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.

        Args:
            t (torch.Tensor): tensor representing the importance scores of the
            parameter to prune.
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """
        pass

    def apply_mask(self, module):
        r"""Simply handles the multiplication between the parameter being
        pruned and the generated mask.
        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """

        assert self._tensor_name is not None, "Module {} has to be pruned".format(
            module
        )
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the
                corresponding elements in the parameter being pruned.
                If unspecified or None, the parameter will be used in its place.
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):

            old_method = None
            found = 0

            hooks_to_remove = []
            for k, hook in module._forward_pre_hooks.items():

                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    hooks_to_remove.append(k)
                    found += 1
            assert (
                found <= 1
            ), "Avoid adding multiple pruning hooks to the\
                same tensor {} of module {}. Use a PruningContainer.".format(
                name, module
            )

            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]

            method = cls(*args, **kwargs)

            method._tensor_name = name

            if old_method is not None:

                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method

                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)

                    container.add_pruning_method(method)
                    method = container
            return method

        method = _get_composite_method(cls, module, name, *args, **kwargs)

        orig = getattr(module, name)
        if importance_scores is not None:
            assert (
                importance_scores.shape == orig.shape
            ), "importance_scores should have the same shape as parameter \
                {} of {}".format(
                name, module
            )
        else:
            importance_scores = orig

        if not isinstance(method, PruningContainer):

            module.register_parameter(name + "_orig", orig)

            del module._parameters[name]
            default_mask = torch.ones_like(orig)

        else:
            default_mask = (
                getattr(module, name + "_mask")
                .detach()
                .clone(memory_format=torch.contiguous_format)
            )

        try:

            mask = method.compute_mask(importance_scores, default_mask=default_mask)

            module.register_buffer(name + "_mask", mask)

            setattr(module, name, method.apply_mask(module))

            module.register_forward_pre_hook(method)

        except Exception as e:
            if not isinstance(method, PruningContainer):
                orig = getattr(module, name + "_orig")
                module.register_parameter(name, orig)
                del module._parameters[name + "_orig"]
            raise e

        return method

    def prune(self, t, default_mask=None, importance_scores=None):
        r"""Computes and returns a pruned version of input tensor ``t``
        according to the pruning rule specified in :meth:`compute_mask`.

        Args:
            t (torch.Tensor): tensor to prune (of same dimensions as
                ``default_mask``).
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as ``t``) used to compute mask for pruning ``t``.
                The values in this tensor indicate the importance of the
                corresponding elements in the ``t`` that is being pruned.
                If unspecified or None, the tensor ``t`` will be used in its place.
            default_mask (torch.Tensor, optional): mask from previous pruning
                iteration, if any. To be considered when determining what
                portion of the tensor that pruning should act on. If None,
                default to a mask of ones.

        Returns:
            pruned version of tensor ``t``.
        """
        if importance_scores is not None:
            assert (
                importance_scores.shape == t.shape
            ), "importance_scores should have the same shape as tensor t"
        else:
            importance_scores = t
        default_mask = default_mask if default_mask is not None else torch.ones_like(t)
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module):
        r"""Removes the pruning reparameterization from a module. The pruned
        parameter named ``name`` remains permanently pruned, and the parameter
        named ``name+'_orig'`` is removed from the parameter list. Similarly,
        the buffer named ``name+'_mask'`` is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """

        assert (
            self._tensor_name is not None
        ), "Module {} has to be pruned\
            before pruning can be removed".format(
            module
        )

        weight = self.apply_mask(module)

        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]
        setattr(module, self._tensor_name, orig)

class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.
    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args):
        self._pruning_methods: Tuple["BasePruningMethod", ...] = tuple()
        if not isinstance(args, Iterable):
            self._tensor_name = args._tensor_name
            self.add_pruning_method(args)
        elif len(args) == 1:
            self._tensor_name = args[0]._tensor_name
            self.add_pruning_method(args[0])
        else:
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method):
        r"""Adds a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """

        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(
                "{} is not a BasePruningMethod subclass".format(type(method))
            )
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(
                "Can only add pruning methods acting on "
                "the parameter named '{}' to PruningContainer {}.".format(
                    self._tensor_name, self
                )
                + " Found '{}'".format(method._tensor_name)
            )

        self._pruning_methods += (method,)

    def __len__(self):
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        r"""Applies the latest ``method`` by computing the new partial masks
        and returning its combination with the ``default_mask``.
        The new partial mask should be computed on the entries or channels
        that were not zeroed out by the ``default_mask``.
        Which portions of the tensor ``t`` the new mask will be calculated from
        depends on the ``PRUNING_TYPE`` (handled by the type handler):

        * for 'unstructured', the mask will be computed from the raveled
          list of nonmasked entries;

        * for 'structured', the mask will be computed from the nonmasked
          channels in the tensor;

        * for 'global', the mask will be computed across all entries.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
                (of same dimensions as ``default_mask``).
            default_mask (torch.Tensor): mask from previous pruning iteration.

        Returns:
            mask (torch.Tensor): new mask that combines the effects
            of the ``default_mask`` and the new mask from the current
            pruning ``method`` (of same dimensions as ``default_mask`` and
            ``t``).
        """

        def _combine_masks(method, t, mask):
            r"""
            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration

            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask
            new_mask = new_mask.to(dtype=t.dtype)

            if method.PRUNING_TYPE == "unstructured":

                slc = mask == 1

            elif method.PRUNING_TYPE == "structured":
                if not hasattr(method, "dim"):
                    raise AttributeError(
                        "Pruning methods of PRUNING_TYPE "
                        '"structured" need to have the attribute `dim` defined.'
                    )

                n_dims = t.dim()
                dim = method.dim

                if dim < 0:
                    dim = n_dims + dim

                if dim < 0:
                    raise IndexError(
                        "Index is out of bounds for tensor with dimensions {}".format(
                            n_dims
                        )
                    )

                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0

                slc = [slice(None)] * n_dims
                slc[dim] = keep_channel

            elif method.PRUNING_TYPE == "global":
                n_dims = len(t.shape)
                slc = [slice(None)] * n_dims

            else:
                raise ValueError(
                    "Unrecognized PRUNING_TYPE {}".format(method.PRUNING_TYPE)
                )

            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)

            return new_mask

        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask

class Identity(BasePruningMethod):
    r"""Utility pruning method that does not prune any units but generates the
    pruning parametrization with a mask of ones.
    """

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super(Identity, cls).apply(module, name)

class RandomUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor at random.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):

        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):

        tensor_size = t.nelement()

        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)

        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:
            prob = torch.rand_like(t)
            topk = torch.topk(prob.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(RandomUnstructured, cls).apply(module, name, amount=amount)

class L1Unstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):

        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):

        tensor_size = t.nelement()

        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)

        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:

            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)

            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(L1Unstructured, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )

class RandomStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, dim=-1):

        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """

        _validate_structured_pruning(t)

        _validate_pruning_dim(t, self.dim)

        tensor_size = t.shape[self.dim]

        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)

        _validate_pruning_amount(nparams_toprune, tensor_size)

        def make_mask(t, dim, nchannels, nchannels_toprune):

            prob = torch.rand(nchannels)

            threshold = torch.kthvalue(prob, k=nchannels_toprune).values
            channel_mask = prob > threshold

            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = channel_mask
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:
            mask = default_mask
        else:

            mask = make_mask(t, self.dim, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, amount, dim=-1):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super(RandomStructured, cls).apply(module, name, amount=amount, dim=dim)

class LnStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor based on their
    L\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, n, dim=-1):

        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a mask to apply on
        top of the ``default_mask`` by zeroing out the channels along the
        specified dim with the lowest L\ ``n``-norm.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied.  Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """

        _validate_structured_pruning(t)

        _validate_pruning_dim(t, self.dim)

        tensor_size = t.shape[self.dim]

        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune

        _validate_pruning_amount(nparams_toprune, tensor_size)

        norm = _compute_norm(t, self.n, self.dim)

        topk = torch.topk(norm, k=nparams_tokeep, largest=True)

        def make_mask(t, dim, indices):

            mask = torch.zeros_like(t)

            slc = [slice(None)] * len(t.shape)

            slc[dim] = indices

            mask[slc] = 1
            return mask

        if nparams_toprune == 0:
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask

    @classmethod
    def apply(cls, module, name, amount, n, dim, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to
                prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(LnStructured, cls).apply(
            module,
            name,
            amount=amount,
            n=n,
            dim=dim,
            importance_scores=importance_scores,
        )

class CustomFromMask(BasePruningMethod):

    PRUNING_TYPE = "global"

    def __init__(self, mask):
        self.mask = mask

    def compute_mask(self, t, default_mask):
        assert default_mask.shape == self.mask.shape
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super(CustomFromMask, cls).apply(module, name, mask=mask)

def identity(module, name):
    r"""Applies pruning reparametrization to the tensor corresponding to the
    parameter called ``name`` in ``module`` without actually pruning any
    units. Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Note:
        The mask is a tensor of ones.

    Args:
        module (nn.Module): module containing the tensor to prune.
        name (str): parameter name within ``module`` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.identity(nn.Linear(2, 3), 'bias')
        >>> print(m.bias_mask)
        tensor([1., 1., 1.])
    """
    Identity.apply(module, name)
    return module

def random_unstructured(module, name, amount):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) units
    selected at random.
    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.random_unstructured(nn.Linear(2, 3), 'weight', amount=1)
        >>> torch.sum(m.weight_mask == 0)
        tensor(1)

    """
    RandomUnstructured.apply(module, name, amount)
    return module

def l1_unstructured(module, name, amount, importance_scores=None):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), 'weight', amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module

def random_structured(module, name, amount, dim):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` selected at random.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int): index of the dim along which we define channels to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.random_structured(
                nn.Linear(5, 3), 'weight', amount=3, dim=1
            )
        >>> columns_pruned = int(sum(torch.sum(m.weight, dim=0) == 0))
        >>> print(columns_pruned)
        3
    """
    RandomStructured.apply(module, name, amount, dim)
    return module

def ln_structured(module, name, amount, n, dim, importance_scores=None):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` with the lowest L\ ``n``-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int): index of the dim along which we define channels to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.ln_structured(
               nn.Conv2d(5, 3, 2), 'weight', amount=0.3, dim=1, n=float('-inf')
            )
    """
    LnStructured.apply(
        module, name, amount, n, dim, importance_scores=importance_scores
    )
    return module

def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters``
    by applying the specified ``pruning_method``.
    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> net = nn.Sequential(OrderedDict([
                ('first', nn.Linear(10, 4)),
                ('second', nn.Linear(4, 1)),
            ]))
        >>> parameters_to_prune = (
                (net.first, 'weight'),
                (net.second, 'weight'),
            )
        >>> prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=10,
            )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10, dtype=torch.uint8)

    """

    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )

    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    container = PruningContainer()
    container._tensor_name = "temp"
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    pointer = 0
    for module, name in parameters:

        param = getattr(module, name)

        num_param = param.numel()

        param_mask = final_mask[pointer : pointer + num_param].view_as(param)

        custom_from_mask(module, name, mask=param_mask)

        pointer += num_param

def custom_from_mask(module, name, mask):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by applying the pre-computed mask in ``mask``.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> m = prune.custom_from_mask(
                nn.Linear(5, 3), name='bias', mask=torch.tensor([0, 1, 0])
            )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    """
    CustomFromMask.apply(module, name, mask)
    return module

def remove(module, name):
    r"""Removes the pruning reparameterization from a module and the
    pruning method from the forward hook. The pruned
    parameter named ``name`` remains permanently pruned, and the parameter
    named ``name+'_orig'`` is removed from the parameter list. Similarly,
    the buffer named ``name+'_mask'`` is removed from the buffers.

    Note:
        Pruning itself is NOT undone or reversed!

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.

    Examples:
        >>> m = random_unstructured(nn.Linear(5, 7), name='weight', amount=0.2)
        >>> m = remove(m, name='weight')
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(
        "Parameter '{}' of module {} has to be pruned "
        "before pruning can be removed".format(name, module)
    )

def is_pruned(module):
    r"""Check whether ``module`` is pruned by looking for
    ``forward_pre_hooks`` in its modules that inherit from the
    :class:`BasePruningMethod`.

    Args:
        module (nn.Module): object that is either pruned or unpruned

    Returns:
        binary answer to whether ``module`` is pruned.

    Examples:
        >>> m = nn.Linear(5, 7)
        >>> print(prune.is_pruned(m))
        False
        >>> prune.random_unstructured(m, name='weight', amount=0.2)
        >>> print(prune.is_pruned(m))
        True
    """
    for _, submodule in module.named_modules():
        for _, hook in submodule._forward_pre_hooks.items():
            if isinstance(hook, BasePruningMethod):
                return True
    return False

def _validate_pruning_amount_init(amount):
    r"""Validation helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
            integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
    """
    if not isinstance(amount, numbers.Real):
        raise TypeError(
            "Invalid type for amount: {}. Must be int or float." "".format(amount)
        )

    if (isinstance(amount, numbers.Integral) and amount < 0) or (
        not isinstance(amount, numbers.Integral)
        and (float(amount) > 1.0 or float(amount) < 0.0)
    ):
        raise ValueError(
            "amount={} should either be a float in the "
            "range [0, 1] or a non-negative integer"
            "".format(amount)
        )

def _validate_pruning_amount(amount, tensor_size):
    r"""Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (`tensor_size`).

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    """

    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            "amount={} should be smaller than the number of "
            "parameters to prune={}".format(amount, tensor_size)
        )

def _validate_structured_pruning(t):
    r"""Validation helper to check that the tensor to be pruned is multi-
    dimensional, such that the concept of "channels" is well-defined.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune

    Raises:
        ValueError: if the tensor `t` is not at least 2D.
    """
    shape = t.shape
    if len(shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to "
            "multidimensional tensors. Found tensor of shape "
            "{} with {} dims".format(shape, len(shape))
        )

def _compute_nparams_toprune(amount, tensor_size):
    r"""Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Returns:
        int: the number of units to prune in the tensor
    """

    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return int(round(amount * tensor_size))

def _validate_pruning_dim(t, dim):
    r"""
    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        dim (int): index of the dim along which we define channels to prune
    """
    if dim >= t.dim():
        raise IndexError("Invalid index {} for tensor of size {}".format(dim, t.shape))

def _compute_norm(t, n, dim):
    r"""Compute the L_n-norm across all entries in tensor `t` along all dimension
    except for the one identified by dim.
    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),
    then norm will have Size [4], and each entry will represent the
    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        dim (int): dim identifying the channels to prune

    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `dim`. By construction, `norm.shape = t.shape[-1]`.
    """

    dims = list(range(t.dim()))

    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    norm = torch.norm(t, p=n, dim=dims)
    return norm
