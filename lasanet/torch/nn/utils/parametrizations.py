from enum import Enum, auto

import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F

from typing import Optional

def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)

    eps = 10. * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.transpose(-2, -1).conj() @ Q, Id, atol=eps)

def _make_orthogonal(A):
    """ Assume that A is a tall matrix.
    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative
    """
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)

    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q

class _OrthMaps(Enum):
    matrix_exp = auto()
    cayley = auto()
    householder = auto()

class _Orthogonal(Module):
    base: Tensor

    def __init__(self,
                 weight,
                 orthogonal_map: _OrthMaps,
                 *,
                 use_trivialization=True) -> None:
        super().__init__()

        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError("The householder parametrization does not support complex tensors.")

        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer("base", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n, k = X.size(-2), X.size(-1)
        transposed = n < k
        if transposed:
            X = X.transpose(-2, -1)
            n, k = k, n

        if self.orthogonal_map == _OrthMaps.matrix_exp or self.orthogonal_map == _OrthMaps.cayley:

            X = X.tril()
            if n != k:

                X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
            A = X - X.transpose(-2, -1).conj()

            if self.orthogonal_map == _OrthMaps.matrix_exp:
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:

                Id = torch.eye(n, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))

            if n != k:
                Q = Q[..., :k]

        else:

            A = X.tril(diagonal=-1)
            tau = 2. / (1. + (A * A).sum(dim=-2))
            Q = torch.linalg.householder_product(A, tau)

            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)

        if hasattr(self, "base"):
            Q = self.base @ Q
        if transposed:
            Q = Q.transpose(-2, -1)
        return Q

    @torch.autograd.no_grad()
    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor:
        if Q.shape != self.shape:
            raise ValueError(f"Expected a matrix or batch of matrices of shape {self.shape}. "
                             f"Got a tensor of shape {Q.shape}.")

        Q_init = Q
        n, k = Q.size(-2), Q.size(-1)
        transpose = n < k
        if transpose:
            Q = Q.transpose(-2, -1)
            n, k = k, n

        if not hasattr(self, "base"):

            if self.orthogonal_map == _OrthMaps.cayley or self.orthogonal_map == _OrthMaps.matrix_exp:
                raise NotImplementedError("It is not possible to assign to the matrix exponential "
                                          "or the Cayley parametrizations when use_trivialization=False.")

            A, tau = torch.geqrf(Q)

            A.diagonal(dim1=-2, dim2=-1).sign_()

            A.diagonal(dim1=-2, dim2=-1)[tau == 0.] *= -1
            return A.transpose(-2, -1) if transpose else A
        else:
            if n == k:

                if not _is_orthogonal(Q):
                    Q = _make_orthogonal(Q)
                else:
                    Q = Q.clone()
            else:

                N = torch.randn(*(Q.size()[:-2] + (n, n - k)), dtype=Q.dtype, device=Q.device)
                Q = torch.cat([Q, N], dim=-1)
                Q = _make_orthogonal(Q)
            self.base = Q

            neg_Id = torch.zeros_like(Q_init)
            neg_Id.diagonal(dim1=-2, dim2=-1).fill_(-1.)
            return neg_Id

def orthogonal(module: Module,
               name: str = 'weight',
               orthogonal_map: Optional[str] = None,
               *,
               use_trivialization: bool = True) -> Module:
    r"""Applies an orthogonal or unitary parametrization to a matrix or a batch of matrices.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the parametrized
    matrix :math:`Q \in \mathbb{K}^{m \times n}` is **orthogonal** as

    .. math::

        \begin{align*}
            Q^{\text{H}}Q &= \mathrm{I}_n \mathrlap{\qquad \text{if }m \geq n}\\
            QQ^{\text{H}} &= \mathrm{I}_m \mathrlap{\qquad \text{if }m < n}
        \end{align*}

    where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex
    and the transpose when :math:`Q` is real-valued, and
    :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.
    In plain words, :math:`Q` will have orthonormal columns whenever :math:`m \geq n`
    and orthonormal rows otherwise.

    If the tensor has more than two dimensions, we consider it as a batch of matrices of shape `(..., m, n)`.

    The matrix :math:`Q` may be parametrized via three different ``orthogonal_map`` in terms of the original tensor:

    - ``"matrix_exp"``/``"cayley"``:
      the :func:`~torch.matrix_exp` :math:`Q = \exp(A)` and the `Cayley map`_
      :math:`Q = (\mathrm{I}_n + A/2)(\mathrm{I}_n - A/2)^{-1}` are applied to a skew-symmetric
      :math:`A` to give an orthogonal matrix.
    - ``"householder"``: computes a product of Householder reflectors
      (:func:`~torch.linalg.householder_product`).

    ``"matrix_exp"``/``"cayley"`` often make the parametrized weight converge faster than
    ``"householder"``, but they are slower to compute for very thin or very wide matrices.

    If ``use_trivialization=True`` (default), the parametrization implements the "Dynamic Trivialization Framework",
    where an extra matrix :math:`B \in \mathbb{K}^{n \times n}` is stored under
    ``module.parametrizations.weight[0].base``. This helps the
    convergence of the parametrized layer at the expense of some extra memory use.
    See `Trivializations for Gradient-Based Optimization on Manifolds`_ .

    Initial value of :math:`Q`:
    If the original tensor is not parametrized and ``use_trivialization=True`` (default), the initial value
    of :math:`Q` is that of the original tensor if it is orthogonal (or unitary in the complex case)
    and it is orthogonalized via the QR decomposition otherwise (see :func:`torch.linalg.qr`).
    Same happens when it is not parametrized and ``orthogonal_map="householder"`` even when ``use_trivialization=False``.
    Otherwise, the initial value is the result of the composition of all the registered
    parametrizations applied to the original tensor.

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`.

    .. _`Cayley map`: https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    .. _`Trivializations for Gradient-Based Optimization on Manifolds`: https://arxiv.org/abs/1909.09501

    Args:
        module (nn.Module): module on which to register the parametrization.
        name (str, optional): name of the tensor to make orthogonal. Default: ``"weight"``.
        orthogonal_map (str, optional): One of the following: ``"matrix_exp"``, ``"cayley"``, ``"householder"``.
            Default: ``"matrix_exp"`` if the matrix is square or complex, ``"householder"`` otherwise.
        use_trivialization (bool, optional): whether to use the dynamic trivialization framework.
            Default: ``True``.

    Returns:
        The original module with an orthogonal parametrization registered to the specified
        weight

    Example::

        >>> orth_linear = orthogonal(nn.Linear(20, 40))
        >>> orth_linear
        ParametrizedLinear(
        in_features=20, out_features=40, bias=True
        (parametrizations): ModuleDict(
            (weight): ParametrizationList(
            (0): _Orthogonal()
            )
        )
        )
        >>> Q = orth_linear.weight
        >>> torch.dist(Q.T @ Q, torch.eye(20))
        tensor(4.9332e-07)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter ot buffer with name '{}'".format(module, name)
        )

    if weight.ndim < 2:
        raise ValueError("Expected a matrix or batch of matrices. "
                         f"Got a tensor of {weight.ndim} dimensions.")

    if orthogonal_map is None:
        orthogonal_map = "matrix_exp" if weight.size(-2) == weight.size(-1) or weight.is_complex() else "householder"

    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError('orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
                         f'Got: {orthogonal_map}')
    orth = _Orthogonal(weight,
                       orth_enum,
                       use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    return module

class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:

            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer('_u', F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer('_v', F.normalize(v, dim=0, eps=self.eps))

            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:

        assert weight.ndim > 1

        if self.dim != 0:

            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:

        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):

            self._u = F.normalize(torch.mv(weight_mat, self._v),
                                  dim=0, eps=self.eps, out=self._u)
            self._v = F.normalize(torch.mv(weight_mat.t(), self._u),
                                  dim=0, eps=self.eps, out=self._v)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:

            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)

            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)

            sigma = torch.dot(u, torch.mv(weight_mat, v))
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:

        return value

def spectral_norm(module: Module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> Module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    Example::

        >>> snm = spectral_norm(nn.Linear(20, 40))
        >>> snm
        ParametrizedLinear(
        in_features=20, out_features=40, bias=True
        (parametrizations): ModuleDict(
            (weight): ParametrizationList(
            (0): _SpectralNorm()
            )
        )
        )
        >>> torch.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0000, grad_fn=<CopyBackwards>)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(module, name, _SpectralNorm(weight, n_power_iterations, dim, eps))
    return module
