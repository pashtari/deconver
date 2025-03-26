from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.func import vmap
import torch.nn.functional as F


CONV = {d: getattr(F, f"conv{d}d") for d in range(1, 4)}


def conv(input: Tensor, weight: Tensor, groups: int = 1, **kwargs) -> Tensor:
    batch_size = input.shape[0]
    spatial_dims = input.ndim - 2

    # Reshape input and weight for batch computation
    input_reshaped = input.reshape(1, batch_size * input.shape[1], *input.shape[2:])
    weight_reshaped = weight.reshape(
        batch_size * weight.shape[1], weight.shape[2], *weight.shape[3:]
    )

    # Adjust groups for batch computation
    groups = groups * batch_size

    # Perform convolution
    output = CONV[spatial_dims](input_reshaped, weight_reshaped, groups=groups, **kwargs)

    # Reshape output back to batch form
    output = output.reshape(batch_size, -1, *output.shape[2:])

    return output


# #### An alternative implementation of `conv` based on vmap
# @vmap
# def conv(input: Tensor, weight: Tensor, **kwargs) -> Tensor:
#     spatial_dims = input.ndim - 1
#     input = input.unsqueeze(0)
#     output = CONV[spatial_dims](input, weight, **kwargs).squeeze(0)
#     return output


@vmap
def sconv(input1: Tensor, input2: Tensor, **kwargs) -> Tensor:
    spatial_dims = input1.ndim - 1
    input1 = input1.unsqueeze(1)
    input2 = input2.unsqueeze(1)
    output = CONV[spatial_dims](input1, input2, **kwargs)
    return output


def t(x: Tensor) -> Tensor:
    return x.transpose(1, 2)


def flip(h: Tensor) -> Tensor:
    return torch.flip(h, dims=tuple(range(3 - h.ndim, 0)))


def softmax(x: torch.Tensor, dim: int | Sequence[int]) -> torch.Tensor:
    """
    Computes the softmax of a tensor over specified dimensions.

    This function calculates the softmax of the input tensor `x` along the specified
    dimensions `dim`. This is computed by flattening the specified dimensions and applying softmax across them.

    Args:
        x (Tensor): The input tensor.
        dim (int | Sequence[int]): The dimension or dimensions over which to apply
            the softmax operation.

    Returns:
        torch.Tensor: A tensor of the same shape as `x`, with the softmax computed
        along the specified dimensions.
    """

    # Convert dim to a list of nonnegative integers
    dims = [dim] if isinstance(dim, int) else dim
    dims = [d if d >= 0 else x.ndim + d for d in dims]

    # Single dimension case
    if len(dims) == 1:
        return F.softmax(x, dim=dims[0])

    # 1. Group target dimensions via permutation
    non_target_dims = [d for d in range(x.ndim) if d not in dims]
    perm = non_target_dims + dims

    # 2. Compute inverse permutation to restore original order
    inverse_perm = [0] * x.ndim
    for i, p in enumerate(perm):
        inverse_perm[p] = i

    # 3. Permute, flatten, and apply softmax
    x_permuted = x.permute(perm)
    start_dim = len(non_target_dims)
    flattened = x_permuted.flatten(start_dim=start_dim)

    # 4. Apply softmax and restore original shape
    return (
        F.softmax(flattened, dim=start_dim).view(x_permuted.shape).permute(inverse_perm)
    )


def norm2(x: Tensor, w: Optional[Tensor] = None) -> Tensor:
    """
    Computes the batched L2 (Euclidean) norm of a tensor.

    If `w` is provided, the weighted L2 norm is used instead.

    Args:
        x (Tensor): The input tensor with shape `(B, ...)`, where `B` is the batch size.
        w (Tensor, optional): An optional weight tensor with the same shape as `x`. Defaults to None.

    Returns:
        Tensor: A vector of length `B`, containing the weighted L2 norms.
    """
    y = x.flatten(1).square()

    if w is not None:
        w = w.flatten(1)
        y *= w

    return torch.sqrt(torch.sum(y, dim=1))


def relative_error(
    x: Tensor,
    y: Tensor,
    w: Optional[Tensor] = None,
    eps: float = 1e-16,
) -> Tensor:
    """
    Computes the batched relative error between two tensors.

    The relative error is computed as the ratio of the L2 norm of the difference between `x` and `y` to the L2 norm of `x`.
    If `w` is provided, the weighted L2 norm is used instead.

    Args:
        x (Tensor): The first input tensor with shape `(B, ...)`, where `B` is the batch size.
        y (Tensor): The second input tensor with the same shape as `x`.
        w (Tensor, optional): An optional weight tensor with the same shape as `x`. Defaults to None.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.

    Returns:
        Tensor: A vector of length `B`, containing the computed relative errors.
    """
    numerator = norm2(x - y, w) + eps
    denominator = norm2(x, w) + eps
    return numerator / denominator


def kl_divergence(
    x: Tensor,
    y: Tensor,
    eps: float = 1e-16,
):
    """
    Computes the batched Kullback-Leibler (KL) divergence between two tensors.

    Args:
        x (Tensor): The first input tensor with shape `(B, ...)`, where `B` is the batch size.
        y (Tensor): The second input tensor with the same shape as `x`.
        eps (float, optional): A small value to avoid division by zero and log(0). Defaults to 1e-16.

    Returns:
        Tensor: A vector of length `B`, containing the computed KL divergences.
    """
    x = x.clamp(min=eps)  # Avoid log(0)
    y = y.clamp(min=eps)  # Avoid division by zero
    kl_div = (x * torch.log(x / y) - x + y).flatten(1).mean(-1)
    return kl_div
