"""Autograd node for one fused multiplicative deconvolution update.

When the filter is the same for every sample of the batch -- which it is whenever
``update_filter`` is off -- the update collapses to plain grouped convolutions on the
unsplit ``(B, C, *S)`` layout:

    s    = relu(z + b)                              z, b = the initializer's projection
    t1   = K(s)                                     K(u) = conv(u, h, groups=G)
    out  = s * (A(x) + eps) / (A(t1) + eps)         A    = K^T

:class:`DeconvUpdate` runs that as a single autograd node so that

* ``A(x)`` and ``A(t1)`` never reach memory -- the fused kernels compute them inline, and
  recompute them in the backward pass where ``x`` and ``t1`` have to be read anyway, so the
  recomputation costs no traffic at all;
* ``relu(z + b)`` is never materialized -- the kernels rectify and bias ``z`` on load.

Only ``x`` (which the projection retains anyway), ``z`` and ``t1`` are kept for backward,
i.e. ``5U`` against the ``17U`` of the eager formulation, where ``U`` is the size of a
``(B, C, *S)`` tensor.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor

from . import kernels


__all__ = ["deconv_update", "fused_kernels_apply"]


def fused_kernels_apply(
    x: Tensor, m: int, sc: int, kernel_size: Sequence[int]
) -> bool:
    """Whether the fused kernels can and should run for this input and geometry."""
    try:
        functorch = torch._C._are_functorch_transforms_active()
    except AttributeError:  # pragma: no cover - very old torch
        functorch = False
    # torch.func wraps tensors in objects without storage, which Triton cannot address
    return (
        not functorch
        and kernels.triton_available(x)
        and kernels.kernels_supported(m, sc, kernel_size)
    )


def _adjoint_weight(h: Tensor, groups: int, m: int, sc: int) -> Tensor:
    """Weight of ``A = K^T`` as a plain grouped convolution, ``(groups * sc, m, *k)``.

    The kernel is flipped along its spatial axes and the two channel axes are swapped; the
    ``(g, o, i) -> (g, i, o)`` regrouping is a no-op only when ``m == 1``.  Needed just for
    the cuDNN filter-gradient fallback, which takes this form as its weight argument.
    """
    p = h.dim() - 2
    ks = h.shape[2:]
    w = h.reshape(groups, m, sc, *ks).flip(dims=tuple(range(-p, 0)))
    return w.transpose(1, 2).reshape(groups * sc, m, *ks).contiguous()


def _filter_grads_conv(z, bias, x, t1, gt1, gnum, gden, h, groups, m, sc, relu_input):
    """The three filter-gradient contributions, via ``aten.convolution_backward``.

    Used for filter geometries too wide for the fused reduction's register tile.  This is
    the one place that has to materialize the rectified source the kernels fold into their
    loads, because cuDNN takes it as an ordinary tensor argument.
    """
    ks = tuple(h.shape[2:])
    p = len(ks)
    padding = tuple(k // 2 for k in ks)
    s = z if bias is None else z + bias.view(1, -1, *([1] * p))
    if relu_input:
        s = torch.relu(s)
    wa = _adjoint_weight(h, groups, m, sc)
    conv_backward = torch.ops.aten.convolution_backward
    args = dict(
        stride=[1] * p, padding=list(padding), dilation=[1] * p, transposed=False,
        output_padding=[0] * p, groups=groups, output_mask=[False, True, False],
    )
    hf = h.reshape(groups * m, sc, *ks)
    gwf = conv_backward(gt1, s, hf, None, **args)[1]
    gwa = (
        conv_backward(gnum, x, wa, None, **args)[1]
        + conv_backward(gden, t1, wa, None, **args)[1]
    )
    undone = (
        gwa.reshape(groups, sc, m, *ks)
        .transpose(1, 2)
        .flip(dims=tuple(range(-p, 0)))
        .reshape_as(hf)
    )
    return (gwf + undone).reshape_as(h)


class DeconvUpdate(torch.autograd.Function):
    """``out = relu(z + b) * (A(x) + eps) / (A(K(relu(z + b))) + eps)``.

    ``bias`` is the projection's bias, kept out of the projection so that the kernels can
    fold it into their loads instead of paying a read-modify-write pass over the largest
    tensor of the layer.
    """

    @staticmethod
    def forward(ctx, x, z, h, bias, groups, m, sc, eps, relu_input):
        ks = tuple(h.shape[2:])
        hv = h.reshape(groups, m, sc, *ks)
        kw = dict(groups=groups, m=m, sc=sc, kernel_size=ks)
        t1 = kernels.gconv(z, hv, adjoint=False, relu_input=relu_input, bias=bias, **kw)
        out = kernels.deconv_fwd(
            x, t1, z, hv, eps=eps, relu_input=relu_input, bias=bias, **kw
        )
        ctx.cfg = (groups, m, sc, eps, relu_input)
        ctx.save_for_backward(x, z, t1, h, bias)
        return out

    @staticmethod
    def backward(ctx, gout):
        # The backward is written against the saved tensors and against kernels autograd
        # cannot see through, so it is not itself differentiable.  `@once_differentiable`
        # would not catch this -- the engine simply never re-enters this node and the
        # second-order term would be dropped in silence -- so refuse explicitly.  The
        # engine enables grad inside `backward` exactly when the caller asked for
        # `create_graph=True`.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "double backward through the fused Deconv update is not supported; "
                "construct the layer with a convolution keyword argument (e.g. stride=1) "
                "to force the general path, which is plain PyTorch and doubly "
                "differentiable"
            )
        x, z, t1, h, bias = ctx.saved_tensors
        groups, m, sc, eps, relu_input = ctx.cfg
        ks = tuple(h.shape[2:])
        need_x, need_z, need_h, need_b = ctx.needs_input_grad[:4]
        gout = gout.contiguous()

        hv = h.reshape(groups, m, sc, *ks)
        kw = dict(groups=groups, m=m, sc=sc, kernel_size=ks)
        # `A(x)` and `A(t1)` are recomputed inside the kernel: nothing was stored for them
        gnum, gden = kernels.deconv_bwd(
            x, t1, z, gout, hv, eps=eps, relu_input=relu_input, bias=bias, **kw
        )
        gt1 = kernels.gconv(gden, hv, adjoint=False, **kw)  # dL/dt1 = A^T(gden) = K(gden)

        gh = None
        if kernels.wgrad_supported(sc, m, ks):
            wg = dict(groups=groups, ca=sc, cb=m, kernel_size=ks)
            # release the two largest temporaries as early as possible: holding them
            # across the convolutions below costs a full-size tensor at the peak
            if need_h:
                gh = kernels.wgrad(gden, t1, **wg)
            gden = None
            gx = kernels.gconv(gnum, hv, adjoint=False, **kw) if need_x else None
            if need_h:
                gh += kernels.wgrad(gnum, x, **wg)
                gnum = None
                gh = kernels.wgrad(z, gt1, relu_a=relu_input, bias=bias, **wg) + gh
                gh = gh.transpose(1, 2).reshape_as(h)
            gnum = None
        else:
            # wide filter geometry: the fused reduction would spill, so cuDNN takes the
            # filter gradient -- and it needs both temporaries alive
            gx = kernels.gconv(gnum, hv, adjoint=False, **kw) if need_x else None
            if need_h:
                gh = _filter_grads_conv(
                    z, bias, x, t1, gt1, gnum, gden, h, groups, m, sc, relu_input
                )
            gnum = gden = None

        # the direct dL/ds term is recomputed here rather than carried over from
        # `deconv_bwd`: one fewer full-size tensor allocated and written
        gz = (
            kernels.deconv_gz(
                x, t1, z, gout, gt1, hv, eps=eps, relu_input=relu_input, bias=bias, **kw
            )
            if (need_z or need_b)
            else None
        )
        gb = gz.sum(dim=[0] + list(range(2, gz.dim()))) if need_b else None
        return gx, gz if need_z else None, gh, gb, None, None, None, None, None


def deconv_update(
    x: Tensor,
    z: Tensor,
    h: Tensor,
    bias: Optional[Tensor],
    *,
    groups: int,
    m: int,
    sc: int,
    eps: float,
    relu_input: bool,
) -> Tensor:
    """One multiplicative source update.

    Args:
        x: input, ``(B, C, *S)``.
        z: projection output, ``(B, groups * sc, *S)``, pre-bias and pre-ReLU: the kernels
            apply both on load, so the rectified source is never materialized.
        h: rectified filter, ``(C, sc, *k)``.
        bias: the projection's bias, folded in by the kernels.
        eps: stabilizer added to the numerator and the denominator.
    """
    return DeconvUpdate.apply(x, z, h, bias, groups, m, sc, eps, relu_input)
