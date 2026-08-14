"""Fused Triton kernels for the multiplicative deconvolution update.

The deconvolution update of :class:`~deconver.deconvolution.deconv.Deconv` is entirely
memory-bandwidth bound: the convolutions involved have tiny per-group channel counts
(``C / groups`` in, ``source_channels`` out), so the arithmetic intensity of every step
is a handful of FLOPs per byte.  cuDNN's grouped 3-D convolutions run 3-8x away from the
bandwidth roofline for these shapes, and the elementwise part of the update
(``s * (num + eps) / (den + eps)``) costs four full passes over the largest tensor of the
layer when executed eagerly.

The kernels below fuse the update into a handful of passes:

``_deconv_fwd_kernel``
    computes ``out = s * (A(x) + eps) / (A(t1) + eps)`` in one pass, without ever
    materializing ``A(x)`` or ``A(t1)``.
``_deconv_bwd_kernel``
    recomputes ``A(x)`` and ``A(t1)`` on the fly (they are never stored for backward) and
    emits the three elementwise gradients of the epilogue in one pass.
``_gconv_kernel``
    a grouped convolution, forward (``K``) or adjoint (``A``), with optional ``+ add`` and
    ReLU-mask epilogues, used for the remaining convolution applications.
``_wgrad_kernel``
    the filter-gradient reduction, accumulated into a small partial-sum workspace which is
    reduced with a regular ``sum``: a fixed reduction tree in true fp32, where cuDNN's
    grouped weight gradient runs on TF32 tensor cores by default (measured: 2.3e-04
    relative error against float64, against 1.1e-07 here).

All kernels address the volume through a *flattened* spatial coordinate and unroll the
filter taps at compile time, which makes them shape-agnostic and identical for 1-D, 2-D
and 3-D inputs.

Operator conventions (per group ``g``, with ``m = C // groups`` filter channels,
``sc = source_channels`` and ``pad = k // 2``):

    K(u)[o, p] = sum_{i, tau} u[i, p + tau - pad] * h[o, i, tau]      (correlation)
    A(v)[i, p] = sum_{o, tau} v[o, p + pad - tau] * h[o, i, tau]      (adjoint of K)

``A`` is the exact adjoint of ``K`` for stride 1, odd kernel sizes and
``padding = k // 2``, which is what the fast path is gated on.
"""

from typing import NamedTuple, Optional, Sequence

import torch
from torch import Tensor

try:  # pragma: no cover - trivial import guard
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


__all__ = [
    "HAS_TRITON",
    "triton_available",
    "kernels_supported",
    "gconv",
    "wgrad_supported",
    "deconv_fwd",
    "deconv_bwd",
    "deconv_gz",
    "wgrad",
]


# Every limit below was set from measurement; the configurations they exclude run the
# original (general) code path instead, which is never slower than itself.
#
# Triton indexing here is 32-bit.  The margin keeps even the *masked* tail lanes (up to one
# block past the end, shifted by a kernel radius) inside int32, though they are never read.
_MAX_NUMEL = 2**31 - 2**20
# The kernels hold one accumulator tile per channel of a group, so wide geometries spill.
# At C=32, 32^3, batch 2, 3x3x3: 16 source channels per group runs at 3.5x the original,
# 32 at 0.06x.  The unroll budget bounds the compiled kernel the same way.
_MAX_LANES = 16
_MAX_UNROLL = 8192
_MAX_TAPS = 128
# Unrolling the tap loop in the Triton frontend constant-folds the tap offsets and their
# boundary masks -- worth ~13% on the 3x3x3 filters the published configurations use -- but
# compilation slows down sharply as the unrolled body grows (measured: 40 s for 5^3 taps,
# 3 min for 3^3 with four input and four output channels per group, 7 min for 5^3 with four
# source channels, at which point the kernel also spills).  Past these budgets it stays
# rolled, and compiles in ~2 s regardless of kernel size.
_MAX_STATIC_TAPS = 32
_MAX_STATIC_WORK = 128
# Register-tile budget of the filter-gradient kernel, in fp32 elements.
_WGRAD_TILE = 16384


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1)).bit_length()


def _taps(kernel_size: Sequence[int]) -> int:
    taps = 1
    for k in kernel_size:
        taps *= k
    return taps


def _unroll_factor(taps: int, cin: int = 1, cout: int = 1) -> bool:
    """Whether to unroll this kernel's tap loop in the Triton frontend.

    The budget is on the number of emitted load/multiply-add blocks, not on taps alone.
    """
    return taps <= _MAX_STATIC_TAPS and taps * cin * _next_pow2(cout) <= _MAX_STATIC_WORK


def triton_available(*tensors: Optional[Tensor]) -> bool:
    """Whether the fused kernels can address ``tensors``."""
    if not HAS_TRITON:
        return False
    return all(
        x is None
        or (
            x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and x.numel() <= _MAX_NUMEL
        )
        for x in tensors
    )


def wgrad_supported(ca: int, cb: int, kernel_size: Sequence[int]) -> bool:
    """Whether the fused filter-gradient reduction fits in registers.

    Its accumulator is a ``(ca, cb, taps, block)`` tile, so wide filter geometries would
    spill; those fall back to cuDNN for the filter gradient only, keeping the fused
    convolutions and epilogue.
    """
    return HAS_TRITON and (
        _next_pow2(ca) * _next_pow2(cb) * _next_pow2(_taps(kernel_size))
        <= _WGRAD_TILE // 16
    )


def kernels_supported(m: int, sc: int, kernel_size: Sequence[int]) -> bool:
    """Whether the fused kernels apply, and are worth using, for this filter geometry.

    Besides the size budgets there is one judgement call: a single source channel with a
    large kernel is left to cuDNN.  The fusion earns its keep by collapsing the elementwise
    passes over the ``groups * source_channels`` source tensor; with one source channel that
    tensor is no bigger than the input, so there is nothing to amortize, while a large
    kernel is compute bound where cuDNN's depthwise kernels are strong.  Measured at C=32,
    32^3, batch 2: 5x5x5 with ``sc = 1`` is 0.82x the original, ``sc = 2`` is 1.36x and
    ``sc = 4`` is 2.93x.
    """
    if not HAS_TRITON or not 1 <= len(kernel_size) <= 3:
        return False
    taps = _taps(kernel_size)
    return (
        not (sc == 1 and taps > _MAX_STATIC_TAPS)
        and _next_pow2(sc) <= _MAX_LANES
        and _next_pow2(m) <= _MAX_LANES
        and taps <= _MAX_TAPS
        and m * _next_pow2(sc) * taps <= _MAX_UNROLL
    )


if HAS_TRITON:

    _CONV_CONFIGS = [
        triton.Config({"BP": 128}, num_warps=4),
        triton.Config({"BP": 256}, num_warps=4),
        triton.Config({"BP": 512}, num_warps=4),
        triton.Config({"BP": 512}, num_warps=8),
        triton.Config({"BP": 1024}, num_warps=8),
    ]

    @triton.jit
    def _tile(SP, G, D, H, W, BP: tl.constexpr):
        """Decode the flat program id into ``(b, g)`` and a block of spatial positions.

        Returns the batch and group index, the flattened positions ``p`` with their
        in-range mask, and the ``(d, h, w)`` decomposition the tap masks need.  The grid is
        one-dimensional because CUDA caps dimension y at 65535, which ``batch * groups``
        exceeds for large batches at the wide stages.
        """
        npid = tl.cdiv(SP, BP)
        bg = tl.program_id(0) // npid
        b = bg // G
        p = (tl.program_id(0) % npid) * BP + tl.arange(0, BP)
        return b, bg - b * G, p, p < SP, p // (W * H), (p // W) % H, p % W

    @triton.jit
    def _tap_mask_dyn(pm, d_, h_, w_, D, H, W, dd, dh, dw):
        """Validity mask of the shifted read ``p + (dd, dh, dw)``, with runtime offsets."""
        return (
            pm
            & (d_ + dd >= 0)
            & (d_ + dd < D)
            & (h_ + dh >= 0)
            & (h_ + dh < H)
            & (w_ + dw >= 0)
            & (w_ + dw < W)
        )

    @triton.jit
    def _gconv_tap(
        acc, IN, BIAS, WT, in_base, wg, wstep, p, pm, d_, h_, w_, D, H, W, g,
        co, com, kd, kh, kw,
        SP, CIN, SC, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
        PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr, SGN: tl.constexpr,
        RELU_IN: tl.constexpr, HAS_BIAS: tl.constexpr,
    ):
        """One filter tap of the grouped convolution.

        Called from a `static_range` loop (where `kd, kh, kw` are compile-time constants and
        the offsets and masks fold away) and from a rolled loop (where they are not).
        """
        KT: tl.constexpr = KD * KH * KW
        # tap offset: `p + tau - pad` for K, `p + pad - tau` for A
        dd = SGN * (kd - PD)
        dh = SGN * (kh - PH)
        dw = SGN * (kw - PW)
        m = _tap_mask_dyn(pm, d_, h_, w_, D, H, W, dd, dh, dw)
        shift = dd * (H * W) + dh * W + dw
        tau = ((kd * KH) + kh) * KW + kw
        for ci in tl.static_range(CIN):
            v = tl.load(IN + in_base + ci * SP + p + shift, mask=m, other=0.0)
            if HAS_BIAS:
                # bias of the input channel, folded in on load: the projection's bias add
                # never touches global memory.  Out-of-bounds taps stay zero -- the
                # reference convolves a zero-padded tensor, not a bias-padded one.
                v = tl.where(m, v + tl.load(BIAS + g * CIN + ci), 0.0)
            if RELU_IN:
                v = tl.maximum(v, 0.0)
            # ADJOINT: o = ci, i = co  |  else: o = co, i = ci
            wbase = wg + (ci * SC * KT if SGN == -1 else ci * KT) + tau
            wv = tl.load(WT + wbase + co * wstep, mask=com, other=0.0)
            acc += v[None, :] * wv[:, None]
        return acc

    @triton.jit
    def _adjoint_tap(
        acc_n, acc_d, acc_g, X, T1, GT1, WT, xbase, wg, p, pm, d_, h_, w_, D, H, W,
        i_, im, kd, kh, kw, SP,
        M: tl.constexpr, SC: tl.constexpr, KD: tl.constexpr, KH: tl.constexpr,
        KW: tl.constexpr, PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
        NACC: tl.constexpr,
    ):
        """One filter tap of ``A`` applied to two or three tensors at once."""
        KT: tl.constexpr = KD * KH * KW
        m = _tap_mask_dyn(pm, d_, h_, w_, D, H, W, PD - kd, PH - kh, PW - kw)
        shift = (PD - kd) * (H * W) + (PH - kh) * W + (PW - kw)
        tau = ((kd * KH) + kh) * KW + kw
        for o in tl.static_range(M):
            wv = tl.load(WT + wg + o * (SC * KT) + tau + i_ * KT, mask=im, other=0.0)
            xv = tl.load(X + xbase + o * SP + p + shift, mask=m, other=0.0)
            tv = tl.load(T1 + xbase + o * SP + p + shift, mask=m, other=0.0)
            acc_n += xv[None, :] * wv[:, None]
            acc_d += tv[None, :] * wv[:, None]
            if NACC == 3:
                gv = tl.load(GT1 + xbase + o * SP + p + shift, mask=m, other=0.0)
                acc_g += gv[None, :] * wv[:, None]
        return acc_n, acc_d, acc_g

    # ------------------------------------------------------------------ grouped conv

    @triton.autotune(configs=_CONV_CONFIGS, key=["SP", "G"])
    @triton.jit
    def _gconv_kernel(
        IN,
        WT,
        OUT,
        BIAS,
        SP,
        G,
        D,
        H,
        W,
        M: tl.constexpr,
        SC: tl.constexpr,
        COUT_P: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        PD: tl.constexpr,
        PH: tl.constexpr,
        PW: tl.constexpr,
        ADJOINT: tl.constexpr,
        RELU_IN: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        UNROLL: tl.constexpr,
        BP: tl.constexpr,
    ):
        CIN: tl.constexpr = M if ADJOINT else SC
        COUT: tl.constexpr = SC if ADJOINT else M
        SGN: tl.constexpr = -1 if ADJOINT else 1
        KT: tl.constexpr = KD * KH * KW

        b, g, p, pm, d_, h_, w_ = _tile(SP, G, D, H, W, BP)

        co = tl.arange(0, COUT_P)
        com = co < COUT

        in_base = (b * G + g) * (CIN * SP)
        out_base = (b * G + g) * (COUT * SP)
        # weight index: ((g * M + o) * SC + i) * KT + tau
        wg = g * (M * SC * KT)
        wstep = SC * KT if not ADJOINT else KT  # stride of the output-channel axis

        acc = tl.zeros((COUT_P, BP), dtype=tl.float32)

        # Unrolling the tap loop in the frontend constant-folds every tap offset and its
        # boundary mask, which is worth ~13% on a 3x3x3 kernel -- but compilation cost
        # grows with the kernel volume (5^3 taps take ~40 s, and ~7 minutes once there are
        # several source channels, at which point the unrolled kernel also spills).  Large
        # kernels therefore keep the loop rolled.
        if UNROLL:
            for kd in tl.static_range(KD):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        acc = _gconv_tap(
                            acc, IN, BIAS, WT, in_base, wg, wstep, p, pm, d_, h_, w_,
                            D, H, W, g, co, com, kd, kh, kw, SP, CIN, SC,
                            KD, KH, KW, PD, PH, PW, SGN, RELU_IN, HAS_BIAS,
                        )
        else:
            for tau in tl.range(0, KT):
                acc = _gconv_tap(
                    acc, IN, BIAS, WT, in_base, wg, wstep, p, pm, d_, h_, w_,
                    D, H, W, g, co, com,
                    tau // (KH * KW), (tau // KW) % KH, tau % KW, SP, CIN, SC,
                    KD, KH, KW, PD, PH, PW, SGN, RELU_IN, HAS_BIAS,
                )

        off = out_base + co[:, None] * SP + p[None, :]
        omask = com[:, None] & pm[None, :]
        tl.store(OUT + off, acc, mask=omask)

    # -------------------------------------------------------------- fused fwd update

    @triton.autotune(configs=_CONV_CONFIGS, key=["SP", "G"])
    @triton.jit
    def _deconv_fwd_kernel(
        X,
        T1,
        Z,
        WT,
        BIAS,
        OUT,
        SP,
        G,
        D,
        H,
        W,
        eps,
        M: tl.constexpr,
        SC: tl.constexpr,
        SC_P: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        PD: tl.constexpr,
        PH: tl.constexpr,
        PW: tl.constexpr,
        RELU_IN: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        UNROLL: tl.constexpr,
        BP: tl.constexpr,
    ):
        KT: tl.constexpr = KD * KH * KW

        b, g, p, pm, d_, h_, w_ = _tile(SP, G, D, H, W, BP)

        i_ = tl.arange(0, SC_P)
        im = i_ < SC

        xbase = (b * G + g) * (M * SP)
        sbase = (b * G + g) * (SC * SP)
        wg = g * (M * SC * KT)

        acc_n = tl.zeros((SC_P, BP), dtype=tl.float32)
        acc_d = tl.zeros((SC_P, BP), dtype=tl.float32)

        if UNROLL:
            for kd in tl.static_range(KD):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        acc_n, acc_d, _ = _adjoint_tap(
                            acc_n, acc_d, acc_n, X, T1, T1, WT, xbase, wg, p, pm,
                            d_, h_, w_, D, H, W, i_, im, kd, kh, kw, SP,
                            M, SC, KD, KH, KW, PD, PH, PW, 2,
                        )
        else:
            for tau in tl.range(0, KT):
                acc_n, acc_d, _ = _adjoint_tap(
                    acc_n, acc_d, acc_n, X, T1, T1, WT, xbase, wg, p, pm,
                    d_, h_, w_, D, H, W, i_, im,
                    tau // (KH * KW), (tau // KW) % KH, tau % KW, SP,
                    M, SC, KD, KH, KW, PD, PH, PW, 2,
                )

        off = sbase + i_[:, None] * SP + p[None, :]
        omask = im[:, None] & pm[None, :]
        s = tl.load(Z + off, mask=omask, other=0.0)
        if HAS_BIAS:
            s = s + tl.load(BIAS + g * SC + i_, mask=im, other=0.0)[:, None]
        if RELU_IN:
            s = tl.maximum(s, 0.0)
        # `ieee_rounding`: Triton's plain `/` lowers to the 2-ulp `div.full`, which would
        # round the layer's output differently from the reference's `torch.div`
        tl.store(
            OUT + off,
            tl.fdiv(s * (acc_n + eps), acc_d + eps, ieee_rounding=True),
            mask=omask,
        )

    # -------------------------------------------------------------- fused bwd update

    @triton.autotune(configs=_CONV_CONFIGS, key=["SP", "G"])
    @triton.jit
    def _deconv_bwd_kernel(
        X,
        T1,
        Z,
        GOUT,
        WT,
        BIAS,
        GNUM,
        GDEN,
        SP,
        G,
        D,
        H,
        W,
        eps,
        M: tl.constexpr,
        SC: tl.constexpr,
        SC_P: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        PD: tl.constexpr,
        PH: tl.constexpr,
        PW: tl.constexpr,
        RELU_IN: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        UNROLL: tl.constexpr,
        BP: tl.constexpr,
    ):
        KT: tl.constexpr = KD * KH * KW

        b, g, p, pm, d_, h_, w_ = _tile(SP, G, D, H, W, BP)

        i_ = tl.arange(0, SC_P)
        im = i_ < SC

        xbase = (b * G + g) * (M * SP)
        sbase = (b * G + g) * (SC * SP)
        wg = g * (M * SC * KT)

        acc_n = tl.zeros((SC_P, BP), dtype=tl.float32)
        acc_d = tl.zeros((SC_P, BP), dtype=tl.float32)

        if UNROLL:
            for kd in tl.static_range(KD):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        acc_n, acc_d, _ = _adjoint_tap(
                            acc_n, acc_d, acc_n, X, T1, T1, WT, xbase, wg, p, pm,
                            d_, h_, w_, D, H, W, i_, im, kd, kh, kw, SP,
                            M, SC, KD, KH, KW, PD, PH, PW, 2,
                        )
        else:
            for tau in tl.range(0, KT):
                acc_n, acc_d, _ = _adjoint_tap(
                    acc_n, acc_d, acc_n, X, T1, T1, WT, xbase, wg, p, pm,
                    d_, h_, w_, D, H, W, i_, im,
                    tau // (KH * KW), (tau // KW) % KH, tau % KW, SP,
                    M, SC, KD, KH, KW, PD, PH, PW, 2,
                )

        off = sbase + i_[:, None] * SP + p[None, :]
        omask = im[:, None] & pm[None, :]
        s = tl.load(Z + off, mask=omask, other=0.0)
        if HAS_BIAS:
            s = s + tl.load(BIAS + g * SC + i_, mask=im, other=0.0)[:, None]
        if RELU_IN:
            s = tl.maximum(s, 0.0)
        gv = tl.load(GOUT + off, mask=omask, other=0.0)
        n = acc_n + eps
        inv = tl.fdiv(1.0, acc_d + eps, ieee_rounding=True)
        gn = gv * s * inv
        tl.store(GNUM + off, gn, mask=omask)
        tl.store(GDEN + off, -(gn * n) * inv, mask=omask)

    # ------------------------------------------------------------- fused source grad

    @triton.autotune(configs=_CONV_CONFIGS, key=["SP", "G"])
    @triton.jit
    def _deconv_gz_kernel(
        X,
        T1,
        Z,
        GOUT,
        GT1,
        WT,
        BIAS,
        GZ,
        SP,
        G,
        D,
        H,
        W,
        eps,
        M: tl.constexpr,
        SC: tl.constexpr,
        SC_P: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        PD: tl.constexpr,
        PH: tl.constexpr,
        PW: tl.constexpr,
        RELU_IN: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        UNROLL: tl.constexpr,
        BP: tl.constexpr,
    ):
        """``gz = mask * (g * (A(x) + eps) / (A(t1) + eps) + A(gt1))``.

        Fusing the direct term with the ``A(gt1)`` application means the direct term is
        never written to (nor read back from) global memory.
        """
        KT: tl.constexpr = KD * KH * KW

        b, g, p, pm, d_, h_, w_ = _tile(SP, G, D, H, W, BP)

        i_ = tl.arange(0, SC_P)
        im = i_ < SC

        xbase = (b * G + g) * (M * SP)
        sbase = (b * G + g) * (SC * SP)
        wg = g * (M * SC * KT)

        acc_n = tl.zeros((SC_P, BP), dtype=tl.float32)
        acc_d = tl.zeros((SC_P, BP), dtype=tl.float32)
        acc_g = tl.zeros((SC_P, BP), dtype=tl.float32)

        if UNROLL:
            for kd in tl.static_range(KD):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        acc_n, acc_d, acc_g = _adjoint_tap(
                            acc_n, acc_d, acc_g, X, T1, GT1, WT, xbase, wg, p, pm,
                            d_, h_, w_, D, H, W, i_, im, kd, kh, kw, SP,
                            M, SC, KD, KH, KW, PD, PH, PW, 3,
                        )
        else:
            for tau in tl.range(0, KT):
                acc_n, acc_d, acc_g = _adjoint_tap(
                    acc_n, acc_d, acc_g, X, T1, GT1, WT, xbase, wg, p, pm,
                    d_, h_, w_, D, H, W, i_, im,
                    tau // (KH * KW), (tau // KW) % KH, tau % KW, SP,
                    M, SC, KD, KH, KW, PD, PH, PW, 3,
                )

        off = sbase + i_[:, None] * SP + p[None, :]
        omask = im[:, None] & pm[None, :]
        gout = tl.load(GOUT + off, mask=omask, other=0.0)
        res = tl.fdiv(gout * (acc_n + eps), acc_d + eps, ieee_rounding=True) + acc_g
        if RELU_IN:
            s = tl.load(Z + off, mask=omask, other=0.0)
            if HAS_BIAS:
                s = s + tl.load(BIAS + g * SC + i_, mask=im, other=0.0)[:, None]
            res = tl.where(s > 0, res, 0.0)
        tl.store(GZ + off, res, mask=omask)


    # ------------------------------------------------------------------ filter grads

    @triton.jit
    def _wgrad_kernel(
        A,
        BT,
        OUT,
        BIAS,
        SP,
        B,
        G,
        D,
        H,
        W,
        CA: tl.constexpr,
        CB: tl.constexpr,
        CA_P: tl.constexpr,
        CB_P: tl.constexpr,
        KT_P: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        PD: tl.constexpr,
        PH: tl.constexpr,
        PW: tl.constexpr,
        RELU_A: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        NBLK: tl.constexpr,
        BP: tl.constexpr,
    ):
        """``out[blk, g, ca, cb, tau] = sum_{b,p} A[b, ca, p] * BT[b, cb, p + pad - tau]``."""
        KT: tl.constexpr = KD * KH * KW
        blk = tl.program_id(0)
        g = tl.program_id(1)

        tau = tl.arange(0, KT_P)
        taum = tau < KT
        kd = tau // (KH * KW)
        kh = (tau // KW) % KH
        kw = tau % KW
        dd = PD - kd
        dh = PH - kh
        dw = PW - kw
        toff = dd * (H * W) + dh * W + dw

        ca_i = tl.arange(0, CA_P)
        cb_i = tl.arange(0, CB_P)
        cam = ca_i < CA
        cbm = cb_i < CB

        acc = tl.zeros((CA_P, CB_P, KT_P, BP), dtype=tl.float32)

        nchunk = tl.cdiv(SP, BP)
        for b in range(B):
            abase = (b * G + g) * (CA * SP)
            bbase = (b * G + g) * (CB * SP)
            for chunk in range(blk, nchunk, NBLK):
                p = chunk * BP + tl.arange(0, BP)
                pm = p < SP
                w_ = p % W
                h_ = (p // W) % H
                d_ = p // (W * H)
                av = tl.load(
                    A + abase + ca_i[:, None] * SP + p[None, :],
                    mask=cam[:, None] & pm[None, :],
                    other=0.0,
                )
                if HAS_BIAS:
                    av = tl.where(
                        cam[:, None] & pm[None, :],
                        av + tl.load(BIAS + g * CA + ca_i, mask=cam, other=0.0)[:, None],
                        0.0,
                    )
                if RELU_A:
                    av = tl.maximum(av, 0.0)
                bm = (
                    taum[:, None]
                    & pm[None, :]
                    & (d_[None, :] + dd[:, None] >= 0)
                    & (d_[None, :] + dd[:, None] < D)
                    & (h_[None, :] + dh[:, None] >= 0)
                    & (h_[None, :] + dh[:, None] < H)
                    & (w_[None, :] + dw[:, None] >= 0)
                    & (w_[None, :] + dw[:, None] < W)
                )
                bv = tl.load(
                    BT
                    + bbase
                    + cb_i[:, None, None] * SP
                    + toff[None, :, None]
                    + p[None, None, :],
                    mask=cbm[:, None, None] & bm[None, :, :],
                    other=0.0,
                )
                acc += av[:, None, None, :] * bv[None, :, :, :]

        res = tl.sum(acc, axis=3)
        obase = (blk * G + g) * (CA * CB * KT)
        ooff = (ca_i[:, None, None] * CB + cb_i[None, :, None]) * KT + tau[None, None, :]
        tl.store(
            OUT + obase + ooff,
            res,
            mask=cam[:, None, None] & cbm[None, :, None] & taum[None, None, :],
        )


# --------------------------------------------------------------------------- wrappers


class _Launch(NamedTuple):
    """Everything the kernels need about a call's geometry.

    1-, 2- and 3-D inputs are all addressed as ``(D, H, W)`` with the missing leading
    dimensions set to 1 and their kernel taps to a single unpadded tap, so one kernel body
    serves every rank.
    """

    sp: int  # D * H * W, the flattened spatial extent
    dhw: tuple[int, int, int]
    ks: tuple[int, int, int]
    pad: tuple[int, int, int]
    bg: int  # batch * groups

    @property
    def taps(self) -> int:
        return self.ks[0] * self.ks[1] * self.ks[2]

    @property
    def grid(self):
        """Flat 1-D grid.  A 2-D grid would put ``batch * groups`` on dimension y, which
        CUDA caps at 65535 -- reachable with a large batch at the wide stages, where it
        fails the launch outright."""
        return lambda meta: (triton.cdiv(self.sp, meta["BP"]) * self.bg,)


def _launch(inp: Tensor, kernel_size: Sequence[int], groups: int) -> _Launch:
    spatial = inp.shape[2:]
    if not 1 <= len(spatial) <= 3:
        raise ValueError("only 1-D, 2-D and 3-D inputs are supported")
    pad = 3 - len(spatial)
    D, H, W = [1] * pad + list(spatial)
    return _Launch(
        sp=D * H * W,
        dhw=(D, H, W),
        ks=tuple([1] * pad + list(kernel_size)),
        pad=tuple([0] * pad + [k // 2 for k in kernel_size]),
        bg=inp.shape[0] * groups,
    )


def gconv(
    inp: Tensor,
    weight: Tensor,
    *,
    groups: int,
    m: int,
    sc: int,
    kernel_size: Sequence[int],
    adjoint: bool,
    relu_input: bool = False,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Grouped convolution ``K`` (``adjoint=False``) or its adjoint ``A``.

    ``weight`` is the ``(groups, m, sc, *kernel_size)`` contiguous filter.  ``bias``, if
    given, is added to the input channels as they are loaded (zero-padded taps stay zero).
    """
    lc = _launch(inp, kernel_size, groups)
    cout = sc if adjoint else m
    out = torch.empty(
        inp.shape[0], groups * cout, *inp.shape[2:], device=inp.device, dtype=inp.dtype
    )
    _gconv_kernel[lc.grid](
        inp, weight, out, bias if bias is not None else inp,
        lc.sp, groups, *lc.dhw,
        M=m, SC=sc, COUT_P=_next_pow2(cout),
        KD=lc.ks[0], KH=lc.ks[1], KW=lc.ks[2],
        PD=lc.pad[0], PH=lc.pad[1], PW=lc.pad[2],
        ADJOINT=adjoint, RELU_IN=relu_input, HAS_BIAS=bias is not None,
        UNROLL=_unroll_factor(lc.taps, m if adjoint else sc, sc if adjoint else m),
    )
    return out


def deconv_fwd(
    x: Tensor,
    t1: Tensor,
    z: Tensor,
    weight: Tensor,
    *,
    groups: int,
    m: int,
    sc: int,
    kernel_size: Sequence[int],
    eps: float,
    relu_input: bool,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """``out = s * (A(x) + eps) / (A(t1) + eps)`` with ``s = relu(z + bias)``."""
    lc = _launch(x, kernel_size, groups)
    out = torch.empty_like(z)
    _deconv_fwd_kernel[lc.grid](
        x, t1, z, weight, bias if bias is not None else z, out,
        lc.sp, groups, *lc.dhw, eps,
        M=m, SC=sc, SC_P=_next_pow2(sc),
        KD=lc.ks[0], KH=lc.ks[1], KW=lc.ks[2],
        PD=lc.pad[0], PH=lc.pad[1], PW=lc.pad[2],
        RELU_IN=relu_input, HAS_BIAS=bias is not None,
        UNROLL=_unroll_factor(lc.taps, m, sc),
    )
    return out


def deconv_bwd(
    x: Tensor,
    t1: Tensor,
    z: Tensor,
    gout: Tensor,
    weight: Tensor,
    *,
    groups: int,
    m: int,
    sc: int,
    kernel_size: Sequence[int],
    eps: float,
    relu_input: bool,
    bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Elementwise backward of the epilogue, recomputing ``A(x)`` and ``A(t1)``.

    Returns ``(dL/dnum, dL/dden)``; the direct ``dL/ds`` term is produced later by
    :func:`deconv_gz`, which recomputes it rather than reading it back from memory.
    """
    lc = _launch(x, kernel_size, groups)
    gnum, gden = torch.empty_like(z), torch.empty_like(z)
    _deconv_bwd_kernel[lc.grid](
        x, t1, z, gout, weight, bias if bias is not None else z, gnum, gden,
        lc.sp, groups, *lc.dhw, eps,
        M=m, SC=sc, SC_P=_next_pow2(sc),
        KD=lc.ks[0], KH=lc.ks[1], KW=lc.ks[2],
        PD=lc.pad[0], PH=lc.pad[1], PW=lc.pad[2],
        RELU_IN=relu_input, HAS_BIAS=bias is not None,
        UNROLL=_unroll_factor(lc.taps, m, sc),
    )
    return gnum, gden


def deconv_gz(
    x: Tensor,
    t1: Tensor,
    z: Tensor,
    gout: Tensor,
    gt1: Tensor,
    weight: Tensor,
    *,
    groups: int,
    m: int,
    sc: int,
    kernel_size: Sequence[int],
    eps: float,
    relu_input: bool,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """``gz = mask * (gout * (A(x) + eps) / (A(t1) + eps) + A(gt1))``."""
    lc = _launch(x, kernel_size, groups)
    gz = torch.empty_like(z)
    _deconv_gz_kernel[lc.grid](
        x, t1, z, gout, gt1, weight, bias if bias is not None else z, gz,
        lc.sp, groups, *lc.dhw, eps,
        M=m, SC=sc, SC_P=_next_pow2(sc),
        KD=lc.ks[0], KH=lc.ks[1], KW=lc.ks[2],
        PD=lc.pad[0], PH=lc.pad[1], PW=lc.pad[2],
        RELU_IN=relu_input, HAS_BIAS=bias is not None,
        UNROLL=_unroll_factor(lc.taps, m, sc),
    )
    return gz


def wgrad(
    a: Tensor,
    bt: Tensor,
    *,
    groups: int,
    ca: int,
    cb: int,
    kernel_size: Sequence[int],
    relu_a: bool = False,
    bias: Optional[Tensor] = None,
    num_blocks: int = 64,
) -> Tensor:
    """``out[g, ca, cb, tau] = sum_{b, p} a[b, ca, p] * bt[b, cb, p + pad - tau]``.

    The reduction is split over spatial blocks whose partial sums are added afterwards,
    which keeps the result deterministic (cuDNN's weight gradient is not).
    """
    lc = _launch(a, kernel_size, groups)
    ca_p, cb_p, kt_p = _next_pow2(ca), _next_pow2(cb), _next_pow2(lc.taps)
    # The accumulator is a (CA_P, CB_P, KT_P, BP) register tile, so BP is the largest
    # power of two that fits the budget -- rounded *down*, and clamped to [16, 128].
    # 128 is measurably the best block at the brats23 stage-0 shape (5.3 ms against
    # 7.7 ms at 64), so the rounding direction here is load-bearing.
    budget = max(16, _WGRAD_TILE // (ca_p * cb_p * kt_p))
    block = min(1 << (budget.bit_length() - 1), 128)
    block = max(16, min(block, _next_pow2(lc.sp)))
    nblk = max(1, min(num_blocks, lc.sp // block or 1))
    partials = torch.empty(
        nblk, groups, ca, cb, lc.taps, device=a.device, dtype=torch.float32
    )
    _wgrad_kernel[(nblk, groups)](
        a, bt, partials, bias if bias is not None else a,
        lc.sp, a.shape[0], groups, *lc.dhw,
        CA=ca, CB=cb, CA_P=ca_p, CB_P=cb_p, KT_P=kt_p,
        KD=lc.ks[0], KH=lc.ks[1], KW=lc.ks[2],
        PD=lc.pad[0], PH=lc.pad[1], PW=lc.pad[2],
        RELU_A=relu_a, HAS_BIAS=bias is not None, NBLK=nblk, BP=block, num_warps=4,
    )
    return partials.sum(0).view(groups, ca, cb, *kernel_size)
