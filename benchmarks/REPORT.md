# Optimizing `Deconv` for the `deconver_brats23` configuration

## Summary

The `Deconv` layer was rewritten around one observation: **every step of the multiplicative
deconvolution update is memory-bandwidth bound**, and the original implementation moved roughly
four times more data than the algorithm requires while asking cuDNN for convolution shapes it
handles poorly.

At the `model_zoo/deconver_brats23` geometry the optimized layer is **4.8× faster in the forward
pass, 3.2× in the backward pass and 3.5× for forward+backward**, with **52% less peak memory in the
forward pass and 45% less for forward+backward**. The full network is **2.0× faster end-to-end**
with **33% less memory**. At the configuration's actual training shape (batch 2, 128³ ROI) the
original implementation runs out of memory on a 16 GB GPU where the optimized one fits.

Numerically the forward output agrees with the original to **2.9e-07** relative, and gradients to
**2.3e-04** relative — the latter is reduction-order noise in fp32, and the optimized filter
gradient is in fact ~1000× *closer* to a float64 evaluation than cuDNN's.

---

## 1. Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB (Ada, sm_89), driver 575.57.08 |
| Measured achievable bandwidth | 230 GB/s (`clone()` of a 1 GiB fp32 tensor; 288 GB/s theoretical) |
| Software | PyTorch 2.10.0+cu126, cuDNN 9.10.2, Triton 3.6.0, Python 3.12, Linux 6.8 |
| dtype | float32 (`amp: false` in the bundle config) |
| cuDNN | `torch.backends.cudnn.benchmark = True`, as set by the bundle's `initialize` section |
| Device | GPU 1 of 2 (GPU 0 was occupied by an unrelated job for the whole session) |

Layer configuration (`model_zoo/deconver_brats23/configs/train.yaml`, `network_def`):
`groups=-1`, `ratio=4`, `kernel_size=[3,3,3]`, `num_iters=1`, `num_grad_iters=null`,
`update_source=True`, `update_filter=False`, `eps=1e-16`.

With `groups=-1` the layer is fully grouped (`groups = channels`, so `m = C/groups = 1` filter
channel per group) and `ratio=4` gives `source_channels = 4`. The network contains nine `Deconv`
instances:

| stage | channels `C` | spatial | `U` = one `(B, C, *S)` fp32 tensor at B=1 |
|---|---|---|---|
| encoder 0 / decoder 3 | 32 | 128³ | 256 MiB |
| encoder 1 / decoder 2 | 64 | 64³ | 64 MiB |
| encoder 2 / decoder 1 | 128 | 32³ | 16 MiB |
| encoder 3 / decoder 0 | 256 | 16³ | 8 MiB |
| encoder 4 | 512 | 8³ | 2 MiB |

Throughout this report `U` denotes the size of one `(B, C, *S)` tensor. The *source* tensors the
layer works with have `groups × source_channels = 4C` channels, i.e. **4U** each, and the two
full-resolution layers account for 75% of the network's `Deconv` activation memory.

### What the layer computes

Per group, with `pad = k // 2`, `K` the filter application and `A` its adjoint:

```
K(u)[o, p] = Σ_{i,τ} u[i, p + τ - pad] · h[o, i, τ]        (sc → m channels)
A(v)[i, p] = Σ_{o,τ} v[o, p + pad - τ] · h[o, i, τ]        (m → sc channels,  A = Kᵀ)

s   = relu(Linear(x))                     # (B, 4C, *S)
t1  = K(s)                                # (B, C,  *S)
out = s · (A(x) + eps) / (A(t1) + eps)    # one ISRA / Lee-Seung update
```

---

## 2. Bottlenecks in the original implementation

All numbers below are measured on the stage-0 geometry (B=1, C=32, 128³, so U = 256 MiB), and
compared against the *bandwidth roofline* — the time the step would take if it only had to read its
inputs and write its outputs once at 230 GB/s.

### 2.1 The batched convolution asks cuDNN for a pathological shape

`operations.conv` folds the batch into the group dimension: it reshapes the input to
`(1, B·G·C_g, *S)`, the weight to `(B·G·C_out, C_in, *k)` and calls `F.conv3d` with
`groups = B·G`. For this configuration that is a 3-D grouped convolution with **B·32 groups of 4→1
(or 1→4) channels**. Measured against the roofline:

| step | shape | measured | roofline | off by |
|---|---|---|---|---|
| `K` = conv3d(4C→C, groups=C) | forward | 23.8 ms | 5.4 ms | 4.4× |
| `A` = conv3d(C→4C, groups=C) | forward | 15.1 ms | 5.4 ms | 2.8× |
| `K` input gradient | backward | 22.5 ms | 5.4 ms | 4.2× |
| `A` input gradient | backward | 12.4 ms | 5.4 ms | 2.3× |
| `K` weight gradient | backward | **42.6 ms** | 5.4 ms | **7.9×** |
| `A` weight gradient | backward | **36.0 ms** | 5.4 ms | **6.7×** |

The filter gradients are the single worst item in the whole layer: three of them are needed per
backward pass, 120 ms in total.

Folding the batch into the groups is also *unnecessary* here. `h = relu(h0)` is a parameter
expanded over the batch, so it is the same for every sample whenever `update_filter=False`; the
per-sample convolution then collapses exactly into an ordinary grouped convolution with `groups = G`
on the un-split `(B, C, *S)` layout. This was verified to be exact (0 to 1e-14 in float64 across
grouping/kernel/rank variations).

### 2.2 Where the forward pass actually spends its time

Kernel profile of the original layer at stage 0 (B=1, C=32, 128³), from
`benchmarks/profile_deconv.py`.  The per-kernel times sum to 117.5 ms, which matches the 117.3 ms
the benchmark reports end to end:

| kernel | calls | ms | what it is |
|---|---|---|---|
| `conv_depthwise3d` | 2 | 28.1 | the two adjoint applications `A(x)`, `A(t1)` |
| `convNd_grouped_direct_kernel` | 1 | 23.9 | the filter application `K(s)` |
| `vectorized_elementwise_kernel` | 2 | 17.1 | `+ eps`, twice |
| `vectorized_elementwise_kernel` | 1 | 12.6 | `s * numerator` |
| `vectorized_elementwise_kernel` | 1 | 12.6 | `/ denominator` |
| `vectorized_elementwise_kernel` | 2 | 8.6 | `relu(s)` and `relu(h)` |
| `elementwise_kernel` | 1 | 8.3 | the projection's **bias add** |
| `ampere_sgemm_128x64_nn` | 1 | 6.3 | the projection itself |

**Half the forward pass (59 ms of 117 ms) is elementwise work**, none of which needs its own pass
over memory.

### 2.3 The elementwise part costs more than the convolutions

`s * (conv(...) + eps) / (conv(...) + eps)` is four eager kernels over 4U tensors — `+eps` twice,
one multiply, one divide — each reading and writing its operands:

| step | traffic | measured |
|---|---|---|
| `s * (num + eps) / (den + eps)` | 40U ≈ 10.2 GiB | **42.5 ms** |

That is more than the two convolutions that produced `num` and `den` put together (measured in
isolation; in the profile above the same four kernels account for 42.3 ms).

### 2.4 Full-size tensors kept alive for the backward pass

Tracing the autograd graph shows **17U retained per layer** (plus three copies of the filter),
excluding the caller's input: `s` (4U), `num` (4U), `den` (4U), `s·num` (4U) and `t1` (1U). At batch
2 and a 128³ ROI that is 22.6 GiB of retained activations for the network's `Deconv` layers alone,
which is why the configuration's own training shape does not fit on a 16 GB GPU.

### 2.5 Smaller, but not free

* `F.relu` is applied to `h0.expand(batch, …)`, which materializes a batch-fold copy of the filter
  and forces a copy in the subsequent `Rearrange`.
* `t(flip(h))` is recomputed twice per update.
* The initializer's projection (`nn.Conv1d` 1×1) adds its bias in a *separate* elementwise kernel:
  8.9 ms of the 14.8 ms the projection costs — a full read-modify-write pass over a 4U tensor.

---

## 3. What the optimized implementation does

The layer keeps its public API, its parameters, its state-dict keys and its general code path. A
**fast path** is taken when the filter is batch-invariant and the convolutions are plain stride-1
same-padded convolutions with odd kernels (`update_source=True`, `update_filter=False`, no extra
convolution kwargs, no autocast, matching input rank); every other configuration runs the original
code unchanged.

### 3.1 Batched convolution → grouped convolution

`h = relu(h0)` is used directly as the weight of a `groups = G` convolution, and its adjoint is
obtained by flipping the spatial axes and swapping the two channel axes,
`(g, o, i, τ) → (g, i, o, flip τ)`. Nothing is expanded over the batch and nothing is rearranged.
The channel-axis swap is a no-op when `C/groups == 1` (the published configurations) but is required
for correctness as soon as `C/groups > 1`.

### 3.2 Fused Triton kernels

Because the arithmetic intensity is a handful of FLOPs per byte, the win comes from *not moving
data*. Five kernels replace the original's fifteen-odd:

| kernel | what it fuses |
|---|---|
| `_gconv_kernel` | grouped convolution `K` or `A`, with the projection bias and the ReLU folded into the load |
| `_deconv_fwd_kernel` | `A(x)`, `A(t1)` **and** the elementwise epilogue — the numerator and denominator never reach memory |
| `_deconv_bwd_kernel` | recomputes `A(x)`, `A(t1)` and emits `dL/dnum`, `dL/dden` in one pass |
| `_deconv_gz_kernel` | recomputes the direct `dL/ds` term and fuses it with `A(dL/dt1)` and the ReLU mask |
| `_wgrad_kernel` | the three filter-gradient reductions, into a deterministic partial-sum workspace |

All kernels address the volume through a flattened spatial coordinate and unroll the filter taps at
compile time, so the same code serves 1-D, 2-D and 3-D inputs.

### 3.3 Recompute instead of store

The custom autograd node keeps only `x` (which the projection retains anyway), `z` (the projection
output, *pre-bias and pre-ReLU*) and `t1` — **5U instead of 17U**. `A(x)` and `A(t1)` are recomputed
inside the backward kernels, where they cost *no extra memory traffic at all*: `x` and `t1` have to
be read there in any case, and reading them (1U each) is cheaper than reading back the stored
numerator and denominator (4U each).

The rectified source `relu(z + b)` is never materialized either: the kernels rectify and add the
bias while loading `z`. That removes two full-size passes from the forward pass and lets the
projection run without a bias epilogue.

### 3.4 Forward traffic accounting

| | original | optimized |
|---|---|---|
| projection (+ bias) | 13U | 5U |
| `relu` | 8U | fused |
| `t1 = K(s)` | 5U | 5U |
| `num = A(x)`, `+eps` | 13U | fused |
| `den = A(t1)`, `+eps` | 13U | fused |
| `s * num / den` | 24U | 10U (one kernel: reads `x`, `t1`, `z`; writes `out`) |
| **total** | **76U ≈ 19.5 GiB** | **20U ≈ 5.1 GiB** |
| predicted at 230 GB/s | 85 ms | 22.3 ms |
| **measured** | **117.4 ms** | **24.5 ms** |

The optimized forward pass runs at 91% of the achievable bandwidth; the original loses a further
27% on top of its 3.8× excess traffic because of the grouped-convolution kernels. (~1.4 ms of the
24.5 ms buys exact IEEE rounding on the final division: Triton's plain `/` lowers to the 2-ulp
`div.full`, so the kernels ask for `ieee_rounding=True` to round exactly as `torch.div` does.)

### 3.5 Profile of the optimized layer

Same geometry, same profiler (per-kernel times again sum to the end-to-end measurement):

| forward, 24.5 ms | ms | | forward + backward, 117.1 ms | calls | ms |
|---|---|---|---|---|---|
| `_deconv_fwd_kernel` | 10.1 | | `_gconv_kernel` (`K(s)`, `K(dL/dden)`, `K(dL/dnum)`) | 3 | 24.4 |
| `_gconv_kernel` (`t1 = K(s)`) | 8.2 | | `_deconv_bwd_kernel` | 1 | 18.6 |
| `ampere_sgemm` (projection) | 6.3 | | `_wgrad_kernel` (three filter gradients) | 3 | 16.0 |
| | | | `_deconv_gz_kernel` | 1 | 14.9 |
| | | | cuDNN `wgrad_alg0` (projection weight grad) | 1 | 14.7 |
| | | | `_deconv_fwd_kernel` | 1 | 10.1 |
| | | | `ampere_sgemm` / cutlass GEMM (projection) | 2 | 11.3 |

Every fused kernel is within ~10% of its bandwidth roofline. The largest remaining item that is
*not* is the projection's own backward pass (cuDNN, 19.8 ms against an 11 ms roofline), which is
the `nn.Conv1d` inside `deconver/layers/linear.py` and identical in both implementations.

---

## 4. Results

### 4.1 `Deconv` layers, batch 1 (median of 12 runs)

| layer | fwd orig | fwd opt | speedup | bwd orig | bwd opt | speedup | f+b orig | f+b opt | speedup | mem F orig | mem F opt | saved | mem F+B orig | mem F+B opt | saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C=32, 128³ | 117.44 ms | 24.50 ms | **4.79×** | 296.12 ms | 92.79 ms | **3.19×** | 413.46 ms | 117.36 ms | **3.52×** | 5376 MiB | 2560 MiB | **52.4%** | 10752 MiB | 5889 MiB | **45.2%** |
| C=64, 64³ | 29.12 ms | 5.94 ms | 4.90× | 70.13 ms | 21.46 ms | 3.27× | 99.29 ms | 27.35 ms | 3.63× | 1344 MiB | 640 MiB | 52.4% | 2688 MiB | 1474 MiB | 45.2% |
| C=128, 32³ | 7.34 ms | 1.46 ms | 5.02× | 17.47 ms | 7.00 ms | 2.50× | 24.82 ms | 8.41 ms | 2.95× | 336 MiB | 160 MiB | 52.3% | 672 MiB | 372 MiB | 44.7% |
| C=256, 16³ | 1.51 ms | 0.37 ms | 4.12× | 4.61 ms | 3.38 ms | 1.37× | 6.10 ms | 3.66 ms | 1.67× | 85 MiB | 41 MiB | 51.6% | 169 MiB | 97 MiB | 42.9% |
| C=512, 8³ | 0.33 ms | 0.14 ms | 2.44× | 1.80 ms | 0.90 ms | 1.99× | 2.14 ms | 1.01 ms | 2.13× | 25 MiB | 14 MiB | 43.3% | 47 MiB | 28 MiB | 39.2% |

Rows the fused kernels do not cover are printed with a `(general)` suffix by
`bench_deconv.py`; every row above uses them.

### 4.2 `Deconv` layers, batch 2 (the configuration's training batch size)

| layer | forward | backward | fwd + bwd | mem F | mem F+B |
|---|---|---|---|---|---|
| C=32, 128³ | 255.06 → 47.34 ms (**5.39×**) | original **out of memory** (optimized: 185.75 ms) | original **out of memory** (optimized: 233.10 ms) | 10752 → 5120 MiB (52.4%) | original **out of memory** (optimized: 11777 MiB) |
| C=64, 64³ | 58.11 → 11.98 ms (4.85×) | 139.98 → 43.06 ms (3.25×) | 198.19 → 54.93 ms (3.61×) | 2688 → 1280 MiB (52.4%) | 5376 → 2946 MiB (45.2%) |
| C=128, 32³ | 14.60 → 2.94 ms (4.97×) | 35.01 → 12.27 ms (2.85×) | 49.57 → 15.06 ms (3.29×) | 672 → 320 MiB (52.4%) | 1345 → 740 MiB (45.0%) |
| C=256, 16³ | 3.72 → 0.79 ms (4.71×) | 9.45 → 4.78 ms (1.98×) | 13.21 → 5.48 ms (2.41×) | 169 → 81 MiB (52.0%) | 338 → 189 MiB (43.9%) |
| C=512, 8³ | 0.70 → 0.24 ms (2.90×) | 3.81 → 1.30 ms (2.92×) | 4.42 → 1.48 ms (2.99×) | 47 → 24 MiB (47.6%) | 89 → 51 MiB (42.5%) |

The stage-0 row is the point of the whole exercise: at the configuration's own training shape the
original cannot complete a backward pass on a 16 GB GPU, and the optimized layer does it in
186 ms within 11.8 GiB.

### 4.3 Full `Deconver` network (brats23 `network_def`), batch 1

| ROI | fwd orig | fwd opt | speedup | bwd orig | bwd opt | speedup | f+b orig | f+b opt | speedup | mem F | mem F+B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64³ | 56.91 ms | 26.46 ms | 2.15× | 129.03 ms | 66.59 ms | 1.94× | 186.33 ms | 93.20 ms | **2.00×** | 919 → 567 MiB (38.3%) | 3311 → 2224 MiB (**32.8%**) |
| 96³ | 198.68 ms | 95.25 ms | 2.09× | 439.48 ms | 220.80 ms | 1.99× | 637.75 ms | 316.06 ms | **2.02×** | 3006 → 1818 MiB (39.5%) | 11091 → 7421 MiB (**33.1%**) |

The network-level speedup is smaller than the layer-level one because roughly half of the network's
time is spent outside `Deconv` — in the MLP blocks, instance norms, and the up/down-sampling
convolutions, none of which were touched.

---

## 5. Correctness

`tests/test_deconv_equivalence.py` (16 tests, 28 sub-tests; all pass on CUDA and on CPU) compares
the optimized module against a verbatim copy of the original
(`deconver/deconvolution/reference.py`):

* forward output and the gradients of the input, `h0`, and the projection weight and bias, over
  every `Deconv` geometry of the brats23 network and over grouping / kernel / rank / batch
  variations;
* every subset of inputs that requires a gradient, so the branches that skip work are
  exercised too;
* configurations that must fall back to the general path (`update_filter=True`,
  `update_source=False`, extra conv kwargs), which must stay **bit-identical** to the original;
* `state_dict` round-trips in both directions, `fit`/`reconstruct`/`loss`, non-contiguous inputs;
* the filter gradient against a float64 ground truth;
* a **whole-network** check: a checkpoint written by the original `Deconver` loads into the
  optimized one with `strict=True` and reproduces its output and every gradient.

### Observed errors

`benchmarks/accuracy_report.py` evaluates both implementations in fp32 on the GPU and compares them
with the *same* computation carried out in float64 on the CPU.

| quantity | max abs error | max relative error |
|---|---|---|
| forward output | 4.8e-07 | **5.1e-07** |
| gradients (input, `h0`, projection weight/bias) | 3.1e-03 (on gradients of magnitude up to 2.6e+02) | **1.2e-05** |

(`python benchmarks/accuracy_report.py --bias 1.0`: nine layer geometries, batch 2, TF32 off. The
relative figure is `max|Δ| / max|reference|`, i.e. normalized by the tensor's own scale — individual
near-zero entries can differ by more in relative terms. With TF32 left on — the PyTorch default —
the same run reports 2.9e-07 and 2.3e-04, because both implementations then share the projection's
TF32 error. The report labels each case `[fused]` or `[torch]` so it is visible which backend was
measured; the `C/G = 4, fused` case exists specifically to cover the adjoint weight's
`(g, o, i) → (g, i, o)` regrouping on the Triton path.)

The gradient differences are *not* a systematic bias. Measured against the float64 ground truth
across all 40 reported tensors, the optimized implementation is farther from exact in 17 cases and
closer in 23, and its worst error over the whole sweep is smaller than the original's
(2.8e-05 against 4.0e-05). The differences come from summation order — and, with TF32 enabled,
overwhelmingly from TF32.

Isolating one filter-gradient step shows where the accuracy actually goes. The measurement below
is the weight gradient of `K` — the `aten::convolution_backward(..., groups=C, output_mask=[F,T,F])`
call the reference's autograd makes, which the profile shows as
`cudnn::cnn::wgrad2d_grouped_direct_kernel` — at B=2, C=32, 16³, against a float64 CPU reference:

| | cuDNN grouped weight gradient | fused reduction |
|---|---|---|
| `cudnn.allow_tf32 = True` (PyTorch's default) | 2.30e-04 | 1.13e-07 |
| `cudnn.allow_tf32 = False` | 1.84e-06 | 1.13e-07 |

So cuDNN's grouped weight gradient runs on **TF32 tensor cores** by default and gives up three
digits; with TF32 disabled the fused reduction is still ~16× closer to exact, which is the part
attributable to reduction order. (Both figures are stable across input distributions: with
half-normal sources they are 2.65e-04 / 1.31e-06 against 1.21e-07.) End-to-end the effect is much
smaller — only one of the three filter-gradient contributions goes through that call, and the shared
1×1 projection dominates — which is why `benchmarks/accuracy_report.py` disables cuDNN's TF32 by
default, so that its "vs float64" columns reflect the deconvolution rather than the projection.

The Triton kernels always compute in true fp32 and their reduction is a fixed partial-sum tree, so
everything the fused kernels produce is bitwise reproducible run to run.

Two properties are preserved exactly rather than approximately: the association
`(s · num) / den` (rewriting it as `s · (num / den)` moves fp32 results by up to 2e-07 relative),
and the placement of `eps` as an addition on each convolution output (in fp32 `+1e-16` is not a
no-op below ≈1.9e-9, so replacing it with a clamp would change results).

---

## 6. Behaviour across the configuration grid

`benchmarks/sweep_configs.py` sweeps `ratio` x `groups` x `kernel_size` and, for each of the
30 combinations, checks the optimized layer against the original and measures both. At
C=32, 32^3, batch 2:

| | |
|---|---|
| worst forward-output error (TF32 off) | **1.35e-06** |
| worst gradient error (TF32 off) | **5.21e-06** |
| configurations slower or heavier than the original | **0 / 30** |
| configurations accelerated by the fused kernels | 14 / 30 (1.11x - 3.12x on forward+backward, 22-37% less forward memory) |
| configurations that fall back to the original path | 16 / 30 (1.00x, identical memory) |

The dispatch is deliberately conservative: the layer takes the fused path only where it is
*measured* to win, and otherwise runs the original code unchanged, so no configuration
regresses. Three boundaries came out of the sweep, each with a mechanism behind it:

* **`source_channels = 1` with a large kernel** (e.g. `ratio=1, groups=-1, k=5^3`). The
  fusion earns its keep by collapsing the elementwise passes over the
  `groups * source_channels` source tensor; with one source channel that tensor is no
  bigger than the input, so there is nothing to amortize, while a 125-tap filter is compute
  bound where cuDNN's depthwise kernels are strong. Measured 0.82x, so it is excluded --
  `sc = 2` is 1.36x and `sc = 4` is 2.93x at the same kernel size.
* **More than 16 channels per group.** The fused kernels hold one accumulator tile per
  source channel; past 16 they spill. Measured at `m=8, sc=32, k=3^3`: **0.06x** -- a 16x
  regression, excluded by the lane cap.
* **Dense geometries** (`groups=1`, or generally `m * sc * taps` above the unroll budget).
  These are ordinary convolutions where cuDNN's algorithm selection for the original's
  batched form is competitive; they run the original path.

A fourth boundary is about compile time rather than throughput: the filter taps are
unrolled in the Triton frontend only for small kernels. Unrolling 5^3 taps with four source
channels took **434 seconds** to compile (and the unrolled kernel spilled, running 9x
slower than the rolled one); with the loop left rolled the same configuration compiles in
**1.6 s**. The 3x3x3 filters the published configurations use are still unrolled, which is
worth ~13% there.

---

## 7. Subpackage structure

Five files, each with one job:

| file | lines | role |
|---|---:|---|
| `kernels.py` | 894 | the five Triton kernels and their launch wrappers |
| `deconv.py` | 295 | `Initializer` / `Deconv`; dispatch, plus the original general path |
| `functional.py` | 207 | the fused update as a single autograd node |
| `operations.py` | 172 | the original public helpers, untouched (used by the general path) |
| `reference.py` | 273 | verbatim copy of the original layer; test/benchmark ground truth only |

---

## 8. Reproducing

| script | what it answers |
|---|---|
| `bench_deconv.py` | latency and peak memory, per `Deconv` geometry and for the whole network |
| `sweep_configs.py` | correctness and efficiency over the `ratio` x `groups` x `kernel_size` grid |
| `accuracy_report.py` | how far each implementation is from a float64 evaluation of itself |
| `profile_deconv.py` | which kernels the time actually goes to |

```bash
# correctness
pytest tests/test_deconv_equivalence.py -o addopts=""

# accuracy against a float64 ground truth
python benchmarks/accuracy_report.py              # random regime (ill-conditioned)
python benchmarks/accuracy_report.py --bias 1.0   # well-conditioned regime

# the configuration grid: correctness and efficiency for ratio x groups x kernel_size
python benchmarks/sweep_configs.py --channels 32 --spatial 32 --batch 2            # speed
python benchmarks/sweep_configs.py --channels 32 --spatial 32 --batch 2 --no-tf32  # accuracy

# performance
python benchmarks/bench_deconv.py --layers --batch 1
python benchmarks/bench_deconv.py --layers --batch 2
python benchmarks/bench_deconv.py --model  --batch 1 --roi 64
python benchmarks/bench_deconv.py --model  --batch 1 --roi 96

# kernel-level profile
python benchmarks/profile_deconv.py --channels 32 --spatial 128 --batch 1
```

Benchmark methodology: both implementations run in the same process with the same weights, shapes,
dtype and device; every measurement is preceded by warm-up iterations that populate the cuDNN
algorithm cache and the Triton autotuning cache; peak-memory counters are reset *after* the warm-up
(cuDNN's `benchmark=True` algorithm search transiently allocates workspaces far larger than the ones
it settles on, which would otherwise be reported as the peak — this is what makes an unwarmed
measurement report 8 GB for a forward pass that actually uses 640 MiB); `torch.cuda.synchronize()`
brackets every timed region; the median of `--iters` runs is reported.

---

## 9. Scope, caveats and what is left

* **Unchanged behaviour.** The fast path is only taken when it is exactly equivalent. Everything
  else — `update_filter=True`, `num_iters` with `update_filter`, even kernel sizes, extra
  convolution keyword arguments, autocast, `verbose`, `update_source=False` — runs the original
  code, bit-for-bit.
* **Backends.** The fused kernels require CUDA, fp32, contiguous inputs, tensors below 2³¹
  elements, at most 16 channels per group, and a filter geometry within the unroll budget
  (`taps ≤ 128`, `m · sc · taps ≤ 8192`). Every configuration outside that — including all CPU
  and float64 use — runs the original code path, unchanged and therefore never slower. There is
  no second implementation of the fast-path mathematics: an earlier pure-PyTorch backend was
  removed once the sweep showed it was not reliably ahead of the original (cuDNN often picks
  better algorithms for the original's batched convolution form), so the update exists once, in
  the kernels.
* **TF32 is the dominant numerical effect on this GPU, and it is not ours.** On the full
  network, the original's own fp32 CUDA output sits ~9e-04 (output) to ~7e-03 (gradients) away
  from a float64 evaluation of itself, because cuDNN runs the *untouched* 3×3×3 and 1×1
  convolutions on TF32 tensor cores. The optimized network is marginally *closer* to exact
  (8e-04). `tests/test_deconv_equivalence.py::TestNetworkCompatibility` therefore asserts
  "no further from float64 than the original" rather than an absolute tolerance.
* **Autograd surface.** The fused update is a single autograd node, so it is *not*
  double-differentiable. A gradient-penalty style `create_graph=True` **raises** rather than
  silently dropping the second-order term — note that `@once_differentiable` does *not* achieve
  this (the engine simply never re-enters the node), so the check is explicit.
  `torch.func.grad` / `jacrev` / `vmap` work and match the original exactly: under a
  `torch.func` transform the layer runs the general path, since functorch's wrapped tensors have
  no storage for Triton to address. A subclass that overrides `update_s`, `update`, `iterative_update` or the
  initializer's `forward` automatically falls back to the general path, so those extension
  points keep working.
* **Pre-existing numerical issue (not introduced here).** With `num_iters ≥ 2` and gradients enabled
  for more than one iteration, the original algorithm produces non-finite filter gradients whenever
  a voxel has `A(K(s)) == 0` exactly: `den = eps = 1e-16` gives a ~1e16 factor that compounds across
  iterations. The optimized implementation reproduces this faithfully (same non-finite pattern);
  it is a property of the update rule, worth knowing before enabling multi-iteration training.
* **Remaining headroom.** Every optimized kernel now runs at the bandwidth roofline except the
  projection's backward pass (cuDNN, 20 ms vs an 11 ms roofline at stage 0) — replacing it with
  explicit GEMMs was measured and rejected: it is faster at batch 1 but 4× *slower* at batch 2, and
  it silently switches the projection from cuDNN's TF32 to true fp32. Fusing the three
  filter-gradient reductions into the convolutions that read the same tensors would save a further
  ~8U of traffic (~9 ms of the 90 ms backward at stage 0); it was left out because the two kernels
  want incompatible register tilings.
