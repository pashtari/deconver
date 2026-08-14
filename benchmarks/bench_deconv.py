"""Benchmark the optimized ``Deconv`` against the original implementation.

Measures forward latency, backward latency, combined forward+backward latency, and peak
GPU memory (forward, and forward+backward) for

* every ``Deconv`` geometry of the ``model_zoo/deconver_brats23`` network, and
* the full ``deconver.Deconver`` network built from that bundle's ``network_def``.

Methodology
-----------
* the two implementations run in the *same* process, on the same device, dtype, shapes and
  weights (the optimized module loads the reference module's ``state_dict``);
* every measurement is preceded by warm-up iterations, which populate the cuDNN algorithm
  cache and the Triton autotuning cache, so that neither is charged to the timings;
* peak-memory counters are reset *after* the warm-up (cuDNN's ``benchmark=True`` algorithm
  search transiently allocates workspaces far larger than the ones it ends up using, which
  would otherwise be reported as the peak) and read right after a synchronized run;
* ``torch.cuda.synchronize()`` brackets every timed region and the median of ``--iters``
  runs is reported.

Examples::

    python benchmarks/bench_deconv.py --layers --batch 1
    python benchmarks/bench_deconv.py --model --batch 1 --roi 64
    python benchmarks/bench_deconv.py --layers --batch 2 --json results.json
"""

import argparse
import gc
import json
import statistics
import time

import torch

import deconver as dc
import deconver.deconver as deconver_module
from deconver.deconvolution.reference import ReferenceDeconv


# ``Deconv`` geometries of model_zoo/deconver_brats23 at a 128^3 ROI:
# encoder widths (32, 64, 128, 256, 512) with strides (1, 2, 2, 2, 2) and the mirrored
# decoder (256, 128, 64, 32).  Each entry is (channels, spatial size, occurrences).
BRATS_LAYERS = [
    (32, (128, 128, 128), 2),
    (64, (64, 64, 64), 2),
    (128, (32, 32, 32), 2),
    (256, (16, 16, 16), 2),
    (512, (8, 8, 8), 1),
]

DECONV_KWARGS = dict(ratio=4, groups=-1, kernel_size=(3, 3, 3), num_iters=1)

NETWORK_DEF = dict(
    in_channels=4,
    out_channels=3,
    spatial_dims=3,
    encoder_depth=(1, 1, 1, 1, 1),
    encoder_width=(32, 64, 128, 256, 512),
    strides=(1, 2, 2, 2, 2),
    decoder_depth=(1, 1, 1, 1),
    norm=torch.nn.InstanceNorm3d,
    act=torch.nn.ReLU,
    groups=-1,
    ratio=4,
    kernel_size=(3, 3, 3),
    num_iters=1,
    num_grad_iters=None,
    mlp_ratio=4,
)


def _uses_fused_path(module, x):
    """Whether any Deconv in ``module`` runs on the fused kernels for this input."""
    from deconver.deconvolution.deconv import Deconv

    layers = [module] if isinstance(module, Deconv) else [
        m for m in module.modules() if isinstance(m, Deconv)
    ]
    return any(layer.fast_path_applicable(x) for layer in layers)


def _sync():
    torch.cuda.synchronize()


def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def guarded(fn):
    """Run ``fn``, returning None if it runs out of memory.

    The traceback of an OOM keeps every frame -- and therefore every tensor -- of the
    failed call alive, which would make the *next* measurement OOM too, so it is dropped
    explicitly before collecting.  Phases are guarded individually: a configuration whose
    backward pass does not fit still reports a valid forward-pass measurement.
    """
    try:
        return fn()
    except torch.OutOfMemoryError as exc:
        exc.__traceback__ = None
        _clear()
        return None


def timed(fn, iters, warmup):
    """Median wall-clock time of ``fn`` in ms, with warm-up and CUDA synchronization."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def peak_mb(fn, warmup=2):
    """Steady-state peak allocated memory of ``fn`` in MiB (warm-up excluded)."""
    for _ in range(warmup):
        fn()
    _sync()
    _clear()
    fn()
    _sync()
    return torch.cuda.max_memory_allocated() / 2**20


def measure(make_module, x_shape, args, label):
    """Latency and memory of one module on one input shape.

    Each phase is measured under its own out-of-memory guard, so a shape whose backward
    pass does not fit still yields forward-pass numbers.  Missing entries are ``None``.
    """
    res = dict(label=label, fwd_ms=None, bwd_ms=None, fwd_bwd_ms=None,
               fwd_mb=None, fwd_bwd_mb=None, fused=None)
    dev = torch.device(args.device)
    module = guarded(lambda: make_module().to(dev))
    if module is None:
        return res
    x = guarded(lambda: torch.rand(*x_shape, device=dev, requires_grad=True))
    if x is None:
        return res
    res["fused"] = _uses_fused_path(module, x)

    def forward_only():
        with torch.no_grad():
            module(x)

    _clear()
    res["fwd_ms"] = guarded(lambda: timed(forward_only, args.iters, args.warmup))
    res["fwd_mb"] = guarded(lambda: peak_mb(forward_only))

    grad = guarded(lambda: torch.randn_like(module(x)).detach())
    if grad is not None:

        def fwd_bwd():
            module.zero_grad(set_to_none=True)
            x.grad = None
            module(x).backward(grad)

        _clear()
        res["fwd_bwd_ms"] = guarded(lambda: timed(fwd_bwd, args.iters, args.warmup))
        res["fwd_bwd_mb"] = guarded(lambda: peak_mb(fwd_bwd))

        # backward alone: build the graph once, then re-run backward with retain_graph
        _clear()
        out = guarded(lambda: module(x))
        if out is not None:

            def bwd_only():
                module.zero_grad(set_to_none=True)
                x.grad = None
                out.backward(grad, retain_graph=True)

            res["bwd_ms"] = guarded(lambda: timed(bwd_only, args.iters, args.warmup))
        out = None
    # drop the references before clearing, so the allocator can reuse the memory for the
    # next measurement (rebinding rather than `del`: the closures above still hold them)
    module = x = grad = None
    _clear()
    return res


def _row(name, ref, new):
    def ratio(a, b):
        return f"{a / b:5.2f}x" if a and b else "  oom" if not (a and b) else "  n/a"

    def pct(a, b):
        return f"{100 * (a - b) / a:6.1f}%" if a and b else "    oom"

    def num(v, w, prec=2):
        return f"{v:{w}.{prec}f}" if v is not None else f"{'oom':>{w}}"

    ref, new = ref or {}, new or {}
    return (
        f"{name:>22} | {num(ref.get('fwd_ms'), 8)} {num(new.get('fwd_ms'), 8)}"
        f" {ratio(ref.get('fwd_ms'), new.get('fwd_ms'))} |"
        f" {num(ref.get('bwd_ms'), 8)} {num(new.get('bwd_ms'), 8)}"
        f" {ratio(ref.get('bwd_ms'), new.get('bwd_ms'))} |"
        f" {num(ref.get('fwd_bwd_ms'), 8)} {num(new.get('fwd_bwd_ms'), 8)}"
        f" {ratio(ref.get('fwd_bwd_ms'), new.get('fwd_bwd_ms'))} |"
        f" {num(ref.get('fwd_mb'), 8, 0)} {num(new.get('fwd_mb'), 8, 0)}"
        f" {pct(ref.get('fwd_mb'), new.get('fwd_mb'))} |"
        f" {num(ref.get('fwd_bwd_mb'), 9, 0)} {num(new.get('fwd_bwd_mb'), 9, 0)}"
        f" {pct(ref.get('fwd_bwd_mb'), new.get('fwd_bwd_mb'))}"
    )


def _report(name, ref, new):
    """Print one comparison row; phases that ran out of memory show as ``oom``."""
    print(_row(name, ref, new))


HEADER = (
    f"{'case':>22} | {'fwd orig':>8} {'fwd opt':>8} {'speedup':>6} |"
    f" {'bwd orig':>8} {'bwd opt':>8} {'speedup':>6} |"
    f" {'f+b orig':>8} {'f+b opt':>8} {'speedup':>6} |"
    f" {'memF org':>8} {'memF opt':>8} {'saved':>7} |"
    f" {'memFB org':>9} {'memFB opt':>9} {'saved':>7}"
)


def bench_layers(args):
    results = []
    print("\n=== Deconv layers (model_zoo/deconver_brats23 geometries) "
          f"batch={args.batch} dtype=float32 ===")
    print(HEADER)
    for channels, spatial, _ in BRATS_LAYERS:
        shape = (args.batch, channels, *spatial)
        name = f"C={channels} {spatial[0]}^3"

        def make_ref(c=channels):
            torch.manual_seed(0)
            return ReferenceDeconv(channels=c, **DECONV_KWARGS)

        def make_new(c=channels):
            torch.manual_seed(0)
            return dc.Deconv(channels=c, **DECONV_KWARGS)

        ref = measure(make_ref, shape, args, f"{name} original")
        new = measure(make_new, shape, args, f"{name} optimized")
        # a row without the fused kernels runs the original code path on both sides
        _report(name if new.get("fused") else f"{name} (general)", ref, new)
        results.append(dict(case=name, shape=list(shape), original=ref, optimized=new))
    return results


def bench_model(args):
    roi = (args.roi,) * 3
    shape = (args.batch, NETWORK_DEF["in_channels"], *roi)
    print(f"\n=== Full Deconver network (brats23 network_def) batch={args.batch} "
          f"roi={roi} dtype=float32 ===")
    print(HEADER)

    def make(reference):
        def factory():
            original = deconver_module.Deconv
            if reference:
                deconver_module.Deconv = ReferenceDeconv
            try:
                torch.manual_seed(0)
                return dc.Deconver(**NETWORK_DEF)
            finally:
                deconver_module.Deconv = original

        return factory

    ref = measure(make(True), shape, args, "network original")
    new = measure(make(False), shape, args, "network optimized")
    _report(f"Deconver {args.roi}^3", ref, new)
    return [dict(case=f"network roi={args.roi}", shape=list(shape),
                 original=ref, optimized=new)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--roi", type=int, default=64, help="cubic ROI for --model")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", action="store_true", help="benchmark Deconv layers")
    parser.add_argument("--model", action="store_true", help="benchmark the full network")
    parser.add_argument("--json", default=None, help="write raw measurements here")
    args = parser.parse_args()
    if not (args.layers or args.model):
        args.layers = args.model = True

    torch.backends.cudnn.benchmark = True  # as set by the bundle's initialize section
    dev = torch.device(args.device)
    print(f"device: {torch.cuda.get_device_name(dev)}")
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"cudnn {torch.backends.cudnn.version()}")
    try:
        import triton

        print(f"triton {triton.__version__}")
    except ImportError:
        print("triton: not available")

    out = {}
    if args.layers:
        out["layers"] = bench_layers(args)
    if args.model:
        out["model"] = bench_model(args)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
