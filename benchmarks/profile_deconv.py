"""Kernel-level profile of the original and the optimized ``Deconv``.

Prints, for one layer geometry, the CUDA kernels that dominate the forward and the
backward pass of each implementation.  This is what identifies *which* step is the
bottleneck, rather than inferring it from the code.

The absolute times carry CUPTI instrumentation overhead and are systematically higher than
the ones ``bench_deconv.py`` reports (more so for the original, which launches more
kernels); read this as an attribution of where the time goes, and take end-to-end latency
from the benchmark.

Usage::

    python benchmarks/profile_deconv.py --channels 32 --spatial 128 --batch 1
"""

import argparse

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import deconver as dc
from deconver.deconvolution.reference import ReferenceDeconv


def summarize(prof, title, limit, iters):
    """Flat per-kernel GPU profile.

    Uses *self* device time: ``device_time_total`` includes the time of child events, so
    summing it over ``key_averages()`` counts every kernel once per enclosing aten op and
    inflates the total several-fold.
    """
    # keep only real GPU kernels: CPU-side events (aten ops, autograd nodes) report the
    # device time of the kernels they launched, which would count every kernel twice
    events = [
        e for e in prof.key_averages()
        if e.self_device_time_total > 0 and e.device_type == DeviceType.CUDA
    ]
    events.sort(key=lambda e: -e.self_device_time_total)
    total = sum(e.self_device_time_total for e in events)
    print(f"\n--- {title}: {total / (1e3 * iters):.2f} ms of GPU time per iteration ---")
    print(f"{'kernel':<62} {'ms/iter':>8} {'%':>6} {'calls':>6}")
    for e in events[:limit]:
        name = e.key if len(e.key) <= 60 else e.key[:57] + "..."
        print(f"{name:<62} {e.self_device_time_total / (1e3 * iters):8.2f} "
              f"{100 * e.self_device_time_total / total:6.1f} {e.count // iters:6d}")
    return total / iters


ITERS = 3


def run_profile(module, x, grad, label, limit, backward):
    for _ in range(3):  # warm up allocator, cuDNN algorithm cache and Triton autotuning
        if backward:
            module(x).backward(grad)
        else:
            with torch.no_grad():
                module(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 acc_events=True) as prof:
        for _ in range(ITERS):
            module.zero_grad(set_to_none=True)
            x.grad = None
            if backward:
                module(x).backward(grad)
            else:
                with torch.no_grad():
                    module(x)
        torch.cuda.synchronize()
    return summarize(prof, label, limit, ITERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--spatial", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    kwargs = dict(channels=args.channels, ratio=4, groups=-1, kernel_size=(3, 3, 3),
                  num_iters=1)
    spatial = (args.spatial,) * 3
    unit = args.batch * args.channels * args.spatial**3 * 4 / 2**30
    print(f"{torch.cuda.get_device_name(dev)} | batch={args.batch} "
          f"channels={args.channels} spatial={spatial} fp32")
    print(f"U (one (B, C, *S) tensor) = {unit * 1024:.0f} MiB; "
          f"the source tensors are 4U each")

    for name, cls in (("original", ReferenceDeconv), ("optimized", dc.Deconv)):
        torch.manual_seed(0)
        module = cls(**kwargs).to(dev)
        x = torch.rand(args.batch, args.channels, *spatial, device=dev, requires_grad=True)
        grad = torch.randn(args.batch, args.channels * 4, *spatial, device=dev)
        run_profile(module, x, grad, f"{name}: forward (no grad)", args.limit, False)
        run_profile(module, x, grad, f"{name}: forward + backward", args.limit, True)
        del module, x, grad
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
