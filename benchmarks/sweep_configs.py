"""Correctness and efficiency of the optimized ``Deconv`` across a grid of configurations.

Sweeps ``ratio`` x ``groups`` x ``kernel_size`` and, for every combination, checks the
optimized layer against the original (`deconver.deconvolution.reference.ReferenceDeconv`)
and measures both implementations' latency and peak memory.

Every row reports which path the optimized layer actually took:

``fused``
    the Triton kernels;
``general``
    the original code path, used wherever the fused kernels do not apply or are not
    measurably ahead -- so those rows are expected to read 1.00x with identical memory.

Usage::

    python benchmarks/sweep_configs.py                     # default grid
    python benchmarks/sweep_configs.py --channels 32 --spatial 24 --batch 2
    python benchmarks/sweep_configs.py --json sweep.json
"""

import argparse
import gc
import itertools
import json
import statistics
import time

import torch

import deconver as dc
from deconver.deconvolution.reference import ReferenceDeconv


RATIOS = (1, 2, 4)
GROUPS = (1, 2, 4, 8, -1)
KERNELS = ((3, 3, 3), (5, 5, 5))


def _clear(collect=False):
    # gc.collect() walks the whole object graph and costs ~1 s once Triton's caches are
    # populated; it is only needed between configurations, not between measurements
    if collect:
        gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def peak_mb(fn, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    _clear()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def errors(a, b):
    diff = (a.double() - b.double()).abs().max().item()
    return diff / max(b.double().abs().max().item(), 1e-30)


def path_of(module, x):
    return "fused" if module.fast_path_applicable(x) else "general"


def measure(module, x, grad, args):
    """(fwd ms, bwd ms, fwd+bwd ms, fwd MiB, fwd+bwd MiB)."""

    def reset():
        # the previous measurement leaves `x.grad` and the parameter grads allocated,
        # which would otherwise be charged to whichever implementation is measured second
        module.zero_grad(set_to_none=True)
        x.grad = None

    def forward_only():
        with torch.no_grad():
            module(x)

    def fwd_bwd():
        module.zero_grad(set_to_none=True)
        x.grad = None
        module(x).backward(grad)

    reset()
    _clear()
    t_f = timed(forward_only, args.iters, args.warmup)
    reset()
    _clear()
    m_f = peak_mb(forward_only)
    _clear()
    t_fb = timed(fwd_bwd, args.iters, args.warmup)
    reset()
    _clear()
    m_fb = peak_mb(fwd_bwd)
    reset()
    _clear()
    out = module(x)

    def bwd_only():
        module.zero_grad(set_to_none=True)
        x.grad = None
        out.backward(grad, retain_graph=True)

    t_b = timed(bwd_only, args.iters, args.warmup)
    out = None  # rebind rather than `del`: the closure above still refers to it
    _clear()
    return t_f, t_b, t_fb, m_f, m_fb


def run_case(kwargs, args, dev):
    torch.manual_seed(0)
    ref = ReferenceDeconv(**kwargs).to(dev)
    torch.manual_seed(0)
    new = dc.Deconv(**kwargs).to(dev)
    # a positive bias keeps the source away from zero, where the update's 1/eps ratios
    # amplify fp32 rounding in the reference algorithm itself
    with torch.no_grad():
        ref.init.linear.linear.bias.add_(1.0)
    new.load_state_dict(ref.state_dict())

    spatial = (args.spatial,) * len(kwargs["kernel_size"])
    torch.manual_seed(1)
    x0 = torch.rand(args.batch, kwargs["channels"], *spatial, device=dev)
    xr = x0.clone().requires_grad_(True)
    xn = x0.clone().requires_grad_(True)

    yr, yn = ref(xr), new(xn)
    torch.manual_seed(2)
    g = torch.randn_like(yr)
    yr.backward(g)
    yn.backward(g)

    worst_fwd = errors(yn, yr)
    worst_grad = errors(xn.grad, xr.grad)
    for (_, pr), (_, pn) in zip(ref.named_parameters(), new.named_parameters()):
        worst_grad = max(worst_grad, errors(pn.grad, pr.grad))

    backend = path_of(new, x0)
    del yr, yn, xr, xn
    _clear()

    x = x0.clone().requires_grad_(True)
    grad = torch.randn(*new(x).shape, device=dev)
    ref_perf = measure(ref, x, grad, args)
    new_perf = measure(new, x, grad, args)
    del ref, new, x, grad, x0
    _clear(collect=True)
    return backend, worst_fwd, worst_grad, ref_perf, new_perf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--spatial", type=int, default=24)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", default=None)
    parser.add_argument("--groups", type=int, nargs="*", default=None,
                        help="restrict the groups axis")
    parser.add_argument("--ratios", type=int, nargs="*", default=None,
                        help="restrict the ratio axis")
    parser.add_argument("--no-tf32", action="store_true",
                        help="disable cuDNN's TF32 tensor cores, which both implementations "
                             "use by default and which dominate the reported errors for the "
                             "dense (large channels-per-group) configurations")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
    print(f"{torch.cuda.get_device_name(dev)} | torch {torch.__version__} | "
          f"channels={args.channels} spatial={args.spatial}^3 batch={args.batch} fp32 | "
          f"cudnn tf32: {torch.backends.cudnn.allow_tf32}")

    header = (f"{'ratio':>5} {'groups':>6} {'kernel':>9} {'m':>3} {'sc':>4} {'backend':>8} | "
              f"{'fwd err':>8} {'grad err':>9} | {'fwd x':>6} {'bwd x':>6} {'f+b x':>6} | "
              f"{'memF %':>7} {'memFB %':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    grid = itertools.product(args.ratios or RATIOS, args.groups or GROUPS, KERNELS)
    for ratio, groups, kernel in grid:
        kwargs = dict(channels=args.channels, ratio=ratio, groups=groups,
                      kernel_size=kernel, num_iters=1)
        try:
            backend, e_f, e_g, ref_p, new_p = run_case(kwargs, args, dev)
        except torch.OutOfMemoryError:
            _clear()
            print(f"{ratio:>5} {groups:>6} {str(kernel):>9} {'':>3} {'':>4} "
                  f"{'oom':>8} |")
            continue
        g = args.channels if groups == -1 else groups
        m = args.channels // g
        sc = round(args.channels * ratio / g)
        speed = [r / n if n > 0 else float("nan") for r, n in zip(ref_p[:3], new_p[:3])]
        mem = [100 * (r - n) / r for r, n in zip(ref_p[3:], new_p[3:])]
        print(f"{ratio:>5} {groups:>6} {str(kernel):>9} {m:>3} {sc:>4} {backend:>8} | "
              f"{e_f:8.1e} {e_g:9.1e} | {speed[0]:6.2f} {speed[1]:6.2f} {speed[2]:6.2f} | "
              f"{mem[0]:6.1f}% {mem[1]:7.1f}%")
        rows.append(dict(ratio=ratio, groups=groups, kernel=list(kernel), m=m, sc=sc,
                         backend=backend, fwd_err=e_f, grad_err=e_g,
                         original=ref_p, optimized=new_p))

    if rows:
        print(f"\nworst forward error: {max(r['fwd_err'] for r in rows):.2e}   "
              f"worst gradient error: {max(r['grad_err'] for r in rows):.2e}")
        # a 3% band: configurations that fall back to the original code path run the very
        # same kernels, so their ratio fluctuates around 1.0 from run to run
        slow = [r for r in rows
                if r["original"][2] / r["optimized"][2] < 0.97
                or r["optimized"][4] > r["original"][4] * 1.03]
        if slow:
            print(f"configurations slower or heavier than the original: "
                  f"{len(slow)}/{len(rows)}")
            for r in slow:
                print(f"  ratio={r['ratio']} groups={r['groups']} k={tuple(r['kernel'])} "
                      f"[{r['backend']}]: f+b {r['original'][2]:.2f} -> "
                      f"{r['optimized'][2]:.2f} ms, memFB {r['original'][4]:.0f} -> "
                      f"{r['optimized'][4]:.0f} MiB")
        else:
            print(f"none of the {len(rows)} configurations is slower or heavier than the "
                  f"original")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
