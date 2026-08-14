"""Numerical-accuracy report for the optimized ``Deconv``.

Both the original implementation (``deconver.deconvolution.reference.ReferenceDeconv``)
and the optimized one are evaluated in fp32 on the GPU and compared against the *same*
computation carried out in float64 on the CPU by the original implementation, which serves
as the ground truth.  Reporting it this way separates two different things:

* how far the optimized fp32 result is from the original fp32 result (the "equivalence"
  error, which is what a user of the layer observes), and
* how far each of them is from exact arithmetic.

By default TF32 is disabled for cuDNN convolutions.  Both implementations call the same
1x1 ``conv1d`` for the initializer projection, and on an Ampere-or-later GPU cuDNN runs it
on TF32 tensor cores, which costs ~3e-04 of relative accuracy -- an order of magnitude more
than anything the deconvolution rewrite contributes.  It cancels in the "opt vs orig"
column but would otherwise swamp both "vs f64" columns.  Pass ``--allow-tf32`` to see the
numbers as they are in a default training run.

Usage::

    python benchmarks/accuracy_report.py [--batch 2] [--spatial 12] [--bias 1.0]
"""

import argparse

import torch

import deconver as dc
from deconver.deconvolution.reference import ReferenceDeconv


# (channels, ratio, groups, kernel_size) of every Deconv in model_zoo/deconver_brats23,
# plus a few other geometries the layer supports.
CASES = [
    ("brats23 stage C=32", dict(channels=32, ratio=4, groups=-1, kernel_size=(3, 3, 3))),
    ("brats23 stage C=64", dict(channels=64, ratio=4, groups=-1, kernel_size=(3, 3, 3))),
    ("brats23 stage C=128", dict(channels=128, ratio=4, groups=-1, kernel_size=(3, 3, 3))),
    ("brats23 stage C=256", dict(channels=256, ratio=4, groups=-1, kernel_size=(3, 3, 3))),
    ("brats23 stage C=512", dict(channels=512, ratio=4, groups=-1, kernel_size=(3, 3, 3))),
    ("fives/glas 2-D k=5", dict(channels=32, ratio=4, groups=-1, kernel_size=(5, 5))),
    ("grouped C/G=4", dict(channels=12, ratio=2, groups=3, kernel_size=(3, 3, 3))),
    ("dense groups=1", dict(channels=8, ratio=1, groups=1, kernel_size=(3, 3, 3))),
    # C/G > 1 with a filter geometry small enough for the fused kernels, so that the
    # (g, o, i) -> (g, i, o) regrouping of the adjoint weight is covered on that path too
    ("grouped C/G=4, fused", dict(channels=8, ratio=1, groups=2, kernel_size=(3, 3, 3))),
]


def run(module, x, grad, dtype, device):
    module = module.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype).detach().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    out = module(x)
    out.backward(grad.to(device=device, dtype=dtype))
    grads = {"input": x.grad.detach().double().cpu()}
    for name, p in module.named_parameters():
        grads[name] = p.grad.detach().double().cpu()
    return out.detach().double().cpu(), grads


def err(a, b):
    """Max abs error and max error relative to the largest reference magnitude."""
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--spatial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bias", type=float, default=0.0,
        help="constant added to the initializer bias.  With bias >= ~1 the source stays "
             "strictly positive, so the denominator A(K(s)) + eps is bounded away from "
             "eps and the update is well conditioned; with the default 0 a sizeable "
             "fraction of the voxels have s = 0, den = 1e-16 and a ratio of ~1e16, which "
             "amplifies fp32 rounding in the *reference* algorithm itself.",
    )
    parser.add_argument(
        "--allow-tf32", action="store_true",
        help="leave cuDNN's TF32 tensor cores enabled (the PyTorch default)",
    )
    args = parser.parse_args()
    if not args.allow_tf32:
        torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {dev}  batch: {args.batch}  spatial: {args.spatial}^d  "
          f"bias: {args.bias}  cudnn tf32: {torch.backends.cudnn.allow_tf32}  "
          f"(float64 CPU ground truth)\n")
    hdr = (f"{'case':<29} {'tensor':<24} {'|opt-orig|':>11} {'rel':>9} "
           f"{'orig vs f64':>12} {'opt vs f64':>11}")
    print(hdr)
    print("-" * len(hdr))

    worst_fwd = 0.0
    worst_grad = 0.0
    for label, kwargs in CASES:
        p = len(kwargs["kernel_size"])
        spatial = (args.spatial,) * p
        torch.manual_seed(args.seed)
        ref = ReferenceDeconv(**kwargs)
        torch.manual_seed(args.seed)
        new = dc.Deconv(**kwargs)
        if args.bias:
            with torch.no_grad():
                ref.init.linear.linear.bias.add_(args.bias)
        new.load_state_dict(ref.state_dict())

        torch.manual_seed(args.seed + 1)
        x = torch.rand(args.batch, kwargs["channels"], *spatial)
        out_shape = (args.batch, new.groups * new.source_channels, *spatial)
        grad = torch.randn(*out_shape)

        # say which path the "optimized" column exercised: geometries the fused kernels do
        # not cover run the original code, and then the two columns are the same code
        fused = new.fast_path_applicable(x.to(dev))
        label = f"{label} [{'fused' if fused else 'general'}]"

        exact_out, exact_grads = run(ref, x, grad, torch.float64, "cpu")
        ref_out, ref_grads = run(ref, x, grad, torch.float32, dev)
        new_out, new_grads = run(new, x, grad, torch.float32, dev)

        rows = [("output", new_out, ref_out, exact_out)]
        for name in ref_grads:
            rows.append((f"grad {name}", new_grads[name], ref_grads[name], exact_grads[name]))

        for i, (name, a, b, e) in enumerate(rows):
            ae, re = err(a, b)
            _, re_ref = err(b, e)
            _, re_new = err(a, e)
            if i == 0:
                worst_fwd = max(worst_fwd, re)
            else:
                worst_grad = max(worst_grad, re)
            print(f"{label if i == 0 else '':<29} {name:<24} {ae:11.3e} {re:9.2e} "
                  f"{re_ref:12.2e} {re_new:11.2e}")
        print()

    print(f"worst forward-output relative error : {worst_fwd:.3e}")
    print(f"worst gradient relative error       : {worst_grad:.3e}")


if __name__ == "__main__":
    main()
