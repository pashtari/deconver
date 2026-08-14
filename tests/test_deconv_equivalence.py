"""Equivalence tests: optimized ``Deconv`` vs the original implementation.

The original code is preserved verbatim in
:mod:`deconver.deconvolution.reference` and is the ground truth here.  The tests
compare forward outputs and the gradients of every input and trainable parameter,
for the fast path (fused Triton kernels and the pure-PyTorch backend) as well as for
the configurations that must fall back to the general path.

Run with::

    pytest tests/test_deconv_equivalence.py -v
"""

import unittest

import torch

import deconver as dc
import deconver.deconver as deconver_module
from deconver.deconvolution import kernels
from deconver.deconvolution.reference import ReferenceDeconv


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# (channels, ratio, groups, kernel_size, spatial_size) -- the first five entries are the
# per-stage geometries of the model_zoo/deconver_brats23 network.
BRATS_STAGES = [
    (32, (12, 14, 13)),
    (64, (10, 9, 11)),
    (128, (8, 8, 8)),
    (256, (6, 5, 4)),
    (512, (3, 4, 3)),
]


def _pair(reference_kwargs, seed=0, bias=0.0):
    """Build a reference module and an optimized module sharing the same weights."""
    torch.manual_seed(seed)
    ref = ReferenceDeconv(**reference_kwargs).to(DEVICE)
    torch.manual_seed(seed)
    new = dc.Deconv(**reference_kwargs).to(DEVICE)
    if bias:
        with torch.no_grad():
            ref.init.linear.linear.bias.add_(bias)
    new.load_state_dict(ref.state_dict())
    return ref, new


def _errors(a, b):
    """(max abs error, max relative error) of ``a`` against reference ``b``.

    The relative error is normalized by the largest magnitude of the reference tensor,
    which is the meaningful scale for tensors that mix very small and very large entries.
    """
    diff = (a.double() - b.double()).abs()
    scale = b.double().abs().max().clamp_min(1e-30)
    return diff.max().item(), (diff.max() / scale).item()


class TestDeconvEquivalence(unittest.TestCase):
    """Forward / backward equivalence against the original implementation."""

    def _compare(self, kwargs, batch, spatial, rtol=1e-5, grad_rtol=1e-3,
                 msg="", seed=0, bias=1.0):
        """Compare forward output and every gradient against the original implementation.

        ``bias`` shifts the initializer bias so that the source stays strictly positive.
        Without it a large fraction of the voxels have ``s = 0`` and therefore
        ``den = eps = 1e-16``; the resulting ~1e16 ratios amplify fp32 rounding *in the
        reference algorithm itself* to ~1e-2 relative, which would make any tolerance
        meaningless.  The well-conditioned regime is the one the trained model operates
        in, and is where equivalence is meaningful.  The tolerances here are the ones
        measured by ``python benchmarks/accuracy_report.py --bias 1.0``.
        """
        ref, new = _pair(kwargs, seed=seed, bias=bias)
        torch.manual_seed(seed + 1)
        # inputs are non-negative in the model (DeconvMixer applies ReLU before Deconv)
        x0 = torch.rand(batch, kwargs["channels"], *spatial, device=DEVICE)
        xr = x0.clone().requires_grad_(True)
        xn = x0.clone().requires_grad_(True)

        yr = ref(xr)
        yn = new(xn)
        self.assertEqual(yr.shape, yn.shape, f"{msg}: output shape mismatch")
        ae, re = _errors(yn, yr)
        self.assertLess(re, rtol, f"{msg}: forward rel err {re:.3e} (abs {ae:.3e})")

        torch.manual_seed(seed + 2)
        g = torch.randn_like(yr)
        yr.backward(g)
        yn.backward(g)

        grads = [("input", xr.grad, xn.grad)]
        for (name, pr), (_, pn) in zip(ref.named_parameters(), new.named_parameters()):
            grads.append((name, pr.grad, pn.grad))

        worst = {}
        for name, gr, gn in grads:
            self.assertIsNotNone(gn, f"{msg}: missing gradient for {name}")
            ae, re = _errors(gn, gr)
            worst[name] = (ae, re)
            # gradients are reductions over the whole volume; the summation order of the
            # fused kernels differs from cuDNN's, which costs ~1e-4 of relative agreement
            # (both are equally far from a float64 evaluation, see
            # test_filter_gradient_accuracy_vs_float64 and benchmarks/accuracy_report.py)
            self.assertLess(re, grad_rtol,
                            f"{msg}: {name} grad rel err {re:.3e} (abs {ae:.3e})")
        return worst

    def test_forward_backward_brats_stages(self):
        """Every Deconv geometry of the brats23 model."""
        for channels, spatial in BRATS_STAGES:
            kwargs = dict(
                channels=channels, ratio=4, groups=-1, kernel_size=(3, 3, 3), num_iters=1
            )
            with self.subTest(channels=channels):
                self._compare(kwargs, 2, spatial, msg=f"C={channels}")

    def test_forward_backward_geometries(self):
        """Grouping / kernel / rank variations that still take the fast path."""
        cases = [
            (dict(channels=12, ratio=2, groups=3, kernel_size=(3, 3, 3)), (7, 6, 5)),
            (dict(channels=8, ratio=1, groups=1, kernel_size=(3, 3, 3)), (6, 6, 6)),
            (dict(channels=16, ratio=4, groups=-1, kernel_size=(3, 3)), (17, 15)),
            (dict(channels=6, ratio=2, groups=2, kernel_size=(3, 5, 3)), (6, 7, 5)),
            (dict(channels=8, ratio=2, groups=-1, kernel_size=(5,)), (11,)),
            (dict(channels=10, ratio=3, groups=5, kernel_size=(3, 3)), (9, 8)),
        ]
        for kwargs, spatial in cases:
            kwargs = dict(num_iters=1, **kwargs)
            with self.subTest(**kwargs):
                self._compare(kwargs, 2, spatial, rtol=1e-5, msg=str(kwargs))

    def test_batch_sizes(self):
        for batch in (1, 3):
            kwargs = dict(channels=16, ratio=4, groups=-1, kernel_size=(3, 3, 3), num_iters=1)
            with self.subTest(batch=batch):
                self._compare(kwargs, batch, (9, 8, 7), rtol=1e-5)

    def test_multiple_iterations(self):
        """num_iters > 1 (only the last iterations carry gradients)."""
        # (3, 3) is the first schedule that back-propagates through several
        # multiplicative updates; it is only well behaved when the source stays positive,
        # which `bias` ensures (see the module docstring of functional.py)
        for num_iters, num_grad_iters in [(3, 1), (5, 1), (2, 2), (3, 3)]:
            kwargs = dict(
                channels=8, ratio=2, groups=-1, kernel_size=(3, 3, 3),
                num_iters=num_iters, num_grad_iters=num_grad_iters,
            )
            with self.subTest(num_iters=num_iters, num_grad_iters=num_grad_iters):
                ref, new = _pair(kwargs, bias=1.0)
                x0 = torch.rand(2, 8, 6, 6, 6, device=DEVICE)
                xr, xn = (x0.to(DEVICE).clone().requires_grad_(True) for _ in range(2))
                yr, yn = ref(xr), new(xn)
                ae, re = _errors(yn, yr)
                self.assertLess(re, 1e-4, f"forward rel err {re:.3e}")
                g = torch.randn_like(yr)
                yr.backward(g)
                yn.backward(g)
                for (name, pr), (_, pn) in zip(
                    ref.named_parameters(), new.named_parameters()
                ):
                    if pr.grad is None:  # the original detaches when num_grad_iters < num_iters
                        self.assertIsNone(pn.grad, f"{name} should have no gradient")
                        continue
                    _, re = _errors(pn.grad, pr.grad)
                    self.assertLess(re, 5e-3, f"{name} grad rel err {re:.3e}")

    def test_fallback_configurations_are_bitwise_identical(self):
        """Configurations outside the fast path must reproduce the original exactly."""
        cases = [
            dict(channels=8, ratio=2, groups=-1, kernel_size=(3, 3), update_filter=True,
                 num_iters=3),
            dict(channels=8, ratio=2, groups=4, kernel_size=(3, 3, 3), update_source=False),
            dict(channels=8, ratio=2, groups=-1, kernel_size=(3, 3, 3), num_iters=1,
                 stride=1, dilation=1),
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                ref, new = _pair(kwargs)
                self.assertFalse(new.fast_path_applicable())
                spatial = (6, 6, 6) if len(kwargs["kernel_size"]) == 3 else (6, 6)
                x0 = torch.rand(2, 8, *spatial, device=DEVICE)
                xr, xn = (x0.to(DEVICE).clone().requires_grad_(True) for _ in range(2))
                yr, yn = ref(xr), new(xn)
                self.assertTrue(torch.equal(yr, yn), "fallback forward is not bitwise equal")
                g = torch.randn_like(yr)
                yr.backward(g)
                yn.backward(g)
                # the backward runs the very same code, but cuDNN's own kernels are not
                # run-to-run deterministic, so compare tightly rather than bitwise
                pairs = [(xr.grad, xn.grad)]
                pairs += [(pr.grad, pn.grad) for (_, pr), (_, pn)
                          in zip(ref.named_parameters(), new.named_parameters())]
                for gr, gn in pairs:
                    if gr is None:
                        self.assertIsNone(gn)
                        continue
                    # multi-iteration mode can produce non-finite filter gradients in the
                    # original algorithm itself (den = eps gives ~1e16 ratios that compound
                    # across iterations); require the same non-finite pattern and compare
                    # the finite entries
                    finite = torch.isfinite(gr)
                    self.assertTrue(torch.equal(finite, torch.isfinite(gn)))
                    if finite.any():
                        _, re = _errors(gn[finite], gr[finite])
                        self.assertLess(re, 1e-6)

    def test_partial_gradient_requests(self):
        """Every subset of inputs that requires grad must match the original.

        The fused node returns gradients selectively, so the branches that skip work are
        worth exercising: an input that does not require grad must not perturb the ones
        that do.
        """
        kwargs = dict(channels=16, ratio=4, groups=-1, kernel_size=(3, 3, 3), num_iters=1)
        names = ["init.h0", "init.linear.linear.weight", "init.linear.linear.bias"]
        for wanted in ([], ["init.h0"], ["init.linear.linear.bias"], names):
            for input_grad in (False, True):
                with self.subTest(params=wanted, input_grad=input_grad):
                    ref, new = _pair(kwargs, bias=1.0)
                    x0 = torch.rand(2, 16, 7, 6, 8, device=DEVICE)
                    outs = []
                    for module in (ref, new):
                        for name, p in module.named_parameters():
                            p.requires_grad_(name in wanted)
                        x = x0.clone().requires_grad_(input_grad)
                        y = module(x)
                        if wanted or input_grad:
                            y.backward(torch.ones_like(y))
                        outs.append(
                            [y.detach()]
                            + [x.grad]
                            + [dict(module.named_parameters())[n].grad for n in wanted]
                        )
                    for a, b in zip(outs[1], outs[0]):
                        self.assertEqual(a is None, b is None)
                        if a is not None:
                            _, re = _errors(a, b)
                            self.assertLess(re, 1e-3)

    def test_state_dict_round_trip(self):
        """Checkpoints written by the original must load into the optimized module."""
        kwargs = dict(channels=16, ratio=4, groups=-1, kernel_size=(3, 3, 3))
        ref, new = _pair(kwargs)
        self.assertEqual(list(ref.state_dict().keys()), list(new.state_dict().keys()))
        for k, v in ref.state_dict().items():
            self.assertEqual(v.shape, new.state_dict()[k].shape, k)
        new.load_state_dict(ref.state_dict(), strict=True)
        ref.load_state_dict(new.state_dict(), strict=True)
        self.assertEqual(new.groups, ref.groups)
        self.assertEqual(new.source_channels, ref.source_channels)

    def test_fit_and_reconstruct(self):
        kwargs = dict(channels=12, ratio=2, groups=3, kernel_size=(3, 3, 3), num_iters=1)
        ref, new = _pair(kwargs)
        x = torch.rand(2, 12, 7, 6, 5, device=DEVICE)
        sr, hr = ref.fit(x)
        sn, hn = new.fit(x)
        self.assertEqual(hr.shape, hn.shape)
        self.assertEqual(hr.is_contiguous(), hn.is_contiguous())
        self.assertTrue(torch.equal(hr, hn), "fit filters differ")
        _, re = _errors(sn, sr)
        self.assertLess(re, 1e-5)
        _, re = _errors(new.reconstruct(sn, hn), ref.reconstruct(sr, hr))
        self.assertLess(re, 1e-5)
        # loss() takes channel-split tensors, the way iterative_update() calls it
        args = (new.split_channels(x), new.split_channels(sn), new.split_channels(hn))
        _, re = _errors(new.loss(*args), ref.loss(*args))
        self.assertLess(re, 1e-5)

    def test_non_contiguous_input(self):
        kwargs = dict(channels=8, ratio=2, groups=-1, kernel_size=(3, 3, 3), num_iters=1)
        ref, new = _pair(kwargs)
        base = torch.rand(2, 8, 6, 6, 12, device=DEVICE)
        x = base[..., ::2]
        self.assertFalse(x.is_contiguous())
        _, re = _errors(new(x), ref(x))
        self.assertLess(re, 1e-5)

    @unittest.skipUnless(CUDA and kernels.HAS_TRITON, "requires Triton on CUDA")
    def test_filter_gradient_accuracy_vs_float64(self):
        """The fused weight-gradient reduction is at least as accurate as cuDNN's."""
        kwargs = dict(channels=16, ratio=4, groups=-1, kernel_size=(3, 3, 3), num_iters=1)
        ref, new = _pair(kwargs)
        x0 = torch.rand(2, 16, 9, 8, 7, device=DEVICE)
        g = torch.randn(2, 64, 9, 8, 7, device=DEVICE)

        def run(module, dtype, device=DEVICE):
            mod = module.to(device=device, dtype=dtype)
            # clone: `.to()` is a no-op when dtype and device already match, and the
            # gradient would then accumulate into `x0` across the three runs
            x = x0.to(device=device, dtype=dtype).clone().requires_grad_(True)
            mod.zero_grad(set_to_none=True)
            mod(x).backward(g.to(device=device, dtype=dtype))
            out = mod.init.h0.grad.double().cpu().clone()
            module.to(device=DEVICE, dtype=torch.float32)
            return out

        exact = run(ref, torch.float64, "cpu")
        e_ref = (run(ref, torch.float32) - exact).abs().max().item()
        e_new = (run(new, torch.float32) - exact).abs().max().item()
        self.assertLessEqual(
            e_new, max(e_ref, 1e-6) * 1.5,
            f"optimized h0 grad error {e_new:.3e} exceeds reference error {e_ref:.3e}",
        )


class TestNetworkCompatibility(unittest.TestCase):
    """A checkpoint written by the original network must load and reproduce it."""

    NETWORK = dict(
        in_channels=4, out_channels=3, spatial_dims=3, encoder_depth=(1, 1, 1),
        encoder_width=(8, 16, 32), strides=(1, 2, 2), decoder_depth=(1, 1),
        norm=torch.nn.InstanceNorm3d, act=torch.nn.ReLU, groups=-1, ratio=4,
        kernel_size=(3, 3, 3), num_iters=1, mlp_ratio=4,
    )

    @staticmethod
    def _network(reference):
        original = deconver_module.Deconv
        if reference:
            deconver_module.Deconv = ReferenceDeconv
        try:
            torch.manual_seed(0)
            return dc.Deconver(**TestNetworkCompatibility.NETWORK).to(DEVICE).eval()
        finally:
            deconver_module.Deconv = original

    def test_checkpoint_loads_and_matches(self):
        """The optimized network must be no further from exact than the original.

        An absolute tolerance would be the wrong test here: on CUDA the *untouched* parts
        of the network (cuDNN's 3x3x3 and 1x1 convolutions) run on TF32 tensor cores, so
        the original network's own fp32 result sits ~1e-3 (output) to ~1e-2 (gradients)
        away from a float64 evaluation of itself -- far more than anything the
        deconvolution rewrite contributes.  So compare both against float64 and require
        the optimized one to be at least as close.
        """
        old = self._network(reference=True)
        new = self._network(reference=False)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        new.load_state_dict(old.state_dict(), strict=True)

        torch.manual_seed(1)
        x0 = torch.rand(2, 4, 16, 16, 16)
        g0 = torch.randn(2, 3, 16, 16, 16)

        def run(module, dtype, device):
            module = module.to(device=device, dtype=dtype)
            # clone: `.to()` is a no-op when dtype and device already match, and the
            # gradient would then accumulate into `x0` across the three runs
            x = x0.to(device=device, dtype=dtype).clone().requires_grad_(True)
            module.zero_grad(set_to_none=True)
            out = module(x)
            out.backward(g0.to(device=device, dtype=dtype))
            # clone: `.double().cpu()` is a no-op (same object) when the tensor already
            # has that dtype and device, and `module.to(...)` below would then move the
            # captured gradients along with the parameters
            tensors = [out.detach().double().cpu().clone(), x.grad.double().cpu().clone()]
            tensors += [p.grad.double().cpu().clone() for p in module.parameters()
                        if p.grad is not None]
            module.to(device=DEVICE, dtype=torch.float32)
            return tensors

        exact = run(old, torch.float64, "cpu")
        ref32 = run(old, torch.float32, DEVICE)
        new32 = run(new, torch.float32, DEVICE)
        self.assertEqual(len(exact), len(new32))
        for i, (e, r, n) in enumerate(zip(exact, ref32, new32)):
            _, e_ref = _errors(r, e)
            _, e_new = _errors(n, e)
            self.assertLessEqual(
                e_new, max(e_ref, 1e-5) * 1.5,
                f"tensor {i}: optimized error {e_new:.3e} exceeds the original's "
                f"{e_ref:.3e}",
            )


if __name__ == "__main__":
    unittest.main()
