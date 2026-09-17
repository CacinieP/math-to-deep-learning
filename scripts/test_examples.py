"""Numerical regression checks against the actual Markdown examples.

Run: python scripts/test_examples.py
Requires: torch, numpy. Tests use small CPU tensors; no datasets or training downloads.
Fenced fragments are syntax-checked, while selected definitions are exercised below.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import re
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.func import jvp
from torch.autograd import grad

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def source(pattern):
    paths = list(ROOT.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return paths[0]


def definitions(pattern):
    """Load the documented definitions, not copied implementations or training loops."""
    p = source(pattern)
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=math, Normal=Normal,
              kl_divergence=kl_divergence, jvp=jvp, grad=grad)
    for code in BLOCK.findall(p.read_text()):
        tree = ast.parse(code)
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(p), 'exec'), ns)
    return ns


class Examples(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        torch.set_num_threads(1)

    def test_all_python_fences_parse(self):
        count = 0
        for p in ROOT.rglob('*.md'):
            if '.repo_memory' in p.parts or any(part.startswith('.') for part in p.relative_to(ROOT).parts):
                continue
            for code in BLOCK.findall(p.read_text()):
                ast.parse(code, filename=str(p))
                count += 1
        self.assertGreaterEqual(count, 100)

    def test_pca_matches_svd_subspace(self):
        pca = definitions('PART-03*/轴线A*/01*.md')['pca_manual']
        x = torch.randn(40, 5, dtype=torch.float64)
        _, q, ratio = pca(x, 2)
        centered = x - x.mean(0)
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        torch.testing.assert_close(q @ q.T, vh[:2].T @ vh[:2])
        torch.testing.assert_close(ratio, (s[:2] ** 2).sum() / (s ** 2).sum())

    def test_recommender_ignores_missing_neighbor_rating(self):
        ns = definitions('PART-03*/轴线A*/02*.md')
        r = torch.tensor([[5., 4., 0.], [5., 4., 0.], [4., 3., 5.]])
        self.assertAlmostEqual(float(ns['user_based_cf'](r, 0, 2)), 5.)
        with self.assertRaises(ValueError):
            ns['matrix_factorization'](torch.zeros(2, 2), epochs=1)

    def test_rk4_converges_for_linear_ode(self):
        rk4 = definitions('PART-03*/轴线A*/03*.md')['rk4_step']
        def solve(h):
            y = torch.tensor(1., dtype=torch.float64)
            for i in range(round(1 / h)):
                y = rk4(lambda t, y: -y, i * h, y, h)
            return abs(y.item() - math.exp(-1))
        self.assertLess(solve(.1), solve(.2) / 10)

    def test_bayesian_network_elbo_and_gradients(self):
        ns = definitions('PART-03*/轴线B*/01*.md')
        model = ns['BayesianMLP'](4, 8, 3).double()
        x, y = torch.randn(6, 4, dtype=torch.float64), torch.randint(0, 3, (6,))
        torch.manual_seed(1)
        large = model.elbo_loss(x, y, dataset_size=60)
        torch.manual_seed(1)
        small = model.elbo_loss(x, y, dataset_size=6)
        torch.testing.assert_close(small - large, model.kl_divergence() * (1/6 - 1/60))
        large.backward()
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_mc_dropout_preserves_batchnorm(self):
        predict = definitions('PART-03*/轴线B*/01*.md')['mc_dropout_predict']
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5), nn.Linear(3, 2))
        model.train()
        original = model[0].running_mean.clone()
        mean, std, entropy = predict(model, torch.randn(8, 3), n_samples=5)
        torch.testing.assert_close(model[0].running_mean, original)
        self.assertTrue(model.training and model[0].training and model[1].training)
        torch.testing.assert_close(mean.sum(-1), torch.ones(8))
        self.assertTrue(torch.isfinite(std).all() and torch.isfinite(entropy).all())

    def test_mc_dropout_invalid_samples_and_exception_restore(self):
        predict = definitions('PART-03*/轴线B*/01*.md')['mc_dropout_predict']
        class Failing(nn.Module):
            def forward(self, x):
                raise RuntimeError("intentional regression fixture")
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5), Failing())
        model.train()
        model[0].eval()  # mixed state must also survive
        before = [m.training for m in model.modules()]
        for n in (0, 1):
            with self.assertRaises(ValueError):
                predict(model, torch.randn(2, 3), n)
        with self.assertRaises(RuntimeError):
            predict(model, torch.randn(2, 3), 2)
        self.assertEqual([m.training for m in model.modules()], before)

    def test_custom_autograd_and_relu_upstream(self):
        ns = definitions('PART-03*/轴线C*/01*.md')
        a = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(ns['MatMulFunction'].apply, (a, b)))
        x = torch.tensor([-1., 0., 1.], requires_grad=True)
        ns['CustomReLU'].apply(x).backward(torch.tensor([2., 3., 4.]))
        torch.testing.assert_close(x.grad, torch.tensor([0., 3., 4.]))

    def test_manual_backprop_matches_autograd(self):
        ns = definitions('PART-03*/轴线C*/01*.md')
        model = ns['TwoLayerNet'](3, 4, 2)
        for name in ('W1', 'b1', 'W2', 'b2'):
            setattr(model, name, getattr(model, name).double())
        x, y = torch.randn(7, 3, dtype=torch.float64), torch.tensor([0, 1, 1, 0, 1, 0, 1])
        logits = model.forward(x)
        delta = logits.softmax(-1)
        delta[torch.arange(len(y)), y] -= 1
        model.backward(delta)
        params = [getattr(model, n).clone().requires_grad_() for n in ('W1', 'b1', 'W2', 'b2')]
        loss = F.cross_entropy(torch.relu(x @ params[0] + params[1]) @ params[2] + params[3], y)
        expected = torch.autograd.grad(loss, params)
        for name, value in zip(('grad_W1', 'grad_b1', 'grad_W2', 'grad_b2'), expected):
            torch.testing.assert_close(getattr(model, name), value)
        numerical = ns['numerical_gradient'](lambda w: F.cross_entropy(model.forward_with_W1(x, w), y), model.W1)
        torch.testing.assert_close(numerical, model.grad_W1, atol=1e-8, rtol=1e-5)

    def test_newton_step_on_quadratic(self):
        step = definitions('PART-03*/轴线C*/02*.md')['newton_step']
        x = torch.tensor([2., -3.], dtype=torch.float64, requires_grad=True)
        loss = (x[0] - 1)**2 + 2 * (x[1] + 1)**2
        torch.testing.assert_close(x + step(loss, x), torch.tensor([1., -1.], dtype=torch.float64))

    def test_attention_and_transformer(self):
        ns = definitions('PART-03*/轴线D*/02*.md')
        q, k, v = [torch.randn(2, 2, 5, 4) for _ in range(3)]
        out = ns['attention'](q, k, v)
        torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-6, rtol=1e-5)
        model = ns['MultiHeadAttention'](8, 2)
        self.assertEqual(model(torch.randn(2, 5, 8)).shape, (2, 5, 8))
        with self.assertRaises(ValueError):
            ns['MultiHeadAttention'](7, 2)
        model = definitions('PART-03*/轴线D*/03*.md')['TransformerLayer'](8, 2)
        model(torch.randn(2, 5, 8)).square().mean().backward()

    def test_kl_mine_and_distillation(self):
        ns = definitions('PART-03*/轴线E*/02*.md')
        zeros = torch.zeros(8)
        torch.testing.assert_close(ns['mine_loss'](zeros, zeros), torch.tensor(0.))
        torch.testing.assert_close(ns['vae_kl'](zeros[None], zeros[None]), torch.tensor(0.))
        student = torch.randn(4, 3, requires_grad=True)
        teacher = torch.randn(4, 3, requires_grad=True)
        ns['distillation_loss'](student, teacher, torch.tensor([0, 1, 2, 0])).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_spectral_estimator_including_zero_jacobian(self):
        estimate = definitions('PART-04*/05*.md')['spectral_norm_estimate']
        x = torch.randn(3, dtype=torch.float64)
        matrix = torch.diag(torch.tensor([3., 2., .5], dtype=torch.float64))
        torch.testing.assert_close(estimate(lambda v: matrix @ v, x), torch.tensor(3., dtype=torch.float64))
        torch.testing.assert_close(estimate(lambda v: v * 0, x), torch.tensor(0., dtype=torch.float64))

    def test_ddpm_forward_and_one_step_mean(self):
        ns = definitions('PART-05*/01*.md')
        diffusion = ns['Diffusion'](T=1, beta_start=.1, beta_end=.1)
        x0 = torch.randn(3, 2)
        xt, noise = diffusion.q_sample(x0, torch.zeros(3, dtype=torch.long), torch.zeros_like(x0))
        torch.testing.assert_close(xt, math.sqrt(.9) * x0)
        class ConstantNoise(nn.Module):
            def forward(self, x, t):
                return torch.ones_like(x)
        torch.manual_seed(3)
        initial = torch.randn(3, 2)
        torch.manual_seed(3)
        actual = ns['sample'](diffusion, ConstantNoise(), (3, 2))
        expected = (initial - .1 / math.sqrt(.1)) / math.sqrt(.9)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(diffusion.loss(ConstantNoise(), x0)))

    def test_flow_inverse_and_logdet(self):
        Layer = definitions('PART-05*/02*.md')['CouplingLayer']
        for dim in (4, 5):
            layer = Layer(dim).double()
            x = torch.randn(2, dim, dtype=torch.float64)
            z, ld = layer(x)
            torch.testing.assert_close(layer.inverse(z), x)
            jac = torch.autograd.functional.jacobian(lambda a: layer(a[None])[0][0], x[0])
            torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet, ld[0])

    def test_gcn_normalization(self):
        ns = definitions('PART-05*/03*.md')
        a = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
        ah = ns['normalize_adj'](a)
        torch.testing.assert_close(ah, ah.T)
        torch.testing.assert_close(torch.linalg.eigvalsh(ah)[-1], torch.tensor(1., dtype=torch.float64))
        ns['GCN'](2, 4, 3).double()(torch.randn(3, 2, dtype=torch.float64), ah).sum().backward()

    def test_policy_gradient_and_ppo_update_direction(self):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2))
            def generate(self, prompts, return_logp=True):
                y = torch.arange(2)
                return y, self.logp(prompts, y)
            def logp(self, prompts, y):
                return self.logits.log_softmax(0)[y]
        class Reward(nn.Module):
            def __init__(self):
                super().__init__()
                self.rewards = nn.Parameter(torch.tensor([1., -1.]))
            def forward(self, prompts, y):
                return self.rewards[y]
        for pattern, name in [('PART-05*/05*.md', 'ppo_step'), ('PART-03*/轴线E*/03*.md', 'policy_gradient_loss')]:
            policy, ref, rm = Policy(), Policy(), Reward()
            loss = definitions(pattern)[name](policy, ref, rm, None)
            loss.backward()
            self.assertLess(float(policy.logits.grad[0]), 0)  # descent increases preferred probability
            self.assertGreater(float(policy.logits.grad[1]), 0)
            self.assertIsNone(ref.logits.grad)
            self.assertIsNone(rm.rewards.grad)
        ns = definitions('PART-05*/05*.md')
        policy, ref, rm = Policy(), Policy(), Reward()
        rollout = ns['collect_rollout'](policy, ref, rm, None)
        old_logp = rollout[1].clone()
        with torch.no_grad():
            policy.logits.copy_(torch.tensor([1., -1.]))
        loss = ns['ppo_step'](policy, ref, rm, None, rollout=rollout)
        loss.backward()
        # Positive advantage ratio > 1.2 and negative advantage ratio < .8:
        # both are clipped, so further movement in this direction has no gradient.
        torch.testing.assert_close(loss, torch.tensor(-.2))
        torch.testing.assert_close(policy.logits.grad, torch.zeros(2))
        torch.testing.assert_close(rollout[1], old_logp)


if __name__ == '__main__':
    unittest.main(verbosity=2)
