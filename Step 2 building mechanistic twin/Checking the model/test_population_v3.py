"""
Tests for the v3 change: fibroblast contact inhibition over a 120 um disc.
v1 and v2 tests are unchanged.
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.set_default_dtype(torch.float64)

from tme_twin.tissue_v2 import TissueV2, _fib_density             # noqa: E402
from tme_twin.tissue_v3 import TissueV3                            # noqa: E402
from tme_twin import priors_v3, priors_v4                          # noqa: E402

CHEAP = {"nx": 100, "ny": 100, "o2_every_h": 10 ** 6, "dt_field": 3600.0}


def test_only_the_fibroblast_disc_changes():
    T2 = TissueV2(2, 0.02, seed=3, overrides={"nx": 40, "ny": 40})
    T3 = TissueV3(2, 0.02, seed=3, overrides={"nx": 40, "ny": 40})
    assert T2.n_nb == 28 and T3.n_nb == 112
    assert torch.equal(T2.kernel, T3.kernel)                       # tumour, M2, CD4 rules keep 60 um
    assert T3.fib_kernel.shape[-1] == 13


def test_low_set_point_is_resolvable():
    # v2's 28-neighbour rule could not go below ~1/28; 112 neighbours can.
    K = torch.tensor([0.005, 0.01, 0.02, 0.05])
    target = torch.full((4,), 0.2)
    r28, r112 = _fib_density(K, target, 28), _fib_density(K, target, 112)
    assert float(r28[0]) == pytest.approx(float(r28[2]))           # v2: flat below 1/28
    assert torch.all(r112[1:] > r112[:-1])                         # v3: follows K_fib
    assert float(r112[0]) < float(r28[0]) / 3


def test_priors_v4_changes_only_K_fib():
    p3, p4 = priors_v3.PRIORS, priors_v4.PRIORS
    assert list(p3) == list(p4)
    assert p4["K_fib"][:2] == (0.005, 0.2)
    assert all(p3[k] == p4[k] for k in p3 if k != "K_fib")
    assert priors_v4.regime_priors("residual")["K_fib"] == p4["K_fib"]
    assert priors_v4.regime_priors("residual")["f0_tumour"] == priors_v3.regime_priors("residual")["f0_tumour"]
    assert priors_v4.by_class("patient") == priors_v3.by_class("patient")


@pytest.mark.parametrize("K", [0.01, 0.03, 0.10])
def test_fibroblasts_hold_at_steady_state(K):
    T = TissueV3(8, 0.0, seed=21, overrides={**CHEAP, "K_fib": K})
    pred = T.analytic["n_fib"].mean().item()
    obs = []
    for day in range(40):
        T.run(24)
        if day >= 20:
            c = T.counts()
            obs.append(float((c["fibroblast"] + c["CAF"]).double().mean()))
    ratio = np.mean(obs) / pred
    assert abs(ratio - 1) < 0.08, (K, pred, np.mean(obs), ratio)
