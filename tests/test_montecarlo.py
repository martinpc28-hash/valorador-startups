import numpy as np
import pytest

from src.montecarlo import MCSettings, simulate
from src.valuation import DCFInputs


def _base():
    return DCFInputs(
        revenue0=2e6, growth_high=0.6, stable_growth=0.03, current_margin=-0.5, target_margin=0.2,
        margin_year=7, tax_rate=0.25, sales_to_capital=1.5, cost_of_capital=0.2,
        mature_cost_of_capital=0.09, terminal_roc=0.15, risk_free=0.04, survival_prob=0.4,
    )


def test_reproducible_with_fixed_seed():
    s = MCSettings(n_sims=2000)
    a = simulate(_base(), 5.0, 5, 2e6, 0.2, 0.3, s)
    b = simulate(_base(), 5.0, 5, 2e6, 0.2, 0.3, s)
    assert np.array_equal(a.moic, b.moic)
    assert a.percentiles == b.percentiles


def test_default_runs_10000_and_correlation_sign():
    r = simulate(_base(), 5.0, 5, 2e6, 0.2, 0.3, MCSettings())
    assert r.moic.shape == (10_000,)
    assert np.corrcoef(r.growth, np.log(r.multiple))[0, 1] > 0.3
    assert 0.3 < r.survived.mean() < 0.5
    p = r.percentiles["MOIC"]
    assert p["P10"] <= p["P50"] <= p["P90"]
    assert 0 <= r.prob_moic_target <= 1


def test_without_failure_all_survive():
    r = simulate(_base(), 5.0, 5, 2e6, 0.2, 0.3, MCSettings(n_sims=1000, include_failure=False))
    assert r.survived.all()
    assert (r.moic > 0).all()
