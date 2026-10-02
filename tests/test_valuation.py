import math

import numpy as np
import pytest

from src.valuation import (
    DCFInputs,
    dcf,
    deal_returns,
    discount_rate,
    implied_dilution,
    monthly_cash_projection,
    multiple_value,
    relever_beta,
    resolve_round,
    round_consistency,
    runway_months,
    total_beta,
    unlever_beta,
    vc_method,
    wacc,
)

# ---------------------------------------------------------------- beta


@pytest.mark.parametrize(
    "bu, de, t, expected",
    [
        (0.80, 0.50, 0.25, 1.10),   # 0.8 · (1 + 0.75 · 0.5)
        (1.00, 0.00, 0.21, 1.00),   # sin deuda la beta no cambia
        (0.76, 0.3517, 0.083, 0.76 * (1 + 0.917 * 0.3517)),
        (1.20, 1.00, 0.00, 2.40),
    ],
)
def test_relever_beta(bu, de, t, expected):
    assert relever_beta(bu, de, t) == pytest.approx(expected)
    assert unlever_beta(expected, de, t) == pytest.approx(bu)


def test_total_beta_matches_damodaran_table():
    # Advertising en totalbeta.html: 1.01 / 21.00 % ≈ 4.80
    assert total_beta(1.01, 0.21) == pytest.approx(4.81, abs=0.01)
    with pytest.raises(ValueError):
        total_beta(1.0, 0.0)


def test_discount_rate_capm():
    r = discount_rate(1.0, 0.25, False, 0.04, 0.05, 0.01, 0.02)
    assert r.cost_of_equity == pytest.approx(0.04 + 1.0 * 0.05 + 0.03)
    assert r.cost_of_capital == pytest.approx(r.cost_of_equity)  # sin deuda
    rt = discount_rate(1.0, 0.25, True, 0.04, 0.05, 0.0, 0.0)
    assert rt.beta_used == pytest.approx(4.0)
    assert wacc(0.10, 0.06, 0.25, 1.0) == pytest.approx(0.5 * 0.10 + 0.5 * 0.045)


# ---------------------------------------------------------------- DCF


def _base(**kw):
    p = dict(
        revenue0=100.0, growth_high=0.0, stable_growth=0.0, current_margin=0.2, target_margin=0.2,
        margin_year=1, tax_rate=0.25, sales_to_capital=2.0, cost_of_capital=0.10,
        mature_cost_of_capital=0.10, terminal_roc=0.15, risk_free=0.04,
    )
    p.update(kw)
    return DCFInputs(**p)


def test_dcf_no_growth_is_perpetuity():
    # Sin crecimiento ni reinversión: valor = EBIT · (1 − t) / r = 20 · 0.75 / 0.10 = 150
    r = dcf(_base())
    assert r.operating_value == pytest.approx(150.0)
    assert r.capital_need == 0


def test_dcf_stable_growth_capped_at_risk_free():
    r = dcf(_base(stable_growth=0.08))
    assert r.stable_growth_used == pytest.approx(0.04)
    assert any("tasa libre de riesgo" in w for w in r.warnings)


def test_dcf_nol_shields_taxes():
    # Pierde 10 el año 1 (margen -10 %) y gana 20 desde el año 2: la pérdida compensa impuestos
    inp = _base(current_margin=-0.40, target_margin=0.20, margin_year=2)
    r = dcf(inp)
    proj = r.projection
    assert proj["EBIT"].iloc[0] == pytest.approx(-10.0)
    assert proj["Impuestos"].iloc[0] == 0
    assert proj["Impuestos"].iloc[1] == pytest.approx((20 - 10) * 0.25)
    assert proj["Impuestos"].iloc[2] == pytest.approx(20 * 0.25)


def test_dcf_survival_adjustment():
    r = dcf(_base(survival_prob=0.4, distress_proceeds=0.0, cash=10.0))
    assert r.survival_adjusted_value == pytest.approx(0.4 * 150.0)
    assert r.equity_value == pytest.approx(0.4 * 150.0 + 10.0)


def test_dcf_reinvestment_uses_sales_to_capital():
    r = dcf(_base(growth_high=0.10, stable_growth=0.02))
    proj = r.projection
    assert proj["Reinversión"].iloc[0] == pytest.approx((110 - 100) / 2.0)
    assert proj["Crecimiento"].iloc[-1] == pytest.approx(0.02)


# ---------------------------------------------------------------- método VC


def test_vc_method_hand_example():
    # Salida 100 M en 5 años, IRR 50 %, dilución futura 30 %, inversión 2 M
    # post = 100 · 0.7 / 1.5^5 = 70 / 7.59375 = 9.21811 M
    r = vc_method(100e6, 5, 2e6, 0.50, 0.30)
    assert r.post_money == pytest.approx(9_218_107, rel=1e-6)
    assert r.pre_money == pytest.approx(7_218_107, rel=1e-6)
    assert r.required_stake == pytest.approx(2e6 / 9_218_107, rel=1e-6)
    assert r.required_stake_at_exit == pytest.approx(r.required_stake * 0.7)


def test_vc_method_survival_mode_does_not_double_count():
    irr_mode = vc_method(100e6, 5, 2e6, 0.50, 0.30, survival_prob=0.3, mode="irr")
    assert irr_mode.survival_prob == 1.0  # la IRR alta ya incluye el fracaso
    surv = vc_method(100e6, 5, 2e6, 0.20, 0.30, survival_prob=0.3, mode="survival")
    assert surv.post_money == pytest.approx(100e6 * 0.7 * 0.3 / 1.2**5)


def test_deal_returns():
    terms = resolve_round(2e6, 8e6, None)
    assert terms.stake == pytest.approx(0.2)
    d = deal_returns(100e6, 5, terms, 0.30, 0.5)
    assert d.stake_exit == pytest.approx(0.14)
    assert d.moic == pytest.approx(7.0)
    assert d.irr == pytest.approx(7.0 ** 0.2 - 1)
    assert d.expected_moic == pytest.approx(3.5)


def test_resolve_round_all_combinations():
    a = resolve_round(2e6, 8e6, None)
    b = resolve_round(2e6, None, 0.2)
    c = resolve_round(None, 8e6, 0.2)
    for t in (a, b, c):
        assert (t.investment, t.pre_money, t.stake) == pytest.approx((2e6, 8e6, 0.2))
    assert round_consistency(2e6, 8e6, 0.2) is None
    assert round_consistency(2e6, 8e6, 0.25) is not None
    with pytest.raises(ValueError):
        resolve_round(2e6, None, None)


# ---------------------------------------------------------------- múltiplos y caja


def test_multiple_value():
    assert multiple_value(10.0, 3.0, 0.25) == pytest.approx(22.5)
    assert math.isnan(multiple_value(-5.0, 10.0, 0.2))  # EBITDA negativo: no aplica


def test_runway_and_cash_projection():
    assert runway_months(1_200_000, 100_000) == 12
    assert runway_months(1_000, 0) == math.inf
    df = monthly_cash_projection(1_200_000, 0.0, 100_000, 0.0, 0.0, months=12)
    assert df["Caja"].iloc[-1] == pytest.approx(0.0)
    # con crecimiento de ingresos el consumo de caja baja mes a mes
    df2 = monthly_cash_projection(1e6, 1.2e6, 50_000, 1.0, 0.0, months=24)
    assert np.all(np.diff(df2["Consumo de caja"]) < 0)


def test_implied_dilution():
    assert implied_dilution(0, 10e6) == 0
    assert implied_dilution(5e6, 15e6) == pytest.approx(0.25)
