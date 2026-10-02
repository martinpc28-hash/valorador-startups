import datetime as dt

import pandas as pd
import pytest

from src.fund import fund_metrics, j_curve, project_cash_flows, xirr, xnpv


def test_xirr_excel_reference():
    # Ejemplo de la documentación de XIRR de Microsoft Excel: resultado 0.373362535
    cfs = [
        (dt.date(2008, 1, 1), -10000),
        (dt.date(2008, 3, 1), 2750),
        (dt.date(2008, 10, 30), 4250),
        (dt.date(2009, 2, 15), 3250),
        (dt.date(2009, 4, 1), 2750),
    ]
    r = xirr(cfs)
    assert r == pytest.approx(0.373362535, abs=1e-6)
    assert xnpv(r, cfs) == pytest.approx(0.0, abs=1e-6)


def test_xirr_simple_one_year():
    cfs = [(dt.date(2021, 1, 1), -100), (dt.date(2022, 1, 1), 110)]
    assert xirr(cfs) == pytest.approx(0.10, abs=1e-6)


def test_xirr_needs_sign_change():
    with pytest.raises(ValueError):
        xirr([(dt.date(2021, 1, 1), 100), (dt.date(2022, 1, 1), 110)])


def test_fund_metrics():
    flows = pd.DataFrame({
        "date": ["2020-01-01", "2021-01-01", "2023-01-01", "2024-01-01"],
        "capital_call": [50, 50, 0, 0],
        "distribution": [0, 0, 40, 30],
        "nav": [None, None, None, 80],
    })
    m = fund_metrics(flows)
    assert m.paid_in == 100
    assert m.dpi == pytest.approx(0.70)
    assert m.rvpi == pytest.approx(0.80)
    assert m.tvpi == pytest.approx(1.50)
    assert m.irr > 0
    jc = j_curve(flows)
    assert jc["Flujo neto acumulado"].min() == -100


def test_projection_calls_sum_to_commitment():
    p = project_cash_flows(100.0)
    assert p["Llamadas"].sum() == pytest.approx(100.0)
