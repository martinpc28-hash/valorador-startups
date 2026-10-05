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


# ----------------------------------------------------------------------- precargados y guardados

def test_preloaded_funds_match_published_calpers_totals():
    from src.fund import load_examples

    ex = load_examples()
    published = {  # millones de USD: (desembolsado, distribuido, valor residual, IRR neta)
        "Insight Venture Partners IX (2015)": (105.669, 287.819, 131.463, 0.226),
        "Lightspeed Venture Partners Select V (2022)": (96.5, 0.0, 179.153, 0.277),
        "Insight Partners XII (2021)": (574.830, 0.663, 628.423, 0.026),
    }
    assert set(ex) == set(published)
    for name, (paid, dist, nav, irr) in published.items():
        m = fund_metrics(ex[name])
        assert m.paid_in == pytest.approx(paid, abs=0.002) and m.distributed == pytest.approx(dist, abs=0.002)
        assert m.nav == pytest.approx(nav, abs=0.002) and m.irr == pytest.approx(irr, abs=0.0006)


def test_flows_round_trip_through_records_keeps_values_and_missing_nav():
    from src.fund import clean_flows, flows_to_records, records_to_flows

    flows = pd.DataFrame({"date": ["2021-12-31", "2020-12-31"], "capital_call": [5.0, 10.0],
                          "distribution": [0.0, None], "nav": [12.0, None], "extra": [1, 2]})
    recs = flows_to_records(flows)
    assert recs[0] == {"date": "2020-12-31", "capital_call": 10.0, "distribution": None, "nav": None}  # ordenado, sin NaN
    back = records_to_flows(recs)
    pd.testing.assert_frame_equal(back, clean_flows(flows))
    with pytest.raises(ValueError, match="nav"):
        clean_flows(flows.drop(columns="nav"))


def test_projection_calibrated_to_a_real_fund_reproduces_its_irr_and_tvpi():
    from src.fund import EXAMPLE_META, calibrate_projection, load_examples, projection_flows

    ex = load_examples()
    assert set(EXAMPLE_META) == set(ex)
    mature = fund_metrics(ex["Insight Venture Partners IX (2015)"])
    years, g = calibrate_projection(mature, 100.0)
    proj = fund_metrics(projection_flows(100.0, years, g, 2.5))
    assert proj.irr == pytest.approx(mature.irr, abs=0.001) and proj.tvpi == pytest.approx(mature.tvpi, abs=0.1)
    young = fund_metrics(ex["Lightspeed Venture Partners Select V (2022)"])
    assert calibrate_projection(young, 100.0)[0] == 12  # TVPI provisional: no se fuerza la vida


def test_xirr_survives_newton_overflow_near_minus_100_percent():
    from src.fund import projection_flows

    m = fund_metrics(projection_flows(100.0, 12, -0.2, 2.5))  # fondo que pierde casi todo
    assert -1 < m.irr < 0
