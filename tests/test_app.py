"""Pruebas de humo de la app: arranca sin internet y cambiar las entradas recalcula sin errores."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.paths import DATA, ROOT

APP = str(ROOT / "app.py")


def run(**changes) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    for label, value in changes.items():
        widget = next(w for w in [*at.selectbox, *at.radio] if w.label == label)
        widget.set_value(value)
    if changes:
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric_value(at: AppTest, label_prefix: str) -> str:
    return next(m.value for m in at.metric if m.label.startswith(label_prefix))


def test_default_run():
    at = run()
    assert at.title[0].value == "Valorador de Startups"
    assert metric_value(at, "DCF (equity)")


def test_changing_industry_stage_currency_updates_results():
    base = run()
    other = run(**{"Industria": "Drugs (Biotechnology)", "Etapa": "Semilla", "Moneda base": "EUR"})
    assert metric_value(base, "DCF (equity)") != metric_value(other, "DCF (equity)")
    assert any("Semilla" in c.value for c in other.caption)


INDUSTRIES = pd.read_csv(DATA / "industry_crosswalk.csv").query("source == 'damodaran' and is_reference == False")["industry_std"].tolist()


@pytest.mark.slow
@pytest.mark.parametrize("industry", INDUSTRIES)
def test_every_industry_runs(industry):
    run(**{"Industria": industry})


@pytest.mark.parametrize("stage", ["Semilla", "Serie A", "Serie B", "Crecimiento", "Madura"])
def test_every_stage_runs(stage):
    run(**{"Etapa": stage})


def test_prefill_values_flow_into_the_model():
    """Simula el botón de precarga: los valores en session_state reemplazan las entradas."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    before = metric_value(at, "DCF (equity)")
    at.session_state["rev0"] = 5_000_000.0
    at.session_state["industry"] = "Drugs (Biotechnology)"
    at.run()
    assert not at.exception
    assert any("Drugs (Biotechnology)" in c.value for c in at.caption)
    assert metric_value(at, "DCF (equity)") != before


def test_comparables_sources_and_fund_tab_render():
    for ds in ("sec_form_c", "sec_s1"):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        next(r for r in at.radio if r.label == "Fuente").set_value(ds)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("TVPI") for m in at.metric)


def test_vc_survival_mode_and_ebitda_exit():
    run(**{"Tratamiento del riesgo de fracaso": "Costo del equity × supervivencia", "Múltiplo de salida": "EV/EBITDA", "Beta": "De mercado"})
