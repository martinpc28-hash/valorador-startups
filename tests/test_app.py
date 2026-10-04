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
        # La barra lateral primero: "Industria" y "Etapa" también existen en el formulario de Mis empresas
        widget = next(w for w in [*at.sidebar.selectbox, *at.sidebar.radio, *at.selectbox, *at.radio] if w.label == label)
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


def test_save_company_and_load_into_model(tmp_path, monkeypatch):
    monkeypatch.setenv("VALORADOR_STORE", "local")
    monkeypatch.setenv("VALORADOR_LOCAL_DIR", str(tmp_path))
    import importlib

    import src.company_store as store_mod
    importlib.reload(store_mod)

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert any("compartida" in w.value for w in at.warning)  # aviso de biblioteca sin contraseña
    next(t for t in at.text_input if t.label == "Nombre").input("Mi Startup")
    next(n for n in at.number_input if n.label == "Ingresos últimos 12 meses").set_value(4_000_000.0)
    at.run()
    next(b for b in at.button if b.label == "Guardar y cargar en el modelo").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [c["name"] for c in store_mod.LocalStore(tmp_path).list("compartida")] == ["Mi Startup"]
    assert next(n for n in at.number_input if n.label.startswith("Ingresos últimos 12 meses (")).value == 4_000_000.0
    assert any(s.label == "Cargar empresa guardada" for s in at.selectbox)
    # Sin inversión ni pre-money guardadas, cargar la empresa no rompe la ronda
    next(b for b in at.button if b.label == "Cargar en el modelo").click()
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]


def test_active_company_survives_new_session_via_url(tmp_path, monkeypatch):
    """Una sesión nueva (reconexión o recarga) con ?empresa=id vuelve a cargar la empresa sola."""
    monkeypatch.setenv("VALORADOR_STORE", "local")
    monkeypatch.setenv("VALORADOR_LOCAL_DIR", str(tmp_path))
    import src.company_store as store_mod

    c = store_mod.new_company("Fija SL")
    c.update({"id": "fija-sl", "industry": "Drugs (Biotechnology)", "currency": "EUR",
              "inputs": {"revenue": 2_500_000.0, "growth": 0.4, "current_margin": -0.2}})
    store_mod.LocalStore(tmp_path).save("compartida", c)

    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["empresa"] = "fija-sl"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["active_company"] == "fija-sl"
    assert at.session_state["rev0"] == 2_500_000.0
    assert at.session_state["industry"] == "Drugs (Biotechnology)"
    assert at.session_state["currency"] == "EUR"
    # y sigue fija tras otra ejecución (cualquier interacción)
    at.run()
    assert at.session_state["rev0"] == 2_500_000.0


def test_every_metric_tooltip_shows_its_formula():
    """Cada cifra calculada (tarjeta st.metric) explica en su ? con qué fórmula se obtuvo."""
    at = run()
    missing = [m.label for m in at.metric if "Fórmula:" not in (m.proto.help or "")]
    assert not missing, f"Tarjetas sin fórmula en el ?: {missing}"
    assert len(at.metric) >= 40


def test_vc_survival_mode_and_ebitda_exit():
    run(**{"Tratamiento del riesgo de fracaso": "Costo del equity × supervivencia", "Múltiplo de salida": "EV/EBITDA", "Beta": "De mercado"})
