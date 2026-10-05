"""Diagnóstico de resultados extremos: detección, explicación, alternativas y botón Aplicar."""

import math
import re
from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from src import diagnostics as diag
from src import memo as memo_mod
from src.paths import ROOT


def _state(**kw) -> diag.DiagState:
    base = dict(
        currency="USD", industry="Software (Internet)", revenue=200e6, growth=0.5, current_margin=-0.058, target_margin=0.037,
        industry_margin=0.037, margin_year=7, sales_to_capital=1.54, use_total_beta=True, beta_unlevered=1.25, correlation=0.27,
        beta_used=4.65, risk_free=0.0418, erp=0.0423, extra_premia=0.02, cost_of_capital=0.27, stable_growth=0.03,
        dcf_equity=0.0, dcf_operating=-600e6, dcf_raw_equity=-493e6, pv_terminal=100e6, capital_need=2_000e6, investment=200e6,
        pre_money=2_500e6, exit_year=6, exit_basis="EV/Sales", exit_multiple=11.4, small_cap_multiple=1.23, exit_revenue=2_000e6,
        exit_value=22_800e6, vc_pre_money=3_180e6, vc_rate=0.3, future_dilution=0.2, stake=0.074, moic=6.0,
        method_mids={"DCF": 1e6, "Método VC": 3_180e6, "Múltiplos": 1_500e6},
    )
    base.update(kw)
    return diag.DiagState(**base)


def fake_eval(changes: dict) -> dict:
    # Un DCF que mejora con margen, ventas/capital y beta de mercado (para comprobar el cableado, no el modelo)
    dcf = -493e6 + 2_000e6 * (changes.get("target_margin", 0.037) - 0.037) + 100e6 * (changes.get("sales_to_capital", 1.54) - 1.54)
    if changes.get("use_total_beta") is False:
        dcf += 300e6
    mult = changes.get("exit_multiple", 11.4)
    return {"dcf": max(dcf, 0.0), "vc_pre": 3_180e6 * mult / 11.4, "moic": 6.0 * mult / 11.4}


def test_negative_dcf_is_explained_first_with_its_drivers():
    notes = diag.diagnose(_state(), fake_eval)
    assert notes[0].id == "dcf_negative" and notes[0].severity == "alta"
    text = " ".join(notes[0].drivers)
    assert "Margen objetivo del 3,7 %" in text and "ventas / capital de 1,54" in text and "beta total" in text
    labels = [a.label for a in notes[0].alternatives]
    assert any("Beta de mercado" in l for l in labels) and any("Ventas / capital" in l for l in labels)
    assert all(a.outcome for a in notes[0].alternatives)  # cada alternativa recalculada
    assert [n.severity for n in notes] == sorted((n.severity for n in notes), key=diag.SEVERITY_ORDER.get)


def test_all_rules_fire_for_an_extreme_case_and_none_for_a_calm_one():
    ids = {n.id for n in diag.diagnose(_state(moic=25.0, dcf_operating=500e6, pv_terminal=480e6), fake_eval)}
    assert {"dcf_negative", "methods_disagree", "terminal_heavy", "high_discount", "moic_extreme", "exit_multiple_high",
            "capital_hungry"} <= ids
    assert "revenue_explosion" in {n.id for n in diag.diagnose(_state(exit_revenue=200e6 * 60), fake_eval)}
    calm = _state(dcf_raw_equity=100e6, dcf_operating=100e6, pv_terminal=50e6, cost_of_capital=0.15, moic=4.0,
                  exit_multiple=2.0, capital_need=100e6, exit_revenue=1_000e6,
                  method_mids={"DCF": 100e6, "Método VC": 150e6})
    assert diag.diagnose(calm, fake_eval) == []


def test_note_texts_have_no_long_dashes():
    notes = diag.diagnose(_state(moic=25.0, dcf_operating=500e6, pv_terminal=480e6), fake_eval)
    blob = " ".join([n.title + n.explanation + " ".join(n.drivers) + " ".join(a.label + a.why for a in n.alternatives) for n in notes])
    assert not re.search("[—–]", blob)


def test_alternative_that_fails_does_not_break_the_diagnosis():
    def broken(_):
        raise ZeroDivisionError

    notes = diag.diagnose(_state(), broken)
    assert notes and all(a.outcome == {} for n in notes for a in n.alternatives)


def test_memo_includes_result_notes():
    notes = diag.diagnose(_state(), fake_eval)
    from tests.test_memo import _ctx  # contexto de ejemplo del memo

    memo = memo_mod.build_memo(_ctx(result_notes=[n.as_text("USD") for n in notes]))
    titles = [s.title for s in memo.sections]
    assert any("Notas sobre los resultados" in t for t in titles)
    assert titles.index(next(t for t in titles if "Notas" in t)) < titles.index(next(t for t in titles if "Riesgos" in t))
    assert "El DCF es negativo" in memo_mod.memo_text(memo)


def test_app_shows_notes_and_apply_changes_the_sidebar():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    keys = {b.key for b in at.button if b.label == "Aplicar"}
    # El ejemplo por defecto (tasa del 27 % con beta total, plan que consume mucha caja) tiene notas que explicar
    assert "diag_res_high_discount_0" in keys and "diag_res_capital_hungry_0" in keys
    assert at.session_state["beta_type"] == "Total"
    s2c_key = next(k for k in at.session_state if str(k).startswith("s2c_"))
    s2c_before = at.session_state[s2c_key]

    at.button(key="diag_res_high_discount_0").click()  # beta de mercado
    at.run()
    assert at.session_state["beta_type"] == "De mercado"
    assert "diag_res_high_discount_0" not in {b.key for b in at.button}  # con beta de mercado ya no es tan alta

    at.button(key="diag_res_capital_hungry_0").click()  # ventas / capital el doble
    at.run()
    assert at.session_state[s2c_key] == pytest.approx(round(s2c_before * 2, 2))
    assert not at.exception
