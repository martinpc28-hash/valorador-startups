"""Integridad de los datos: procedencia, equivalencias de industrias y conteos."""

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from src import sources
from src.data import MetricResolver, load_crosswalk, load_industry_metrics, load_metric_definitions
from src.paths import DATA, ROOT
from src.sources import SCHEMA
from src.valuation import discount_rate

PROVENANCE = ["source", "url", "as_of", "retrieved_at", "license"]


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(DATA / "industry_metrics.csv")


def test_schema(metrics):
    assert list(metrics.columns) == SCHEMA


SEC_FILES = ["sec_form_d.csv.gz", "sec_form_d_funds.csv.gz", "sec_form_c.csv.gz", "sec_s1.csv.gz",
             "bde_ratios.csv", "borme_companies.csv.gz"]


@pytest.mark.parametrize("name", ["industry_metrics.csv", "size_class_metrics.csv", "market_metrics.csv", "fx_rates.csv"] + SEC_FILES)
def test_every_row_has_provenance(name):
    df = pd.read_csv(DATA / name)
    assert not df.empty
    for col in PROVENANCE:
        assert col in df.columns, f"{name} sin columna {col}"
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        assert not blank.any(), f"{name}: {int(blank.sum())} filas sin {col}"


def test_sources_registry_lists_every_source(metrics):
    registry = pd.read_csv(DATA / "sources.csv")
    for f in ["industry_metrics.csv", "size_class_metrics.csv", "market_metrics.csv", "fx_rates.csv"] + SEC_FILES:
        used = set(pd.read_csv(DATA / f)["source"])
        assert used <= set(registry["source_id"]), f"{f}: fuentes sin registrar {used - set(registry['source_id'])}"


def test_metrics_are_defined(metrics):
    defs = load_metric_definitions()
    assert set(metrics["metric"]) <= set(defs.index)


def test_crosswalk_covers_every_industry(metrics):
    xw = load_crosswalk()
    for source, grp in metrics.groupby("source"):
        mapped = set(xw[xw["source"] == source]["industry_original"])
        missing = set(grp["industry_original"]) - mapped
        assert not missing, f"{source}: industrias sin equivalencia {missing}"
        # y el industry_std guardado coincide con la tabla de equivalencias
        m = xw[xw["source"] == source].set_index("industry_original")["industry_std"]
        assert (grp["industry_original"].map(m) == grp["industry_std"]).all()


def test_industry_count_matches_source_pages(metrics):
    manifest = pd.read_csv(DATA / "damodaran_manifest.csv")
    xw = load_crosswalk()
    refs = set(xw[xw["is_reference"].astype(bool)]["industry_std"])
    page_of = metrics["url"].str.extract(r"/([^/]+)\.html$")[0]
    for row in manifest[manifest["kind"] == "industry"].itertuples():
        sub = metrics[(metrics["source"] == "damodaran") & (page_of == row.page)]
        industries = set(sub["industry_std"]) - refs
        assert len(industries) == row.n_industries == 94, row.page


def test_no_excel_files_in_repo():
    bad = [p for p in ROOT.rglob("*") if p.suffix.lower() in {".xls", ".xlsx", ".xlsm"} and ".venv" not in p.parts]
    assert not bad, bad


def test_typo_preserved_in_original_but_clean_in_std(metrics):
    row = metrics[metrics["industry_original"] == "Heathcare Information and Technology"]
    assert not row.empty
    assert set(row["industry_std"]) == {"Healthcare Information and Technology"}


def test_percentages_stored_as_decimals(metrics):
    adv = metrics[(metrics["industry_std"] == "Advertising") & (metrics["metric"] == "de_ratio")]
    assert adv["value"].iloc[0] == pytest.approx(0.402)


# ---------------------------------------------------------------- fuentes y resolución


@pytest.fixture
def fake_source():
    """Fuente de prueba registrada en tiempo de ejecución, sin tocar valuation.py."""

    def load():
        return pd.DataFrame([{
            "source": "fuente_prueba", "region": "EU", "industry_std": "Software (Internet)",
            "industry_original": "Internet Software", "metric": "unlevered_beta_cash_adj",
            "value": 1.50, "unit": "beta", "as_of": "2026-06-30", "retrieved_at": "2026-10-02",
            "url": "https://example.org/test", "license": "test",
        }])

    sources.register("fuente_prueba")(load)
    yield
    sources.unregister("fuente_prueba")


def test_new_source_without_touching_valuation(fake_source):
    df = load_industry_metrics()
    assert "fuente_prueba" in set(df["source"])
    defs = load_metric_definitions()

    pref_test = MetricResolver(df, defs, priority=["fuente_prueba", "damodaran"])
    beta = pref_test.get("Software (Internet)", "unlevered_beta_cash_adj")
    assert beta.source == "fuente_prueba" and beta.value == 1.50
    assert set(beta.alternatives) == {"fuente_prueba", "damodaran"}  # se muestran ambas, sin promediar

    # Métrica que la fuente de prueba no tiene: reemplazo por la siguiente, con aviso
    corr = pref_test.get("Software (Internet)", "correlation_market")
    assert corr.source == "damodaran" and corr.fallback
    assert corr.warnings()

    r = discount_rate(beta.value, corr.value, False, 0.04, 0.05, 0.0, 0.0)
    assert r.cost_of_equity == pytest.approx(0.04 + 1.5 * 0.05)

    # valuation.py no depende de las fuentes ni de la capa de datos
    tree = ast.parse((ROOT / "src" / "valuation.py").read_text(encoding="utf-8"))
    imported = {
        (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    } | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("src.sources", "src.data")) for m in imported)


def test_resolver_caps_extremes_and_warns():
    df = load_industry_metrics()
    res = MetricResolver(df, load_metric_definitions())
    g = res.get("Air Transport", "exp_revenue_growth_2y")
    assert g.raw_value == pytest.approx(2.0503)
    assert g.capped and g.value == pytest.approx(0.6)
    rr = res.get("Software (Internet)", "reinvestment_rate")
    assert rr.capped and rr.raw_value > 14
    off = MetricResolver(df, load_metric_definitions(), apply_caps=False)
    assert not off.get("Air Transport", "exp_revenue_growth_2y").capped


def test_resolver_missing_value_falls_back_to_reference():
    df = load_industry_metrics()
    res = MetricResolver(df, load_metric_definitions())
    m = res.get("Bank (Money Center)", "ev_ebitda_pos")  # "NA" en la página de Damodaran
    assert m.reference_fallback and m.industry_used == "Total Market (without financials)"
    assert not math.isnan(m.value)
    strict = MetricResolver(df, load_metric_definitions(), reference_industry=None)
    assert strict.get("Bank (Money Center)", "ev_ebitda_pos").missing
