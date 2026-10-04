"""Fuentes españolas: ratios del Banco de España y sociedades del BORME."""

import math
import re

import pandas as pd
import pytest

from src import comparables as cmp
from src.data import MetricResolver, load_crosswalk, load_industry_metrics, load_metric_definitions
from src.paths import DATA

# En el BORME los nombres de personas van siempre tras un cargo y dos puntos ("Adm. Unico: APELLIDOS NOMBRE").
# Palabras sueltas como "administrador" pueden ser parte del objeto social ("administrador de fincas").
PERSONAL = re.compile(
    r"(?:Adm\.\s*(?:Unico|Solid\.?|Mancom\.?)|Apoderado|Socio [uú]nico|Consejero|Liquidador|Auditor|Presidente|Secretario|"
    r"Vicepresidente|Cons\.\s*Del\.?|Con\.Delegado|LiquiSoli)\s*:\s*[A-ZÁÉÍÓÚÑ]{2,}"
)


@pytest.fixture(scope="module")
def bde():
    return pd.read_csv(DATA / "bde_ratios.csv")


@pytest.fixture(scope="module")
def borme():
    return cmp.load("borme")


def test_bde_detail_shape_and_units(bde):
    assert set(bde["metric"].unique()) == {
        "revenue_growth_1y", "sales_per_employee", "personnel_cost_per_employee", "ebitda_margin", "cost_of_debt",
        "roi_ordinary", "roe", "debt_to_liabilities", "receivable_days", "payable_days"}
    assert bde["size_id"].nunique() == 7
    assert bde["year"].between(2015, 2030).all()
    pct = bde[bde["unit"] == "decimal"]
    # porcentajes guardados como decimales: la mediana del EBITDA/ventas ronda 0,0x a 0,3
    med = pct[(pct["metric"] == "ebitda_margin") & (pct["sector_code"] == "ZC") & (pct["size_id"] == "0")]["p50"]
    assert med.between(-0.2, 0.5).all()
    ordered = bde.dropna(subset=["p25", "p50", "p75"]).copy()
    assert ((ordered["p25"] <= ordered["p50"] + 1e-9) & (ordered["p50"] <= ordered["p75"] + 1e-9)).all()


def test_bde_sectors_follow_crosswalk(bde):
    xw = load_crosswalk()
    pairs = set(map(tuple, xw[xw["source"] == "bde"][["industry_original", "industry_std"]].values))
    assert set(map(tuple, bde[["sector_code", "industry_std"]].drop_duplicates().values)) == pairs


def test_software_maps_to_j62_and_information_services_to_j63():
    xw = load_crosswalk()
    by_std = xw[xw["source"] == "bde"].set_index("industry_std")["industry_original"].to_dict()
    assert by_std["Software (System & Application)"] == "J62"
    assert by_std["Software (Internet)"] == "J62"
    assert by_std["Information Services"] == "J63"


def test_bde_suppressed_zeros_are_empty(bde):
    # el servicio devuelve 0 en los tres valores cuando no publica el dato: no deben quedar así
    assert not ((bde["p25"] == 0) & (bde["p50"] == 0) & (bde["p75"] == 0)).any()
    j63_big = bde[(bde["sector_code"] == "J63") & (bde["size_id"].astype(str) == "2")]
    assert j63_big["p50"].isna().all()


def test_bde_is_a_resolver_source_without_changing_defaults():
    df = load_industry_metrics()
    assert "bde" in set(df["source"])
    assert set(df[df["source"] == "bde"]["region"]) == {"ES"}
    defs = load_metric_definitions()
    default = MetricResolver(df, defs)
    assert default.get("Software (System & Application)", "ebitda_margin").source == "damodaran"
    es_first = MetricResolver(df, defs, priority=["bde", "damodaran"])
    r = es_first.get("Software (System & Application)", "ebitda_margin")
    assert r.source == "bde" and set(r.alternatives) == {"bde", "damodaran"}
    # métricas que el Banco de España no tiene siguen saliendo de Damodaran, con aviso de reemplazo
    beta = es_first.get("Software (System & Application)", "unlevered_beta_cash_adj")
    assert beta.source == "damodaran" and beta.fallback


def test_borme_has_no_personal_data(borme):
    assert not {"administrators", "partners", "officers"} & set(borme.columns)
    assert not borme["purpose"].dropna().str.contains(PERSONAL).any()
    assert not borme["name"].str.contains(PERSONAL).any()


def test_borme_only_companies_with_relevant_acts(borme):
    assert borme["name"].is_unique
    form = re.compile(r"\b(SOCIEDAD|S\.?L\.?U?|S\.?A\.?U?|S\.?L\.?L|SLNE|COOPERATIVA|S\.?COOP|AGRUPACION DE INTERES)\b", re.I)
    assert borme["name"].str.contains(form).all()
    assert ((borme["constitution_date"].notna()) | (borme["n_capital_increases"] > 0)).all()
    assert (borme["capital_latest"].dropna() >= 0).all()


def test_borme_industry_from_cnae(borme):
    with_ind = borme.dropna(subset=["industry_std"])
    assert not with_ind.empty
    xw = load_crosswalk()
    mapped = (xw[xw["source"] == "bde"].drop_duplicates("industry_original", keep="first")
              .set_index("industry_original")["industry_std"].to_dict())
    assert (with_ind["cnae_division"].map(mapped) == with_ind["industry_std"]).all()


def test_borme_search_matches_purpose(borme):
    hits = cmp.search(borme, "software")
    assert not hits.empty
    assert (hits["name"].str.contains("software", case=False) | hits["purpose"].str.contains("software", case=False, na=False)).all()


def test_borme_prefill_only_industry(borme):
    row = borme.dropna(subset=["industry_std"]).iloc[0]
    p = cmp.prefill_from(row, "borme")
    assert set(p.values) == {"industry"}  # el capital nominal no es ni ingresos ni ronda


def test_size_for_revenue_buckets(bde):
    from src.sources.bde import size_for_revenue

    assert [size_for_revenue(x) for x in (1e6, 5e6, 20e6, 80e6)] == ["3", "4", "1b", "2"]
    assert {size_for_revenue(x) for x in (1e6, 5e6, 20e6, 80e6)} <= set(bde["size_id"].astype(str))
