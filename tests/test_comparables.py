"""Datos de la SEC: privacidad, equivalencias de industria, búsqueda y precarga."""

import math

import pandas as pd
import pytest

from src import comparables as cmp
from src.data import load_crosswalk
from src.paths import DATA

PERSONAL_COLUMNS = {"street1", "street2", "zipcode", "phone", "issuerphonenumber", "firstname", "lastname",
                    "signaturename", "nameofsigner", "personsignature", "issuersignature"}


@pytest.mark.parametrize("dataset", ["sec_form_d", "sec_form_c", "sec_s1", "sec_form_d_funds"])
def test_no_personal_data(dataset):
    df = cmp.load(dataset)
    assert not df.empty
    assert not PERSONAL_COLUMNS & {c.lower() for c in df.columns}


def test_one_row_per_company():
    for ds in ("sec_form_d", "sec_form_c", "sec_s1"):
        df = cmp.load(ds)
        assert df["cik"].is_unique, ds


def test_form_d_excludes_pooled_funds_and_maps_industries():
    d = cmp.load("sec_form_d")
    assert not (d["industry_group"] == "Pooled Investment Fund").any()
    xw = load_crosswalk()
    mapped = set(xw[xw["source"] == "sec_form_d"]["industry_original"])
    assert set(d["industry_group"].dropna()) <= mapped
    assert d["industry_std"].notna().all()


def test_s1_sic_codes_mapped_and_no_spacs():
    s = cmp.load("sec_s1")
    xw = load_crosswalk()
    mapped = set(xw[xw["source"] == "sec_sic"]["industry_original"].astype(str))
    assert set(s["sic"].dropna()) <= mapped
    assert not (s["sic"] == "6770").any()
    assert (s["revenue"] > 0).all() or s["revenue"].notna().all()


def test_crosswalk_targets_exist_in_taxonomy():
    xw = load_crosswalk()
    std = set(xw[xw["source"] == "damodaran"]["industry_std"])
    assert set(xw["industry_std"]) <= std


def test_search_filters():
    d = cmp.load("sec_form_d")
    soft = cmp.search(d, industries=["Software (System & Application)"], young_only=True)
    assert len(soft) <= 500
    assert set(soft["industry_std"]) <= {"Software (System & Application)"}
    assert soft["inc_within_5y"].all()
    assert soft["filing_date"].is_monotonic_decreasing
    name = d["name"].iloc[0]
    assert (cmp.search(d, text=name[:6].lower())["name"].str.lower().str.contains(name[:6].lower(), regex=False)).all()


def test_percentile_of():
    s = pd.Series([1, 2, 3, 4])
    assert cmp.percentile_of(3.5, s) == 0.75
    assert math.isnan(cmp.percentile_of(1.0, pd.Series([], dtype=float)))


def test_prefill_form_c():
    row = pd.Series({"name": "X", "revenue": 200_000.0, "revenue_prior": 100_000.0, "net_income": -100_000.0,
                     "cash": 50_000.0, "offering_max": 1_000_000.0, "offering_target": 10_000.0, "cogs": 0.0})
    df = cmp.add_derived(row.to_frame().T.astype({k: float for k in row.index if k != "name"}), "sec_form_c")
    p = cmp.prefill_from(df.iloc[0], "sec_form_c")
    assert p.values["revenue"] == 200_000
    assert p.values["growth"] == pytest.approx(1.0)
    assert p.values["current_margin"] == pytest.approx(-0.5)
    assert p.values["cash"] == 50_000
    assert p.values["investment"] == 1_000_000
    assert "industry" not in p.values  # Form C no informa la industria
    assert set(p.notes) == set(p.values)


def test_prefill_form_d_uses_range_midpoint():
    d = cmp.load("sec_form_d")
    row = d[(d["revenue_range"] == "$1,000,001 - $5,000,000") & d["round_size"].notna()].iloc[0]
    p = cmp.prefill_from(row, "sec_form_d")
    assert p.values["revenue"] == 3_000_000
    assert p.values["industry"] == row["industry_std"]
    assert p.values["investment"] == row["round_size"]


def test_comparison_table_has_user_and_companies():
    s = cmp.load("sec_s1").head(2)
    t = cmp.comparison_table({"revenue": 1e6, "industry_std": "Advertising"}, s, lambda x: f"{x:.0f}", lambda x: f"{x:.0%}")
    assert list(t.columns[:2]) == ["Métrica", "Tu startup"]
    assert t.shape[1] == 4
