"""Empresas individuales de la SEC (Form D, Form C, S-1): búsqueda, comparación y precarga.

Los datos se generan con `scripts/build_sec.py` y se leen de CSV locales. Importes en USD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.paths import DATA

DATASETS = {
    "sec_form_d": "Rondas privadas (Form D)",
    "sec_form_c": "Startups con financieros (Form C)",
    "sec_s1": "Salidas a bolsa (S-1)",
}

# Punto medio de los rangos de ingresos de Form D (para estimar ingresos)
REVENUE_RANGE_MID = {
    "No Revenues": 0.0,
    "$1 - $1,000,000": 500_000.0,
    "$1,000,001 - $5,000,000": 3_000_000.0,
    "$5,000,001 - $25,000,000": 15_000_000.0,
    "$25,000,001 - $100,000,000": 62_500_000.0,
}


def load(dataset: str) -> pd.DataFrame:
    name = {"sec_form_d": "sec_form_d.csv.gz", "sec_form_c": "sec_form_c.csv.gz",
            "sec_s1": "sec_s1.csv.gz", "sec_form_d_funds": "sec_form_d_funds.csv.gz"}[dataset]
    df = pd.read_csv(DATA / name, dtype={"sic": str, "accession": str})
    return add_derived(df, dataset)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).where((den > 0) & num.notna())


def add_derived(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.copy()
    if dataset == "sec_form_c":
        df["growth"] = (_ratio(df["revenue"], df["revenue_prior"]) - 1).where(df["revenue_prior"] > 0)
        df["net_margin"] = _ratio(df["net_income"], df["revenue"])
        df["gross_margin"] = _ratio(df["revenue"] - df["cogs"], df["revenue"])
        df["round_size"] = df["offering_max"].fillna(df["offering_target"])
    elif dataset == "sec_s1":
        df["growth"] = (_ratio(df["revenue"], df["revenue_prior"]) - 1).where(df["revenue_prior"] > 0)
        df["operating_margin"] = _ratio(df["operating_income"], df["revenue"])
        df["net_margin"] = _ratio(df["net_income"], df["revenue"])
    elif dataset == "sec_form_d":
        df["revenue_est"] = df["revenue_range"].map(REVENUE_RANGE_MID)
        df["round_size"] = df["total_offering"].fillna(df["total_sold"])
    return df


def search(
    df: pd.DataFrame,
    text: str = "",
    industries: list[str] | None = None,
    young_only: bool = False,
    with_revenue: bool = False,
    limit: int = 500,
) -> pd.DataFrame:
    out = df
    if text.strip():
        out = out[out["name"].str.contains(text.strip(), case=False, na=False, regex=False)]
    if industries and "industry_std" in out.columns:
        out = out[out["industry_std"].isin(industries)]
    if young_only and "inc_within_5y" in out.columns:
        out = out[out["inc_within_5y"].astype(bool)]
    if with_revenue and "revenue" in out.columns:
        out = out[out["revenue"] > 0]
    return out.sort_values("filing_date", ascending=False).head(limit).reset_index(drop=True)


def percentile_of(value: float, sample: pd.Series) -> float:
    """Porcentaje de la muestra que queda por debajo de `value` (NaN si no hay muestra)."""
    s = pd.to_numeric(sample, errors="coerce").dropna()
    if s.empty or value is None or math.isnan(value):
        return math.nan
    return float((s < value).mean())


# ---------------------------------------------------------------- precarga


@dataclass
class Prefill:
    """Valores a cargar en las entradas del modelo (en USD y decimales) y su origen."""

    values: dict[str, float | str]
    notes: dict[str, str]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def prefill_from(row: pd.Series, dataset: str) -> Prefill:
    v: dict[str, float | str] = {}
    notes: dict[str, str] = {}

    def ok(x) -> bool:
        return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)

    if dataset in ("sec_form_c", "sec_s1"):
        if ok(row.get("revenue")) and row["revenue"] > 0:
            v["revenue"] = float(row["revenue"])
            notes["revenue"] = f"Ingresos del último ejercicio ({row.get('fiscal_year', '')})".replace(" ()", "")
        if ok(row.get("growth")):
            v["growth"] = _clip(row["growth"], -0.5, 5.0)
            notes["growth"] = "Crecimiento de ingresos del último ejercicio frente al anterior"
        margin_col = "operating_margin" if dataset == "sec_s1" else "net_margin"
        if ok(row.get(margin_col)):
            v["current_margin"] = _clip(row[margin_col], -10.0, 0.9)
            notes["current_margin"] = ("Margen operativo" if dataset == "sec_s1"
                                       else "Margen neto (Form C no informa el operativo)")
    if dataset == "sec_form_c":
        if ok(row.get("cash")):
            v["cash"] = float(row["cash"])
            notes["cash"] = "Caja al cierre del último ejercicio"
        if ok(row.get("round_size")) and row["round_size"] > 0:
            v["investment"] = float(row["round_size"])
            notes["investment"] = "Importe máximo de la oferta de crowdfunding"
    if dataset == "sec_form_d":
        if ok(row.get("round_size")) and row["round_size"] > 0:
            v["investment"] = float(row["round_size"])
            notes["investment"] = ("Importe total de la oferta" if ok(row.get("total_offering"))
                                   else "Importe vendido (la oferta era indefinida)")
        if ok(row.get("revenue_est")) and row["revenue_est"] > 0:
            v["revenue"] = float(row["revenue_est"])
            notes["revenue"] = f"Punto medio del rango declarado ({row['revenue_range']})"
    if ok(row.get("industry_std")):
        v["industry"] = str(row["industry_std"])
        notes["industry"] = f"Equivalencia desde la clasificación de la SEC ({row.get('industry_group') or 'SIC ' + str(row.get('sic', ''))})"
    return Prefill(v, notes)


# ---------------------------------------------------------------- comparación


COMPARE_ROWS = [
    ("Industria", "industry_std", "text"),
    ("Ingresos", "revenue", "money"),
    ("Crecimiento de ingresos", "growth", "pct"),
    ("Margen operativo", "operating_margin", "pct"),
    ("Margen neto", "net_margin", "pct"),
    ("Caja", "cash", "money"),
    ("Ronda / oferta", "round_size", "money"),
    ("Vendido en la oferta", "total_sold", "money"),
    ("Inversores", "n_investors", "int"),
    ("Empleados", "employees", "int"),
    ("Fecha de presentación", "filing_date", "text"),
]


def comparison_table(user: dict, companies: pd.DataFrame, fmt_money, fmt_pct) -> pd.DataFrame:
    """Filas = métricas; columnas = tu startup + cada empresa seleccionada (valores formateados)."""

    def fmt(x, kind):
        if x is None or (isinstance(x, float) and math.isnan(x)) or (not isinstance(x, str) and pd.isna(x)):
            return "n/d"
        if kind == "money":
            return fmt_money(x)
        if kind == "pct":
            return fmt_pct(x)
        if kind == "int":
            return f"{int(x):,}".replace(",", ".")
        return str(x)

    data = {"Métrica": [label for label, _, _ in COMPARE_ROWS]}
    data["Tu startup"] = [fmt(user.get(col), kind) for _, col, kind in COMPARE_ROWS]
    for _, row in companies.iterrows():
        rev = row.get("revenue", np.nan)
        if (pd.isna(rev)) and not pd.isna(row.get("revenue_est", np.nan)):
            rev = row["revenue_est"]
        vals = []
        for _, col, kind in COMPARE_ROWS:
            x = rev if col == "revenue" else row.get(col, np.nan)
            vals.append(fmt(x, kind))
        name = str(row["name"])[:40]
        while name in data:
            name += " "
        data[name] = vals
    return pd.DataFrame(data)
