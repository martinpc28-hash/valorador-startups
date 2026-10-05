"""Métricas de un fondo de VC: DPI, RVPI, TVPI, MOIC, IRR (XIRR propio) y curva J."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd


def _years(d0: dt.date, d: dt.date) -> float:
    return (d - d0).days / 365.0


def xnpv(rate: float, cashflows: list[tuple[dt.date, float]]) -> float:
    d0 = min(d for d, _ in cashflows)
    return sum(cf / (1.0 + rate) ** _years(d0, d) for d, cf in cashflows)


def xirr(cashflows: list[tuple[dt.date, float]], guess: float = 0.1, tol: float = 1e-10) -> float:
    """IRR con fechas irregulares (convención de Excel: días / 365).

    Newton-Raphson con derivada analítica y, si no converge, bisección en [-0.9999, 100].
    """
    values = [cf for _, cf in cashflows]
    if not (any(v > 0 for v in values) and any(v < 0 for v in values)):
        raise ValueError("XIRR necesita al menos un flujo positivo y uno negativo")
    d0 = min(d for d, _ in cashflows)
    ts = [_years(d0, d) for d, _ in cashflows]

    rate = guess
    for _ in range(100):
        if rate <= -1:
            break
        try:
            f = sum(v / (1 + rate) ** t for v, t in zip(values, ts))
            df = sum(-t * v / (1 + rate) ** (t + 1) for v, t in zip(values, ts))
        except (OverflowError, ZeroDivisionError):  # Newton se fue cerca de -100 %: pasamos a bisección
            break
        if df == 0:
            break
        new = rate - f / df
        if abs(new - rate) < tol:
            return new
        rate = new

    lo, hi = -0.9999, 100.0
    f_lo, f_hi = xnpv(lo, cashflows), xnpv(hi, cashflows)
    if f_lo * f_hi > 0:
        raise ValueError("XIRR no converge: no hay cambio de signo en el rango buscado")
    for _ in range(500):
        mid = (lo + hi) / 2
        f_mid = xnpv(mid, cashflows)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


@dataclass
class FundMetrics:
    paid_in: float
    distributed: float
    nav: float
    dpi: float
    rvpi: float
    tvpi: float
    moic: float
    irr: float


def fund_metrics(flows: pd.DataFrame) -> FundMetrics:
    """`flows` con columnas: date, capital_call (>= 0), distribution (>= 0), nav (valor residual).

    La IRR usa las llamadas como salidas, las distribuciones como entradas y el último NAV
    como valor terminal en su fecha.
    """
    f = flows.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.date
    f = f.sort_values("date")
    paid_in = float(f["capital_call"].fillna(0).sum())
    distributed = float(f["distribution"].fillna(0).sum())
    nav_rows = f.dropna(subset=["nav"])
    nav = float(nav_rows["nav"].iloc[-1]) if not nav_rows.empty else 0.0
    nav_date = nav_rows["date"].iloc[-1] if not nav_rows.empty else f["date"].iloc[-1]

    cfs = [(d, float(dist or 0) - float(call or 0)) for d, call, dist in zip(f["date"], f["capital_call"], f["distribution"])]
    cfs.append((nav_date, nav))
    cfs = [(d, v) for d, v in cfs if v != 0]
    try:
        irr = xirr(cfs)
    except ValueError:
        irr = math.nan

    dpi = distributed / paid_in if paid_in else math.nan
    rvpi = nav / paid_in if paid_in else math.nan
    return FundMetrics(paid_in, distributed, nav, dpi, rvpi, dpi + rvpi, dpi + rvpi, irr)


def j_curve(flows: pd.DataFrame) -> pd.DataFrame:
    """Flujo neto acumulado del LP (sin NAV) y valor total (con NAV) por fecha."""
    f = flows.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date")
    net = f["distribution"].fillna(0) - f["capital_call"].fillna(0)
    f["Flujo neto acumulado"] = net.cumsum()
    f["Valor total (con NAV)"] = f["Flujo neto acumulado"] + f["nav"].ffill().fillna(0)
    return f[["date", "Flujo neto acumulado", "Valor total (con NAV)"]]


def project_cash_flows(
    commitment: float,
    years: int = 12,
    call_rates: tuple[float, ...] = (0.25, 0.25, 0.2, 0.15, 0.1, 0.05),
    growth: float = 0.12,
    bow: float = 2.5,
) -> pd.DataFrame:
    """Proyección simple tipo Takahashi-Alexander: llamadas por calendario y distribuciones crecientes."""
    rows, nav, uncalled = [], 0.0, commitment
    for y in range(1, years + 1):
        call = commitment * call_rates[y - 1] if y - 1 < len(call_rates) else 0.0
        call = min(call, uncalled)
        uncalled -= call
        rd = (y / years) ** bow
        dist = nav * (1 + growth) * rd
        nav = nav * (1 + growth) + call - dist
        rows.append({"Año": y, "Llamadas": call, "Distribuciones": dist, "NAV": nav})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- fondos precargados y guardados

FLOW_COLS = ["date", "capital_call", "distribution", "nav"]
EXAMPLES_SOURCE = ("CalPERS, Private Equity Program Fund Performance Review (31/03/2026): "
                   "https://www.calpers.ca.gov/investments/about-investment-office/investment-organization/pep-fund-performance")


def clean_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas estándar, fechas como fecha e importes numéricos (vacío = sin dato)."""
    missing = [c for c in FLOW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {', '.join(missing)}")
    out = df[FLOW_COLS].copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in FLOW_COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def flows_to_records(df: pd.DataFrame) -> list[dict]:
    """Para guardar en la biblioteca (JSON / Firestore): fechas ISO y NaN como None."""
    d = clean_flows(df)
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in d.to_dict("records")]


def records_to_flows(records: list[dict]) -> pd.DataFrame:
    return clean_flows(pd.DataFrame(records, columns=FLOW_COLS))


def load_examples(path=None) -> dict[str, pd.DataFrame]:
    """Fondos reales de data/fund_examples.csv (ver scripts/build_fund_examples.py), en millones de USD."""
    from src.paths import ROOT

    df = pd.read_csv(path or ROOT / "data" / "fund_examples.csv")
    return {name: clean_flows(g) for name, g in df.groupby("fund", sort=False)}


# Tamaño total de cada fondo precargado (no solo la parte de CalPERS), en millones de USD
EXAMPLE_META = {
    "Insight Venture Partners IX (2015)": dict(
        commitment=100.0, fund_size=3_290.0,
        size_source="Insight Venture Partners, cierre del fondo IX (11/08/2015): USD 3.290 M"),
    "Lightspeed Venture Partners Select V (2022)": dict(
        commitment=100.0, fund_size=2_260.0,
        size_source="Lightspeed, cierre de Select V (12/07/2022): USD 2.260 M"),
    "Insight Partners XII (2021)": dict(
        commitment=600.0, fund_size=20_000.0,
        size_source="Insight Partners, cierre del fondo XII (24/02/2022): más de USD 20.000 M junto con su fondo de coinversión"),
}


def projection_flows(commitment: float, years: int, growth: float, bow: float, start_year: int = 2026) -> pd.DataFrame:
    """La proyección en el formato de flujos (una fecha por año) para calcular TVPI, DPI e IRR."""
    p = project_cash_flows(commitment, years, growth=growth, bow=bow)
    return pd.DataFrame({
        "date": pd.to_datetime([f"{start_year + y}-12-31" for y in p["Año"]]),
        "capital_call": p["Llamadas"], "distribution": p["Distribuciones"], "nav": p["NAV"],
    })


def calibrate_growth(target_irr: float, commitment: float = 100.0, years: int = 12, bow: float = 2.5) -> float:
    """Crecimiento anual del NAV con el que la proyección da la IRR objetivo (bisección en [-20 %, 50 %])."""
    lo, hi = -0.2, 0.5
    irr = lambda g: fund_metrics(projection_flows(commitment, years, g, bow)).irr  # noqa: E731
    if not (irr(lo) <= target_irr <= irr(hi)):
        return min(max(target_irr, lo), hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if irr(mid) < target_irr else (lo, mid)
    return (lo + hi) / 2


def calibrate_projection(fm: FundMetrics, commitment: float, bow: float = 2.5) -> tuple[int, float]:
    """Vida y crecimiento del NAV con los que la proyección reproduce el fondo real.

    El crecimiento se ajusta a la IRR. La vida solo se ajusta al TVPI en fondos maduros (DPI >= 1): en uno
    joven el TVPI es provisional y se deja la vida típica de 12 años.
    """
    if not math.isfinite(fm.irr):
        return 12, 0.12
    if fm.dpi < 1:
        return 12, calibrate_growth(fm.irr, commitment, 12, bow)
    best = min(range(6, 16), key=lambda y: abs(
        fund_metrics(projection_flows(commitment, y, calibrate_growth(fm.irr, commitment, y, bow), bow)).tvpi - fm.tvpi))
    return best, calibrate_growth(fm.irr, commitment, best, bow)
