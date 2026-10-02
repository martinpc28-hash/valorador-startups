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
        f = sum(v / (1 + rate) ** t for v, t in zip(values, ts))
        df = sum(-t * v / (1 + rate) ** (t + 1) for v, t in zip(values, ts))
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
