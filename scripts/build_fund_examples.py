"""Construye data/fund_examples.csv: fondos de VC reales para precargar en la pestaña Fondos.

Fuente: CalPERS, Private Equity Program Fund Performance Review (datos a 31/03/2026),
https://www.calpers.ca.gov/investments/about-investment-office/investment-organization/pep-fund-performance

CalPERS publica solo totales (capital desembolsado, distribuido, valor residual e IRR neta), no las
fechas de cada flujo. Los totales se respetan exactamente; el reparto año a año se reconstruye con
un calendario típico y se ajusta un parámetro de tiempo para que la XIRR coincida con la IRR publicada.

Uso: python scripts/build_fund_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fund import fund_metrics  # noqa: E402
from src.paths import ROOT  # noqa: E402

AS_OF = "2026-03-31"

# Importes en millones de USD, tal como los publica CalPERS (columna Cash In, Cash Out, Cash Out & Remaining Value)
FUNDS = [
    dict(fund="Insight Venture Partners IX (2015)", vintage=2015, cash_in=105.668637, cash_out=287.819011,
         total=419.282453, irr=0.226),
    dict(fund="Lightspeed Venture Partners Select V (2022)", vintage=2022, cash_in=96.5, cash_out=0.0,
         total=179.153201, irr=0.277),
    dict(fund="Insight Partners XII (2021)", vintage=2021, cash_in=574.829570, cash_out=0.663047,
         total=629.086271, irr=0.026),
]


def _dates(vintage: int) -> list[str]:
    return [f"{y}-12-31" for y in range(vintage, 2026)] + [AS_OF]


def _build(f: dict, k: float) -> pd.DataFrame:
    """k mueve en el tiempo las llamadas (fondos sin distribuciones) o el pico de distribuciones."""
    dates = _dates(f["vintage"])
    n = len(dates)
    t = np.arange(n, dtype=float)
    has_dist = f["cash_out"] > 0.05 * f["cash_in"]
    if has_dist:
        calls_w = np.array([0.15, 0.25, 0.25, 0.2, 0.1, 0.05] + [0.0] * n)[:n]
        dist_w = np.where(t >= 4, np.exp(-((t - k) ** 2) / 6), 0.0)
    else:
        # Fondo joven: llamadas en los primeros años con un centro ajustable; distribuciones testimoniales al final
        calls_w = np.exp(-((t - k) ** 2) / 2) * (t < n - 1)
        dist_w = (t == n - 1).astype(float)
    calls = calls_w / calls_w.sum() * f["cash_in"]
    dist = dist_w / dist_w.sum() * f["cash_out"] if f["cash_out"] else np.zeros(n)
    nav = [None] * (n - 1) + [f["total"] - f["cash_out"]]
    return pd.DataFrame({"date": pd.to_datetime(dates), "capital_call": calls, "distribution": dist, "nav": nav})


def _calibrate(f: dict) -> pd.DataFrame:
    lo, hi = (4.0, 30.0) if f["cash_out"] > 0.05 * f["cash_in"] else (-3.0, len(_dates(f["vintage"])) + 3.0)
    gap = lambda k: fund_metrics(_build(f, k)).irr - f["irr"]  # noqa: E731
    for _ in range(80):  # bisección
        mid = (lo + hi) / 2
        if gap(lo) * gap(mid) <= 0:
            hi = mid
        else:
            lo = mid
    df = _build(f, (lo + hi) / 2)
    df[["capital_call", "distribution"]] = df[["capital_call", "distribution"]].round(3)
    df["nav"] = df["nav"].astype(float).round(3)
    # El redondeo no debe mover los totales publicados
    df.loc[df["capital_call"].idxmax(), "capital_call"] += round(f["cash_in"] - df["capital_call"].sum(), 3)
    if f["cash_out"]:
        df.loc[df["distribution"].idxmax(), "distribution"] += round(f["cash_out"] - df["distribution"].sum(), 3)
    return df.round({"capital_call": 3, "distribution": 3, "nav": 3})


def main() -> None:
    out = []
    for f in FUNDS:
        df = _calibrate(f)
        m = fund_metrics(df)
        print(f"{f['fund']}: TVPI {m.tvpi:.2f}x, DPI {m.dpi:.2f}x, IRR {m.irr:.1%} (publicada {f['irr']:.1%})")
        df.insert(0, "fund", f["fund"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        out.append(df)
    pd.concat(out).to_csv(ROOT / "data" / "fund_examples.csv", index=False)


if __name__ == "__main__":
    main()
