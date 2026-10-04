"""Fuente 2: ratios sectoriales de empresas españolas del Banco de España (Central de Balances).

Mediana del último ejercicio, todos los tamaños. El detalle por tamaño y cuartil está en
`data/bde_ratios.csv`. Los datos se generan con `scripts/build_bde.py`.
"""

from __future__ import annotations

import pandas as pd

from src.paths import DATA
from src.sources import SCHEMA, register

SOURCE_ID = "bde"


@register(SOURCE_ID)
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / "industry_metrics.csv")
    df = df[df["source"] == SOURCE_ID]
    return df if not df.empty else pd.DataFrame(columns=SCHEMA)


def size_for_revenue(rev_eur: float) -> str:
    """Tramo de tamaño del Banco de España según la cifra de negocios en euros."""
    if rev_eur < 2e6:
        return "3"   # Menos de 2 millones
    if rev_eur < 10e6:
        return "4"   # De 2 a 10 millones
    if rev_eur < 50e6:
        return "1b"  # De 10 a 50 millones
    return "2"       # Más de 50 millones


def load_detail() -> pd.DataFrame:
    path = DATA / "bde_ratios.csv"
    return pd.read_csv(path, dtype={"size_id": str, "sector_code": str}) if path.exists() else pd.DataFrame()
