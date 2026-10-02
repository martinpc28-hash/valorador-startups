"""Fuente 1: tablas por industria de Aswath Damodaran (EE. UU., enero de 2026).

Los datos se generan con `scripts/build_damodaran.py` y se leen del CSV local.
"""

from __future__ import annotations

import pandas as pd

from src.paths import DATA
from src.sources import register

SOURCE_ID = "damodaran"


@register(SOURCE_ID)
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / "industry_metrics.csv")
    return df[df["source"] == SOURCE_ID]
