"""Registro de fuentes de métricas por industria.

Cada fuente es un módulo de este paquete que registra una función sin argumentos que
devuelve un DataFrame con el esquema único (`SCHEMA`). Para añadir una fuente:

1. Crear `src/sources/<fuente>.py` con una función decorada con `@register("<fuente>")`.
2. Añadir sus filas a `data/industry_metrics.csv` (o leerlas de otro CSV local).
3. Añadir la fuente a `data/sources.csv` y sus industrias a `data/industry_crosswalk.csv`.

`valuation.py` no cambia: solo recibe números ya resueltos.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

import pandas as pd

SCHEMA = [
    "source", "region", "industry_std", "industry_original", "metric",
    "value", "unit", "as_of", "retrieved_at", "url", "license",
]
PROVENANCE = ["source", "url", "as_of", "retrieved_at", "license"]

Loader = Callable[[], pd.DataFrame]
REGISTRY: dict[str, Loader] = {}


def register(source_id: str) -> Callable[[Loader], Loader]:
    def deco(fn: Loader) -> Loader:
        REGISTRY[source_id] = fn
        return fn
    return deco


def unregister(source_id: str) -> None:
    REGISTRY.pop(source_id, None)


def discover() -> None:
    """Importa todos los módulos del paquete para que se registren."""
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


def validate(df: pd.DataFrame, source_id: str) -> pd.DataFrame:
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"La fuente {source_id!r} no cumple el esquema: faltan {missing}")
    df = df[SCHEMA].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    no_prov = df[PROVENANCE].isna().any(axis=1) | (df[PROVENANCE].astype(str) == "").any(axis=1)
    if no_prov.any():
        raise ValueError(f"La fuente {source_id!r} tiene {int(no_prov.sum())} filas sin procedencia")
    return df


def load_all() -> pd.DataFrame:
    discover()
    frames = [validate(loader(), sid) for sid, loader in REGISTRY.items()]
    if not frames:
        return pd.DataFrame(columns=SCHEMA)
    return pd.concat(frames, ignore_index=True)
