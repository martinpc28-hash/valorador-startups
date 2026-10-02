"""Limpieza de datos crudos: porcentajes, NA, nombres de industria y conversión de tasas."""

from __future__ import annotations

import math
import re

NA_TOKENS = {"", "na", "n/a", "nan", "#n/a", "-", "--", "#div/0!"}


def parse_number(raw: object) -> float:
    """Convierte un texto de tabla a float.

    - "40.20%" -> 0.402 (los porcentajes pasan a decimales)
    - "$1,885.26" -> 1885.26
    - "-$8.43" -> -8.43
    - "NA", "" -> NaN
    """
    if raw is None:
        return math.nan
    if isinstance(raw, (int, float)):
        return float(raw)
    text = " ".join(str(raw).split())
    if text.lower() in NA_TOKENS:
        return math.nan
    is_pct = text.endswith("%")
    cleaned = re.sub(r"[%$,\s]", "", text)
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return math.nan
    return value / 100.0 if is_pct else value


def normalize_name(raw: str) -> str:
    """Quita saltos de línea y espacios dobles de un nombre, sin cambiar su ortografía."""
    return " ".join(str(raw).split())


def annual_to_monthly(rate: float) -> float:
    """Tasa anual compuesta -> tasa mensual equivalente: (1 + g)^(1/12) - 1."""
    return (1.0 + rate) ** (1.0 / 12.0) - 1.0


def monthly_to_annual(rate: float) -> float:
    return (1.0 + rate) ** 12.0 - 1.0


def clip(value: float, lo: float | None, hi: float | None) -> tuple[float, bool]:
    """Recorta un valor a [lo, hi]. Devuelve (valor, fue_recortado). NaN pasa sin cambios."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return value, False
    out = value
    if lo is not None and not math.isnan(lo) and out < lo:
        out = lo
    if hi is not None and not math.isnan(hi) and out > hi:
        out = hi
    return out, out != value
