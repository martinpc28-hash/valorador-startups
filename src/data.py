"""Carga de CSV locales, taxonomía de industrias y resolución de prioridades entre fuentes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from src import sources
from src.cleaning import clip
from src.paths import DATA

REFERENCE_INDUSTRY = "Total Market (without financials)"


# ---------------------------------------------------------------- carga

def load_industry_metrics() -> pd.DataFrame:
    return sources.load_all()


def load_crosswalk() -> pd.DataFrame:
    return pd.read_csv(DATA / "industry_crosswalk.csv")


def load_metric_definitions() -> pd.DataFrame:
    return pd.read_csv(DATA / "metric_definitions.csv").set_index("metric")


def load_sources() -> pd.DataFrame:
    return pd.read_csv(DATA / "sources.csv", dtype=str).fillna("")


def load_stage_assumptions() -> pd.DataFrame:
    return pd.read_csv(DATA / "stage_assumptions.csv").sort_values("order").reset_index(drop=True)


def load_size_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA / "size_class_metrics.csv")


def load_market_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA / "market_metrics.csv")


def load_fx() -> pd.DataFrame:
    return pd.read_csv(DATA / "fx_rates.csv")


def industry_table(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Industrias seleccionables (sin filas de referencia), con su sector."""
    xw = crosswalk[~crosswalk["is_reference"].astype(bool)]
    return (
        xw[["industry_std", "sector"]]
        .drop_duplicates("industry_std")
        .sort_values(["sector", "industry_std"])
        .reset_index(drop=True)
    )


def latest_series(market: pd.DataFrame, series: str) -> pd.Series | None:
    rows = market[market["series"] == series].dropna(subset=["value"])
    if rows.empty:
        return None
    return rows.sort_values("date").iloc[-1]


def latest_fx(fx: pd.DataFrame, base: str = "EUR", quote: str = "USD") -> pd.Series | None:
    rows = fx[(fx["base"] == base) & (fx["quote"] == quote)].dropna(subset=["rate"])
    if rows.empty:
        return None
    return rows.sort_values("date").iloc[-1]


# ---------------------------------------------------------------- resolución

@dataclass
class Resolved:
    """Un número de industria con su procedencia y las banderas de calidad."""

    metric: str
    label: str
    industry: str
    value: float
    raw_value: float = math.nan
    unit: str = ""
    source: str = ""
    url: str = ""
    as_of: str = ""
    retrieved_at: str = ""
    license: str = ""
    industry_used: str = ""
    fallback: bool = False          # se usó una fuente distinta de la preferida
    preferred_source: str = ""
    reference_fallback: bool = False  # la industria no tenía el dato; se usó el total de mercado
    capped: bool = False
    missing: bool = False
    alternatives: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing and not math.isnan(self.value)

    def warnings(self) -> list[str]:
        out = []
        if self.missing:
            out.append(f"Sin dato de «{self.label}» para {self.industry} en ninguna fuente.")
        if self.fallback:
            out.append(
                f"«{self.label}»: la fuente preferida ({self.preferred_source}) no tiene el dato; "
                f"se usa {self.source}."
            )
        if self.reference_fallback:
            out.append(
                f"«{self.label}»: {self.industry} no tiene dato; se usa {self.industry_used} como referencia."
            )
        if self.capped:
            out.append(
                f"«{self.label}»: valor extremo {self.raw_value:.4g} recortado a {self.value:.4g}."
            )
        return out

    def provenance(self) -> str:
        if self.missing:
            return "Sin dato"
        parts = [
            f"Fuente: {self.source}",
            f"Industria: {self.industry_used}",
            f"Fecha de los datos: {self.as_of}",
            f"Extraído: {self.retrieved_at}",
            f"URL: {self.url}",
        ]
        if self.capped:
            parts.append(f"Valor original {self.raw_value:.4g} (recortado)")
        if len(self.alternatives) > 1:
            alt = ", ".join(f"{k}: {v:.4g}" for k, v in self.alternatives.items())
            parts.append(f"Otras fuentes: {alt}")
        return "  \n".join(parts)


class MetricResolver:
    """Elige el valor de cada métrica según un orden de prioridad de fuentes.

    No promedia: toma la primera fuente del orden que tenga el dato, guarda los valores
    de las demás en `alternatives` y marca si hubo reemplazo o recorte.
    """

    def __init__(
        self,
        metrics: pd.DataFrame,
        definitions: pd.DataFrame,
        priority: list[str] | None = None,
        apply_caps: bool = True,
        reference_industry: str | None = REFERENCE_INDUSTRY,
    ):
        self.metrics = metrics
        self.definitions = definitions
        available = list(dict.fromkeys(metrics["source"]))
        self.priority = [s for s in (priority or available) if s in available] + [
            s for s in available if s not in (priority or [])
        ]
        self.apply_caps = apply_caps
        self.reference_industry = reference_industry
        self._index = {
            key: grp for key, grp in metrics.dropna(subset=["value"]).groupby(["industry_std", "metric"])
        }
        self.used: dict[tuple[str, str], Resolved] = {}

    def _lookup(self, industry: str, metric: str) -> pd.DataFrame | None:
        return self._index.get((industry, metric))

    def get(self, industry: str, metric: str) -> Resolved:
        label = str(self.definitions.loc[metric, "label_es"]) if metric in self.definitions.index else metric
        rows = self._lookup(industry, metric)
        industry_used = industry
        reference_fallback = False
        if (rows is None or rows.empty) and self.reference_industry:
            rows = self._lookup(self.reference_industry, metric)
            industry_used = self.reference_industry
            reference_fallback = rows is not None and not rows.empty
        if rows is None or rows.empty:
            res = Resolved(metric, label, industry, math.nan, missing=True, industry_used=industry)
            self.used[(industry, metric)] = res
            return res

        by_source = {r.source: r for r in rows.itertuples()}
        chosen = next(s for s in self.priority if s in by_source)
        row = by_source[chosen]
        raw = float(row.value)
        value, capped = raw, False
        if self.apply_caps and metric in self.definitions.index:
            d = self.definitions.loc[metric]
            value, capped = clip(raw, d["cap_min"], d["cap_max"])
        res = Resolved(
            metric=metric, label=label, industry=industry, value=value, raw_value=raw,
            unit=row.unit, source=chosen, url=row.url, as_of=str(row.as_of),
            retrieved_at=str(row.retrieved_at), license=row.license, industry_used=industry_used,
            fallback=chosen != self.priority[0], preferred_source=self.priority[0],
            reference_fallback=reference_fallback, capped=capped,
            alternatives={s: float(by_source[s].value) for s in self.priority if s in by_source},
        )
        self.used[(industry, metric)] = res
        return res

    def comparison(self, industry: str, metrics: list[str]) -> pd.DataFrame:
        """Tabla métrica × fuente para mostrar valores distintos sin promediarlos."""
        df = self.metrics[(self.metrics["industry_std"] == industry) & (self.metrics["metric"].isin(metrics))]
        return df.pivot_table(index="metric", columns="source", values="value", aggfunc="first")
