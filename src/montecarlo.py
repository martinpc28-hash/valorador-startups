"""Simulación de Monte Carlo sobre crecimiento, margen y múltiplo de salida.

Las tres variables se generan correlacionadas (cópula gaussiana con Cholesky). La semilla
es fija para que el resultado sea reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.valuation import DCFInputs, apply_survival, dcf_vectorized, equity_floor


@dataclass
class MCSettings:
    n_sims: int = 10_000
    seed: int = 42
    growth_sd: float = 0.10          # desviación del crecimiento anual (puntos)
    margin_sd: float = 0.05          # desviación del margen objetivo (puntos)
    multiple_sd: float = 0.35        # desviación del log del múltiplo
    corr_growth_margin: float = -0.2  # crecer más suele costar margen
    corr_growth_multiple: float = 0.5  # el mercado paga más por crecer más
    corr_margin_multiple: float = 0.2
    include_failure: bool = True
    moic_target: float = 3.0


@dataclass
class MCResult:
    growth: np.ndarray
    margin: np.ndarray
    multiple: np.ndarray
    survived: np.ndarray
    dcf_equity: np.ndarray
    exit_value: np.ndarray
    moic: np.ndarray
    percentiles: dict[str, dict[str, float]]
    prob_moic_target: float
    prob_loss: float


def correlated_normals(n: int, corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Si la matriz no es definida positiva, se ajusta al vecino más cercano por autovalores.
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(corr)
        fixed = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
        d = np.sqrt(np.diag(fixed))
        chol = np.linalg.cholesky(fixed / np.outer(d, d))
    return rng.standard_normal((n, corr.shape[0])) @ chol.T


def simulate(
    base: DCFInputs,
    exit_multiple: float,
    exit_year: int,
    investment: float,
    stake_entry: float,
    future_dilution: float,
    settings: MCSettings,
) -> MCResult:
    rng = np.random.default_rng(settings.seed)
    corr = np.array([
        [1.0, settings.corr_growth_margin, settings.corr_growth_multiple],
        [settings.corr_growth_margin, 1.0, settings.corr_margin_multiple],
        [settings.corr_growth_multiple, settings.corr_margin_multiple, 1.0],
    ])
    z = correlated_normals(settings.n_sims, corr, rng)

    growth = np.clip(base.growth_high + settings.growth_sd * z[:, 0], -0.5, 3.0)
    margin = np.clip(base.target_margin + settings.margin_sd * z[:, 1], -0.5, 0.7)
    s = settings.multiple_sd
    multiple = exit_multiple * np.exp(s * z[:, 2] - 0.5 * s * s)  # lognormal con media = múltiplo base

    out = dcf_vectorized(
        base.revenue0, growth, base.stable_growth, base.current_margin, margin, base.margin_year,
        base.tax_rate, base.sales_to_capital, base.cost_of_capital, base.mature_cost_of_capital,
        base.terminal_roc, base.risk_free, base.nol0, base.years, base.high_growth_years,
    )
    if settings.include_failure:
        survived = rng.random(settings.n_sims) < base.survival_prob
        op = np.where(survived, out["operating_value"], apply_survival(out["operating_value"], 0.0, base.distress_proceeds))
    else:
        survived = np.ones(settings.n_sims, dtype=bool)
        op = out["operating_value"]
    equity = equity_floor(op + base.cash - base.debt)  # responsabilidad limitada: nunca menos de 0

    year_idx = min(max(exit_year, 1), base.years) - 1
    exit_value = out["revenue"][:, year_idx] * multiple
    exit_value = np.where(survived, exit_value, 0.0)
    moic = stake_entry * (1.0 - future_dilution) * exit_value / investment

    def pct(a: np.ndarray) -> dict[str, float]:
        p10, p50, p90 = np.percentile(a, [10, 50, 90])
        return {"P10": float(p10), "P50": float(p50), "P90": float(p90), "Media": float(a.mean())}

    percentiles = {"Valor DCF": pct(equity), "Valor de salida": pct(exit_value), "MOIC": pct(moic)}
    if survived.any() and not survived.all():
        percentiles["Valor DCF si sobrevive"] = pct(equity[survived])
        percentiles["MOIC si hay salida"] = pct(moic[survived])

    return MCResult(
        growth=growth, margin=margin, multiple=multiple, survived=survived,
        dcf_equity=equity, exit_value=exit_value, moic=moic,
        percentiles=percentiles,
        prob_moic_target=float((moic >= settings.moic_target).mean()),
        prob_loss=float((moic < 1.0).mean()),
    )
