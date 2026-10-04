"""Cálculos de valoración: beta, costo de capital, DCF, método VC, múltiplos y escenarios.

Este módulo no lee datos: recibe números ya resueltos. Así, añadir una fuente nueva no lo toca.
Las funciones del DCF aceptan escalares o arrays de numpy (una fila por simulación).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.cleaning import annual_to_monthly

# ---------------------------------------------------------------- beta y costo de capital


def relever_beta(unlevered_beta: float, de_ratio: float, tax_rate: float) -> float:
    """Beta apalancada (Hamada): βL = βU · (1 + (1 − t) · D/E)."""
    return unlevered_beta * (1.0 + (1.0 - tax_rate) * de_ratio)


def unlever_beta(levered_beta: float, de_ratio: float, tax_rate: float) -> float:
    return levered_beta / (1.0 + (1.0 - tax_rate) * de_ratio)


def total_beta(market_beta: float, correlation: float) -> float:
    """Beta total = beta de mercado / correlación con el mercado (inversor no diversificado)."""
    if correlation <= 0:
        raise ValueError("La correlación con el mercado debe ser positiva")
    return market_beta / correlation


def cost_of_equity(risk_free: float, beta: float, erp: float, extra_premium: float = 0.0) -> float:
    """CAPM más primas adicionales (tamaño, iliquidez)."""
    return risk_free + beta * erp + extra_premium


def wacc(cost_equity: float, pre_tax_cost_debt: float, tax_rate: float, de_ratio: float) -> float:
    weight_debt = de_ratio / (1.0 + de_ratio)
    return (1.0 - weight_debt) * cost_equity + weight_debt * pre_tax_cost_debt * (1.0 - tax_rate)


@dataclass
class DiscountRateResult:
    unlevered_beta: float
    beta_used: float
    beta_type: str
    cost_of_equity: float
    cost_of_capital: float


def discount_rate(
    unlevered_beta_cash_adj: float,
    correlation: float,
    use_total_beta: bool,
    risk_free: float,
    erp: float,
    size_premium: float,
    illiquidity_premium: float,
    de_ratio: float = 0.0,
    tax_rate: float = 0.25,
    pre_tax_cost_debt: float = 0.0,
) -> DiscountRateResult:
    """Beta de la industria → reapalancada con el D/E de la startup → costo del equity y WACC."""
    beta_u = unlevered_beta_cash_adj
    if use_total_beta:
        beta_u = total_beta(beta_u, correlation)
    beta_l = relever_beta(beta_u, de_ratio, tax_rate)
    coe = cost_of_equity(risk_free, beta_l, erp, size_premium + illiquidity_premium)
    return DiscountRateResult(
        unlevered_beta=beta_u,
        beta_used=beta_l,
        beta_type="total" if use_total_beta else "mercado",
        cost_of_equity=coe,
        cost_of_capital=wacc(coe, pre_tax_cost_debt, tax_rate, de_ratio),
    )


# ---------------------------------------------------------------- DCF


@dataclass
class DCFInputs:
    revenue0: float                 # ingresos de los últimos 12 meses
    growth_high: float              # crecimiento anual en la fase de alto crecimiento
    stable_growth: float            # crecimiento perpetuo (se limita a la tasa libre de riesgo)
    current_margin: float           # margen operativo actual (puede ser negativo)
    target_margin: float            # margen operativo objetivo (por defecto, el de la industria)
    margin_year: int                # año en que se alcanza el margen objetivo
    tax_rate: float                 # tasa marginal de impuestos
    sales_to_capital: float         # ventas por cada unidad de capital invertido (industria)
    cost_of_capital: float          # tasa de descuento de la startup
    mature_cost_of_capital: float   # tasa de descuento al final de la proyección (industria)
    terminal_roc: float             # ROC en la fase estable
    risk_free: float
    survival_prob: float = 1.0
    distress_proceeds: float = 0.0  # fracción del valor operativo recuperada si fracasa
    nol0: float = 0.0               # pérdidas fiscales acumuladas al inicio
    cash: float = 0.0
    debt: float = 0.0
    years: int = 10
    high_growth_years: int = 5


@dataclass
class DCFResult:
    projection: pd.DataFrame
    pv_fcff: float
    terminal_value: float
    pv_terminal: float
    operating_value: float          # empresa en marcha, antes de supervivencia
    survival_adjusted_value: float
    equity_value: float
    stable_growth_used: float
    capital_need: float             # caja que consumen los FCFF negativos acumulados
    warnings: list[str]


def _growth_path(g_high, g_stable, years: int, hg_years: int) -> np.ndarray:
    """Crecimiento constante `hg_years` años y luego convergencia lineal a la tasa estable."""
    g_high = np.atleast_1d(np.asarray(g_high, dtype=float))[:, None]
    g_stable = np.atleast_1d(np.asarray(g_stable, dtype=float))[:, None]
    t = np.arange(1, years + 1)[None, :]
    hg = min(hg_years, years)
    frac = np.clip((t - hg) / max(years - hg, 1), 0.0, 1.0)
    return g_high + (g_stable - g_high) * frac


def _linear_path(start, end, years: int, reach_year: int) -> np.ndarray:
    start = np.atleast_1d(np.asarray(start, dtype=float))[:, None]
    end = np.atleast_1d(np.asarray(end, dtype=float))[:, None]
    t = np.arange(1, years + 1)[None, :]
    frac = np.clip(t / max(reach_year, 1), 0.0, 1.0)
    return start + (end - start) * frac


def _rate_path(r_start, r_mature, years: int, hg_years: int) -> np.ndarray:
    """Tasa de descuento fija en alto crecimiento y convergencia lineal a la tasa madura."""
    return _growth_path(r_start, r_mature, years, hg_years)


def dcf_vectorized(
    revenue0,
    growth_high,
    stable_growth,
    current_margin,
    target_margin,
    margin_year: int,
    tax_rate: float,
    sales_to_capital,
    cost_of_capital,
    mature_cost_of_capital,
    terminal_roc,
    risk_free: float,
    nol0: float = 0.0,
    years: int = 10,
    high_growth_years: int = 5,
) -> dict[str, np.ndarray]:
    """DCF de FCFF para n escenarios a la vez. Devuelve matrices (n, years) y vectores (n,)."""
    g_stable = np.minimum(np.atleast_1d(np.asarray(stable_growth, dtype=float)), risk_free)
    growth = _growth_path(growth_high, g_stable, years, high_growth_years)
    n = growth.shape[0]
    revenue0 = np.broadcast_to(np.asarray(revenue0, dtype=float), (n,))
    revenue = revenue0[:, None] * np.cumprod(1.0 + growth, axis=1)
    margin = _linear_path(np.broadcast_to(current_margin, (n,)), np.broadcast_to(target_margin, (n,)), years, margin_year)
    ebit = revenue * margin

    # Impuestos con pérdidas fiscales acumuladas (NOL)
    taxes = np.zeros_like(ebit)
    nol = np.full(n, float(nol0))
    for t in range(years):
        e = ebit[:, t]
        loss = e < 0
        nol = np.where(loss, nol - e, nol)
        used = np.where(loss, 0.0, np.minimum(nol, e))
        nol = nol - used
        taxes[:, t] = np.where(loss, 0.0, (e - used) * tax_rate)

    prev_rev = np.concatenate([revenue0[:, None], revenue[:, :-1]], axis=1)
    s2c = np.broadcast_to(np.asarray(sales_to_capital, dtype=float), (n,))[:, None]
    reinvestment = (revenue - prev_rev) / s2c
    fcff = ebit - taxes - reinvestment

    rates = _rate_path(np.broadcast_to(cost_of_capital, (n,)), np.broadcast_to(mature_cost_of_capital, (n,)), years, high_growth_years)
    discount = np.cumprod(1.0 + rates, axis=1)
    pv_fcff = (fcff / discount).sum(axis=1)

    r_mature = rates[:, -1]
    roc = np.broadcast_to(np.asarray(terminal_roc, dtype=float), (n,))
    roc = np.where(roc > 0, roc, r_mature)
    ebit_next = revenue[:, -1] * (1.0 + g_stable) * margin[:, -1]
    nopat_next = np.where(ebit_next > 0, ebit_next * (1.0 - tax_rate), ebit_next)
    reinvest_rate = np.clip(g_stable / roc, 0.0, 1.0)
    fcff_next = nopat_next * (1.0 - reinvest_rate)
    spread = np.maximum(r_mature - g_stable, 1e-4)
    terminal_value = fcff_next / spread
    pv_terminal = terminal_value / discount[:, -1]

    cum = np.cumsum(fcff, axis=1)
    capital_need = np.maximum(-cum.min(axis=1), 0.0)

    return {
        "growth": growth, "revenue": revenue, "margin": margin, "ebit": ebit, "taxes": taxes,
        "reinvestment": reinvestment, "fcff": fcff, "rates": rates, "discount": discount,
        "pv_fcff": pv_fcff, "terminal_value": terminal_value, "pv_terminal": pv_terminal,
        "operating_value": pv_fcff + pv_terminal, "stable_growth": g_stable,
        "capital_need": capital_need,
    }


def equity_floor(value):
    """Responsabilidad limitada: el equity no vale menos de 0 (escalar o array)."""
    return np.maximum(value, 0.0) if isinstance(value, np.ndarray) else max(float(value), 0.0)


def apply_survival(operating_value, survival_prob, distress_proceeds: float = 0.0):
    """Valor esperado = p · valor en marcha + (1 − p) · valor de liquidación."""
    return survival_prob * operating_value + (1.0 - survival_prob) * distress_proceeds * np.maximum(operating_value, 0.0)


def dcf(inp: DCFInputs) -> DCFResult:
    out = dcf_vectorized(
        inp.revenue0, inp.growth_high, inp.stable_growth, inp.current_margin, inp.target_margin,
        inp.margin_year, inp.tax_rate, inp.sales_to_capital, inp.cost_of_capital,
        inp.mature_cost_of_capital, inp.terminal_roc, inp.risk_free, inp.nol0, inp.years,
        inp.high_growth_years,
    )
    years = np.arange(1, inp.years + 1)
    proj = pd.DataFrame({
        "Año": years,
        "Crecimiento": out["growth"][0],
        "Ingresos": out["revenue"][0],
        "Margen operativo": out["margin"][0],
        "EBIT": out["ebit"][0],
        "Impuestos": out["taxes"][0],
        "Reinversión": out["reinvestment"][0],
        "FCFF": out["fcff"][0],
        "Tasa de descuento": out["rates"][0],
        "Factor de descuento": 1.0 / out["discount"][0],
        "VP del FCFF": out["fcff"][0] / out["discount"][0],
    })
    op_value = float(out["operating_value"][0])
    adj = float(apply_survival(op_value, inp.survival_prob, inp.distress_proceeds))
    g_used = float(out["stable_growth"][0])

    warnings = []
    if adj + inp.cash - inp.debt < 0:
        warnings.append(
            "El valor del DCF es negativo: el plan consume más caja de la que genera. "
            "El equity se limita a 0, porque un accionista no puede perder más de lo invertido (responsabilidad limitada). "
            "Revisa el margen objetivo y el ratio ventas / capital."
        )
    if inp.stable_growth > inp.risk_free:
        warnings.append(
            f"El crecimiento estable ({inp.stable_growth:.2%}) supera la tasa libre de riesgo; "
            f"se limita a {inp.risk_free:.2%}."
        )
    if out["terminal_value"][0] < 0:
        warnings.append("El valor terminal es negativo: el margen final no cubre la reinversión necesaria.")
    if inp.mature_cost_of_capital <= g_used:
        warnings.append("La tasa de descuento madura no supera el crecimiento estable; el valor terminal no es fiable.")

    return DCFResult(
        projection=proj,
        pv_fcff=float(out["pv_fcff"][0]),
        terminal_value=float(out["terminal_value"][0]),
        pv_terminal=float(out["pv_terminal"][0]),
        operating_value=op_value,
        survival_adjusted_value=adj,
        equity_value=equity_floor(adj + inp.cash - inp.debt),
        stable_growth_used=g_used,
        capital_need=float(out["capital_need"][0]),
        warnings=warnings,
    )


# ---------------------------------------------------------------- ronda y método VC


@dataclass
class RoundTerms:
    investment: float
    pre_money: float
    stake: float  # participación del inversor al entrar (post-money)

    @property
    def post_money(self) -> float:
        return self.pre_money + self.investment


def resolve_round(investment: float | None, pre_money: float | None, stake: float | None) -> RoundTerms:
    """Completa la tercera variable de la ronda: participación = inversión / (pre-money + inversión)."""
    given = [v is not None for v in (investment, pre_money, stake)]
    if sum(given) < 2:
        raise ValueError("Indica al menos dos de: inversión, pre-money y participación")
    if (investment is not None and investment <= 0) or (pre_money is not None and pre_money <= 0):
        raise ValueError("La inversión y la pre-money deben ser mayores que cero")
    if investment is not None and pre_money is not None:
        return RoundTerms(investment, pre_money, investment / (pre_money + investment))
    if investment is not None and stake is not None:
        if not 0 < stake < 1:
            raise ValueError("La participación debe estar entre 0 % y 100 %")
        return RoundTerms(investment, investment / stake - investment, stake)
    if not 0 < stake < 1:
        raise ValueError("La participación debe estar entre 0 % y 100 %")
    post = pre_money / (1.0 - stake)
    return RoundTerms(post - pre_money, pre_money, stake)


def round_consistency(investment: float, pre_money: float, stake: float, tol: float = 0.005) -> str | None:
    implied = investment / (pre_money + investment)
    if abs(implied - stake) > tol:
        return (
            f"La participación indicada ({stake:.2%}) no cuadra con inversión y pre-money "
            f"(implica {implied:.2%})."
        )
    return None


@dataclass
class VCResult:
    exit_value: float
    discount_rate: float
    mode: str
    retention: float
    survival_prob: float
    post_money: float
    pre_money: float
    required_stake: float           # participación necesaria hoy para lograr la tasa objetivo
    required_stake_at_exit: float


def vc_method(
    exit_value: float,
    years_to_exit: float,
    investment: float,
    discount_rate: float,
    future_dilution: float,
    survival_prob: float = 1.0,
    mode: str = "irr",
) -> VCResult:
    """Método VC.

    mode="irr": se descuenta a la IRR objetivo, que ya incluye el riesgo de fracaso (sin supervivencia).
    mode="survival": se descuenta a una tasa sin riesgo de fracaso (p. ej. costo del equity)
    y se multiplica por la probabilidad de supervivencia. Nunca se aplican las dos a la vez.
    """
    retention = 1.0 - future_dilution
    p = survival_prob if mode == "survival" else 1.0
    post = exit_value * retention * p / (1.0 + discount_rate) ** years_to_exit
    required = investment / post if post > 0 else math.inf
    return VCResult(
        exit_value=exit_value, discount_rate=discount_rate, mode=mode, retention=retention,
        survival_prob=p, post_money=post, pre_money=post - investment,
        required_stake=required, required_stake_at_exit=required * retention,
    )


@dataclass
class DealReturns:
    stake_entry: float
    stake_exit: float
    proceeds: float
    moic: float
    irr: float
    expected_moic: float


def deal_returns(
    exit_value: float, years_to_exit: float, terms: RoundTerms, future_dilution: float, survival_prob: float
) -> DealReturns:
    """Retorno del inversor con las condiciones propuestas, si la empresa llega a la salida."""
    stake_exit = terms.stake * (1.0 - future_dilution)
    proceeds = stake_exit * exit_value
    moic = proceeds / terms.investment
    irr = moic ** (1.0 / years_to_exit) - 1.0 if moic > 0 else -1.0
    return DealReturns(terms.stake, stake_exit, proceeds, moic, irr, moic * survival_prob)


def exit_ebitda_margin(operating_margin_exit: float, industry_ebitda_margin: float, industry_operating_margin: float) -> float:
    """Margen EBITDA a la salida = margen operativo proyectado + D&A/ventas de la industria."""
    da = max(industry_ebitda_margin - industry_operating_margin, 0.0)
    return operating_margin_exit + da


# ---------------------------------------------------------------- múltiplos


def multiple_value(metric_value: float, multiple: float, illiquidity_discount: float) -> float:
    """Valor por múltiplo con descuento por iliquidez. NaN si la métrica no es positiva."""
    if metric_value is None or metric_value <= 0 or multiple is None or math.isnan(multiple):
        return math.nan
    return metric_value * multiple * (1.0 - illiquidity_discount)


# ---------------------------------------------------------------- caja


def runway_months(cash: float, monthly_burn: float) -> float:
    return math.inf if monthly_burn <= 0 else cash / monthly_burn


def monthly_cash_projection(
    cash: float,
    revenue_annual: float,
    monthly_burn: float,
    revenue_growth_annual: float,
    cost_growth_annual: float,
    months: int = 36,
    new_money: float = 0.0,
) -> pd.DataFrame:
    """Caja mes a mes. Las tasas anuales se pasan a mensuales con (1 + g)^(1/12) − 1."""
    g_rev = annual_to_monthly(revenue_growth_annual)
    g_cost = annual_to_monthly(cost_growth_annual)
    rev0 = revenue_annual / 12.0
    cost0 = rev0 + monthly_burn
    t = np.arange(1, months + 1)
    rev = rev0 * (1.0 + g_rev) ** t
    cost = cost0 * (1.0 + g_cost) ** t
    burn = cost - rev
    balance = cash + new_money - np.cumsum(burn)
    return pd.DataFrame({"Mes": t, "Ingresos": rev, "Costos": cost, "Consumo de caja": burn, "Caja": balance})


def implied_dilution(funding_gap: float, post_money: float) -> float:
    """Dilución si el déficit de financiación se levantara hoy al post-money propuesto (cota superior)."""
    if funding_gap <= 0:
        return 0.0
    return funding_gap / (post_money + funding_gap)


# ---------------------------------------------------------------- escenarios


@dataclass
class ScenarioFactors:
    name: str
    growth: float
    margin: float
    multiple: float


DEFAULT_SCENARIOS = [
    ScenarioFactors("Pesimista", 0.6, 0.8, 0.7),
    ScenarioFactors("Base", 1.0, 1.0, 1.0),
    ScenarioFactors("Optimista", 1.3, 1.15, 1.3),
]


def scale_margin(margin: float, factor: float) -> float:
    """Escala un margen hacia arriba o hacia abajo en términos de mejora (sirve para márgenes negativos)."""
    return margin * factor if margin >= 0 else margin * (2.0 - factor)


def scenario_inputs(base: DCFInputs, f: ScenarioFactors) -> DCFInputs:
    return replace(
        base,
        growth_high=base.growth_high * f.growth,
        target_margin=scale_margin(base.target_margin, f.margin),
    )
