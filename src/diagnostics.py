"""Diagnóstico de resultados extremos: detectar, explicar y proponer qué probar.

Capa 1: reglas que detectan resultados extremos o contradictorios.
Capa 2: una nota por señal que descompone el resultado en sus motores, con los números del modelo.
Capa 3: alternativas razonables, cada una recalculada con el modelo completo (`evaluate`) para mostrar
        su impacto antes de aplicarla.

El módulo no depende de Streamlit: recibe un `DiagState` con los números y una función `evaluate(changes)`
que devuelve el resultado del modelo con esos cambios. Así se puede probar de forma aislada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from src.charts import fmt_money, fmt_mult, fmt_num, fmt_pct

SEVERITY_ORDER = {"alta": 0, "media": 1, "baja": 2}
SEVERITY_ICON = {"alta": "🔴", "media": "🟠", "baja": "🔵"}

# Cambios que entiende `evaluate` (y que la app sabe aplicar a la barra lateral)
CHANGE_KEYS = {"target_margin", "sales_to_capital", "use_total_beta", "margin_year", "growth", "exit_multiple", "stable_growth"}


@dataclass
class DiagState:
    currency: str
    industry: str
    revenue: float
    growth: float
    current_margin: float
    target_margin: float
    industry_margin: float
    margin_year: int
    sales_to_capital: float
    use_total_beta: bool
    beta_unlevered: float
    correlation: float
    beta_used: float
    risk_free: float
    erp: float
    extra_premia: float
    cost_of_capital: float
    stable_growth: float
    dcf_equity: float
    dcf_operating: float
    dcf_raw_equity: float          # antes del suelo en 0
    pv_terminal: float
    capital_need: float
    investment: float
    pre_money: float
    exit_year: int
    exit_basis: str
    exit_multiple: float
    small_cap_multiple: float
    exit_revenue: float
    exit_value: float
    vc_pre_money: float
    vc_rate: float
    future_dilution: float
    stake: float
    moic: float
    method_mids: dict[str, float] = field(default_factory=dict)


@dataclass
class Alternative:
    label: str
    why: str
    changes: dict
    outcome: dict = field(default_factory=dict)  # dcf, vc_pre, moic tras el cambio

    def result_text(self, focus: str, currency: str) -> str:
        v = self.outcome.get(focus, math.nan)
        if focus == "moic":
            return fmt_mult(v)
        return fmt_money(v, currency)


@dataclass
class Note:
    id: str
    severity: str
    title: str
    explanation: str
    drivers: list[str]
    focus: str                     # métrica que mueven las alternativas: dcf, vc_pre o moic
    alternatives: list[Alternative] = field(default_factory=list)
    tabs: tuple[str, ...] = ()     # pestañas donde se muestra además del Resumen

    def as_text(self, currency: str) -> dict:
        """Versión en texto para el memo."""
        return {
            "title": self.title, "explanation": self.explanation, "drivers": list(self.drivers),
            "alternatives": [f"{a.label}: {a.result_text(self.focus, currency)}. {a.why}" for a in self.alternatives],
        }


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def diagnose(s: DiagState, evaluate: Callable[[dict], dict]) -> list[Note]:
    m = lambda x: fmt_money(x, s.currency)  # noqa: E731
    notes: list[Note] = []
    reinv_per_dollar = 1.0 / s.sales_to_capital if s.sales_to_capital > 0 else math.nan

    def alt(label: str, why: str, changes: dict) -> Alternative:
        assert set(changes) <= CHANGE_KEYS, changes
        a = Alternative(label, why, changes)
        try:
            a.outcome = evaluate(changes)
        except Exception:  # noqa: BLE001: una alternativa que no se puede calcular no rompe el diagnóstico
            a.outcome = {}
        return a

    def dcf_levers() -> list[Alternative]:
        out = []
        if s.target_margin < max(s.industry_margin, 0.25):
            tm = max(s.target_margin + 0.10, min(0.35, s.industry_margin + 0.10))
            out.append(alt(f"Margen objetivo del {fmt_pct(tm)}", "Si el modelo de negocio es más rentable que la media de la industria "
                           "(por ejemplo, un SaaS maduro).", {"target_margin": tm}))
        out.append(alt(f"Ventas / capital de {fmt_num(s.sales_to_capital * 2, 2)} (el doble)",
                       "Si la empresa necesita menos capital para crecer que la media de su industria.",
                       {"sales_to_capital": s.sales_to_capital * 2}))
        if s.use_total_beta:
            out.append(alt("Beta de mercado en lugar de total", "Si el inversor está diversificado (un fondo con cartera amplia) y "
                           "no debe cobrar por el riesgo específico de la empresa.", {"use_total_beta": False}))
        if s.margin_year > 3:
            out.append(alt(f"Alcanzar el margen en el año {s.margin_year - 2}", "Si el camino a la rentabilidad es más corto.",
                           {"margin_year": s.margin_year - 2}))
        return out

    # 1. DCF negativo (o con suelo en 0)
    if s.dcf_raw_equity < 0:
        drivers = [
            f"Margen objetivo del {fmt_pct(s.target_margin)} en el año {s.margin_year} (industria: {fmt_pct(s.industry_margin)}): "
            "es el margen que alcanza la empresa al final de la proyección.",
            f"Reinversión de {fmt_num(reinv_per_dollar, 2)} por cada unidad de ventas nuevas (ventas / capital de "
            f"{fmt_num(s.sales_to_capital, 2)}): crecer al {fmt_pct(s.growth, 0)} consume {m(s.capital_need)} de caja.",
            f"Tasa de descuento inicial del {fmt_pct(s.cost_of_capital)} (beta {'total' if s.use_total_beta else 'de mercado'} de "
            f"{fmt_num(s.beta_used, 2)}): los flujos positivos lejanos valen poco hoy.",
        ]
        notes.append(Note(
            "dcf_negative", "alta", f"El DCF es negativo ({m(s.dcf_raw_equity)}) y se muestra como 0",
            "El plan consume más caja de la que genera en valor presente. El equity se limita a 0 porque un accionista no puede "
            "perder más de lo invertido, pero el resultado indica que, con estos supuestos, crecer destruye valor.",
            drivers, "dcf", dcf_levers(), ("DCF",)))

    # 2. Métodos que no se ponen de acuerdo
    mids = {k: v for k, v in s.method_mids.items() if _ok(v) and v > 0}
    if len(mids) >= 2 and max(mids.values()) / min(mids.values()) > 3:
        hi_k = max(mids, key=mids.get)
        lo_k = min(mids, key=mids.get)
        drivers = [f"{k}: {m(v)}" for k, v in sorted(mids.items(), key=lambda kv: -kv[1])]
        drivers.append(f"El método VC valora la salida con un múltiplo de mercado ({s.exit_basis} de {fmt_mult(s.exit_multiple)}) "
                       f"y descuenta al {fmt_pct(s.vc_rate, 0)}; el DCF cobra la reinversión para crecer y descuenta los flujos "
                       f"al {fmt_pct(s.cost_of_capital)}.")
        alts = []
        if s.exit_basis == "EV/Sales" and _ok(s.small_cap_multiple) and s.exit_multiple > s.small_cap_multiple * 1.5:
            alts.append(alt(f"Múltiplo de salida de {fmt_mult(s.small_cap_multiple)} (cotizadas pequeñas)",
                            "Si la empresa saldrá a una valoración de empresa mediana, no de las grandes de su industria.",
                            {"exit_multiple": s.small_cap_multiple}))
        if s.use_total_beta:
            alts.append(alt("Beta de mercado en lugar de total", "Acerca el DCF si el inversor está diversificado.",
                            {"use_total_beta": False}))
        notes.append(Note(
            "methods_disagree", "media",
            f"Los métodos no se ponen de acuerdo: {hi_k} da {fmt_num(mids[hi_k] / mids[lo_k], 1)} veces más que {lo_k}",
            "Es normal que el DCF y el método VC difieran en una startup, pero una diferencia tan grande indica que uno de los dos "
            "depende de un supuesto muy optimista o muy exigente. Conviene explicar en el memo cuál se usa como referencia.",
            drivers, "vc_pre" if alts and "exit_multiple" in alts[0].changes else "dcf", alts, ("DCF", "Método VC")))

    # 3. El valor depende casi todo del valor terminal
    if s.dcf_operating > 0 and s.pv_terminal / s.dcf_operating > 0.85:
        share = s.pv_terminal / s.dcf_operating
        lower_g = max(s.stable_growth - 0.01, 0.0)
        notes.append(Note(
            "terminal_heavy", "media", f"El {fmt_pct(share, 0)} del valor del DCF llega después del año 10",
            "Casi todo el valor depende del valor terminal: de cómo será la empresa cuando ya sea madura. Es habitual en "
            "startups, pero hace el resultado muy sensible al crecimiento estable y al margen final.",
            [f"Valor terminal descontado: {m(s.pv_terminal)} de un valor operativo de {m(s.dcf_operating)}.",
             f"Crecimiento estable del {fmt_pct(s.stable_growth)} y margen final del {fmt_pct(s.target_margin)}."],
            "dcf",
            [alt(f"Crecimiento estable del {fmt_pct(lower_g)}", "Para ver cuánto cae el valor con un supuesto perpetuo más prudente.",
                 {"stable_growth": lower_g}),
             alt(f"Margen objetivo del {fmt_pct(s.target_margin - 0.05)}", "Para ver cuánto depende el valor del margen final.",
                 {"target_margin": s.target_margin - 0.05})],
            ("DCF",)))

    # 4. Tasa de descuento muy alta
    if s.cost_of_capital > 0.25:
        drivers = [f"Coste del equity = {fmt_pct(s.risk_free, 2)} + {fmt_num(s.beta_used, 2)} × {fmt_pct(s.erp, 2)} + "
                   f"{fmt_pct(s.extra_premia)} de primas = {fmt_pct(s.cost_of_capital)}."]
        if s.use_total_beta:
            drivers.append(f"Beta total = beta desapalancada / correlación = {fmt_num(s.beta_unlevered, 2)} / "
                           f"{fmt_pct(s.correlation)}: divide entre la correlación porque supone un inversor no diversificado.")
        alts = [alt("Beta de mercado en lugar de total", "Si el inversor está diversificado.", {"use_total_beta": False})] if s.use_total_beta else []
        notes.append(Note(
            "high_discount", "baja", f"Tasa de descuento del {fmt_pct(s.cost_of_capital)}",
            "Una tasa tan alta es coherente con el riesgo de una startup, pero reduce mucho el valor de los flujos lejanos.",
            drivers, "dcf", alts, ("DCF",)))

    # 5. MOIC irreal
    if _ok(s.moic) and (s.moic > 20 or s.moic < 1):
        cagr = (s.exit_revenue / s.revenue) ** (1 / s.exit_year) - 1 if s.revenue > 0 and s.exit_revenue > 0 else math.nan
        drivers = [
            f"MOIC = participación × (1 − dilución) × valor de salida / inversión = {fmt_pct(s.stake)} × "
            f"(1 − {fmt_pct(s.future_dilution, 0)}) × {m(s.exit_value)} / {m(s.investment)}.",
            f"Valor de salida = ingresos del año {s.exit_year} ({m(s.exit_revenue)}) × {fmt_mult(s.exit_multiple)}.",
        ]
        if _ok(cagr):
            drivers.append(f"Los ingresos pasan de {m(s.revenue)} a {m(s.exit_revenue)}: un {fmt_pct(cagr, 0)} anual compuesto.")
        alts = []
        if s.exit_basis == "EV/Sales" and _ok(s.small_cap_multiple) and s.moic > 20:
            alts.append(alt(f"Múltiplo de salida de {fmt_mult(s.small_cap_multiple)} (cotizadas pequeñas)",
                            "Salida a una valoración de empresa mediana.", {"exit_multiple": s.small_cap_multiple}))
        if s.moic > 20:
            alts.append(alt(f"Crecimiento del {fmt_pct(s.growth * 0.7, 0)} (un 30 % menos)", "Si el crecimiento actual no se mantiene.",
                            {"growth": s.growth * 0.7}))
        notes.append(Note(
            "moic_extreme", "media",
            f"MOIC de {fmt_mult(s.moic)} si hay salida" + (": muy por encima de lo habitual" if s.moic > 20 else ": se pierde dinero"),
            "Un MOIC por encima de 20x es posible en el mejor caso, pero suele indicar un crecimiento o un múltiplo de salida muy "
            "optimistas." if s.moic > 20 else "Con estas condiciones, ni siquiera el caso de éxito devuelve lo invertido.",
            drivers, "moic", alts, ("Método VC",)))

    # 6. Múltiplo de salida de empresa grande
    if s.exit_basis == "EV/Sales" and _ok(s.small_cap_multiple) and s.exit_multiple > 2 * s.small_cap_multiple:
        mid = math.sqrt(s.exit_multiple * s.small_cap_multiple)
        notes.append(Note(
            "exit_multiple_high", "media",
            f"Múltiplo de salida de {fmt_mult(s.exit_multiple)}, más del doble del de las cotizadas pequeñas ({fmt_mult(s.small_cap_multiple)})",
            "El múltiplo de la industria viene de empresas cotizadas grandes. Usarlo supone que la startup saldrá valorada como "
            "una de ellas.",
            [f"Múltiplo de la industria ({s.industry}): {fmt_mult(s.exit_multiple)}.",
             f"Decil más pequeño de cotizadas de EE. UU.: {fmt_mult(s.small_cap_multiple)} (Damodaran)."],
            "vc_pre",
            [alt(f"Múltiplo de {fmt_mult(mid)} (punto medio)", "Un punto intermedio entre ambos.", {"exit_multiple": mid}),
             alt(f"Múltiplo de {fmt_mult(s.small_cap_multiple)}", "La referencia más prudente.", {"exit_multiple": s.small_cap_multiple})],
            ("Método VC",)))

    # 7. El plan devora caja
    if s.investment > 0 and s.capital_need > 5 * s.investment:
        notes.append(Note(
            "capital_hungry", "media",
            f"El plan consume {m(s.capital_need)}: {fmt_num(s.capital_need / s.investment, 1)} veces la ronda",
            "Para cumplir la proyección del DCF harán falta muchas más rondas. Cada una diluye a los inversores actuales.",
            [f"Reinversión de {fmt_num(reinv_per_dollar, 2)} por cada unidad de ventas nuevas (ventas / capital de "
             f"{fmt_num(s.sales_to_capital, 2)}, media de la industria).",
             f"Crecimiento del {fmt_pct(s.growth, 0)} anual y margen actual del {fmt_pct(s.current_margin)}."],
            "dcf",
            [alt(f"Ventas / capital de {fmt_num(s.sales_to_capital * 2, 2)} (el doble)", "Si el negocio necesita menos capital para crecer.",
                 {"sales_to_capital": s.sales_to_capital * 2})],
            ("Caja y ronda",)))

    # 8. Ingresos de salida que exigen un crecimiento muy alto
    if s.revenue > 0 and s.exit_revenue / s.revenue > 50:
        notes.append(Note(
            "revenue_explosion", "media",
            f"Los ingresos se multiplican por {fmt_num(s.exit_revenue / s.revenue, 0)} hasta el año {s.exit_year}",
            "Muy pocas empresas mantienen ese ritmo tantos años. El valor de salida, y con él el MOIC, depende de que se cumpla.",
            [f"Crecimiento del {fmt_pct(s.growth, 0)} anual durante los años de alto crecimiento.",
             f"Ingresos del año {s.exit_year}: {m(s.exit_revenue)} frente a {m(s.revenue)} hoy."],
            "moic",
            [alt(f"Crecimiento del {fmt_pct(s.growth * 0.7, 0)} (un 30 % menos)", "Un plan más conservador.", {"growth": s.growth * 0.7})],
            ("Método VC",)))

    return sorted(notes, key=lambda n: SEVERITY_ORDER[n.severity])
