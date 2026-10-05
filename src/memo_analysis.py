"""Comentarios analíticos del memo: interpretan las cifras que muestra cada sección.

Cada función recibe el contexto del memo y devuelve frases listas para el comité. Son reglas
explícitas (umbrales a la vista), no texto libre: el mismo análisis da siempre la misma lectura y
cada frase cita las cifras en las que se apoya. Si falta un dato, la frase que lo necesita se omite.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.charts import fmt_money, fmt_mult, fmt_num, fmt_pct

if TYPE_CHECKING:  # evita la importación circular con src.memo
    from src.memo import MemoContext


def _ok(*xs) -> bool:
    return all(x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) for x in xs)


def _m(c: MemoContext, v: float) -> str:
    return fmt_money(v, c.currency)


def required_moic(c: MemoContext) -> float:
    """MOIC que exige la IRR objetivo de la etapa en el plazo de salida."""
    return (1 + c.target_irr) ** c.exit_year if _ok(c.target_irr) else math.nan


def _mids(c: MemoContext) -> list[tuple[str, float]]:
    return [(r["method"], r["mid"]) for r in c.ff_rows if _ok(r.get("mid")) and r["mid"] > 0]


# ---------------------------------------------------------------- resumen ejecutivo


def key_points(c: MemoContext) -> list[str]:
    """Las cuatro o cinco ideas que el comité debe llevarse: precio, retorno, riesgo, caja y dispersión."""
    out = []
    mids = _mids(c)
    if mids and _ok(c.pre_money):
        med = float(np.median([v for _, v in mids]))
        lo_n, lo_v = min(mids, key=lambda t: t[1])
        hi_n, hi_v = max(mids, key=lambda t: t[1])
        gap = c.pre_money / med - 1
        if c.pre_money > hi_v:
            out.append(f"Precio exigente: la pre-money ({_m(c, c.pre_money)}) supera incluso el valor central más alto "
                       f"({hi_n}, {_m(c, hi_v)}) y está un {fmt_pct(gap, 0)} por encima de la mediana de los métodos.")
        elif c.pre_money < lo_v:
            out.append(f"Precio atractivo: la pre-money ({_m(c, c.pre_money)}) queda por debajo del valor central más bajo "
                       f"({lo_n}, {_m(c, lo_v)}), un {fmt_pct(-gap, 0)} por debajo de la mediana de los métodos.")
        else:
            out.append(f"Precio dentro del rango: la pre-money ({_m(c, c.pre_money)}) está entre {lo_n} ({_m(c, lo_v)}) y "
                       f"{hi_n} ({_m(c, hi_v)}), un {fmt_pct(abs(gap), 0)} {'por encima' if gap > 0 else 'por debajo'} "
                       "de la mediana de los métodos.")
    req = required_moic(c)
    if _ok(c.moic, req):
        verb = "supera" if c.moic >= req else "no alcanza"
        out.append(f"Retorno: si hay salida, el MOIC de {fmt_mult(c.moic)} {verb} el {fmt_mult(req)} que exige una IRR "
                   f"del {fmt_pct(c.target_irr, 0)} en {c.exit_year} años; ponderado por la supervivencia queda en "
                   f"{fmt_mult(c.expected_moic)}." if _ok(c.expected_moic) else
                   f"Retorno: si hay salida, el MOIC de {fmt_mult(c.moic)} {verb} el {fmt_mult(req)} que exige la IRR objetivo.")
    if _ok(c.mc_prob_fail, c.mc_prob_target):
        out.append(f"Riesgo: en la simulación la inversión fracasa en el {fmt_pct(c.mc_prob_fail, 0)} de los casos y alcanza "
                   f"un MOIC de {fmt_mult(c.mc_target, 1)} o más en el {fmt_pct(c.mc_prob_target, 0)}.")
    if _ok(c.runway):
        gap_txt = (f" y el plan deja un déficit de {_m(c, c.funding_gap)} tras esta ronda" if _ok(c.funding_gap) and c.funding_gap > 0
                   else " y la caja más esta ronda cubren el plan")
        out.append(f"Caja: {fmt_num(c.runway, 1)} meses de runway sin la ronda{gap_txt}.")
    if len(mids) >= 2:
        ratio = max(v for _, v in mids) / min(v for _, v in mids)
        if ratio > 3:
            out.append(f"Dispersión alta entre métodos ({fmt_num(ratio, 1)} veces entre el mayor y el menor): el valor depende "
                       "sobre todo de qué supuesto se acepte; ver la sección de notas.")
    return out


# ---------------------------------------------------------------- empresa y ronda


def read_company(c: MemoContext) -> list[str]:
    out = []
    if _ok(c.growth, c.current_margin):
        r40 = c.growth + c.current_margin
        out.append(f"Regla del 40: crecimiento ({fmt_pct(c.growth, 0)}) más margen operativo ({fmt_pct(c.current_margin, 0)}) "
                   f"suman {fmt_pct(r40, 0)}, " + ("por encima del 40 % que se usa como referencia de una empresa de software sana." if r40 >= 0.4 else
                                                  "por debajo del 40 % de referencia: el crecimiento no compensa todavía las pérdidas."))
    if _ok(c.burn, c.revenue) and c.revenue > 0 and c.burn > 0:
        out.append(f"El consumo de caja anual ({_m(c, c.burn * 12)}) equivale al {fmt_pct(c.burn * 12 / c.revenue, 0)} de los "
                   "ingresos de los últimos 12 meses.")
    if _ok(c.current_margin, c.target_margin):
        delta = c.target_margin - c.current_margin
        out.append(f"El plan necesita mejorar el margen operativo en {fmt_num(delta * 100, 1)} puntos (del "
                   f"{fmt_pct(c.current_margin)} al {fmt_pct(c.target_margin)}) en {c.margin_year} años"
                   + (": es el supuesto que más conviene validar." if delta > 0.3 else "."))
    return out


def read_round(c: MemoContext) -> list[str]:
    out = []
    if _ok(c.post_money, c.revenue) and c.revenue > 0:
        entry = c.post_money / c.revenue
        txt = f"La ronda valora la empresa a {fmt_mult(entry)} sus ingresos (post-money / ingresos)"
        if _ok(c.exit_multiple) and c.exit_basis == "EV/Sales":
            rel = "por encima" if entry > c.exit_multiple else "por debajo"
            txt += (f", {rel} del múltiplo de la industria ({fmt_mult(c.exit_multiple)}): "
                    + ("se paga hoy un precio que la salida tiene que superar." if entry > c.exit_multiple
                       else "hay margen para que el múltiplo se comprima y aun así se gane."))
        out.append(txt + ("" if txt.endswith(".") else "."))
    if _ok(c.stake, c.stake_exit):
        out.append(f"La participación pasa del {fmt_pct(c.stake)} al {fmt_pct(c.stake_exit)} en la salida por la dilución "
                   f"futura supuesta ({fmt_pct(c.future_dilution, 0)}): sin derechos de prorrata, el fondo pierde "
                   f"{fmt_num((c.stake - c.stake_exit) * 100, 1)} puntos.")
    return out


# ---------------------------------------------------------------- valoración


def read_valuation(c: MemoContext) -> list[str]:
    out = []
    mids = _mids(c)
    if len(mids) >= 2:
        hi_n, hi_v = max(mids, key=lambda t: t[1])
        lo_n, lo_v = min(mids, key=lambda t: t[1])
        ratio = hi_v / lo_v
        txt = (f"{hi_n} da el valor más alto ({_m(c, hi_v)}) y {lo_n} el más bajo ({_m(c, lo_v)}), "
               f"{fmt_num(ratio, 1)} veces más.")
        if ratio > 2:
            txt += (" Los métodos de mercado (múltiplos y método VC) reflejan lo que pagan hoy los inversores; el DCF, lo "
                    "que la empresa genera con su plan. Una brecha así suele indicar que el precio de mercado descuenta más "
                    "crecimiento del que el plan justifica, o al revés.")
        else:
            txt += " Los métodos coinciden razonablemente, lo que da solidez al rango."
        out.append(txt)
    if mids and _ok(c.pre_money):
        above = [n for n, v in mids if v >= c.pre_money]
        out.append(f"{len(above)} de {len(mids)} métodos sostienen la pre-money propuesta"
                   + (f" ({', '.join(above)})." if above else "."))
    return out


def read_dcf(c: MemoContext) -> list[str]:
    out = []
    if _ok(c.dcf_pv_terminal, c.dcf_operating) and c.dcf_operating > 0:
        share = c.dcf_pv_terminal / c.dcf_operating
        out.append(f"El {fmt_pct(share, 0)} del valor operativo viene del valor terminal"
                   + (" (más del 100 % porque los flujos de los 10 primeros años son negativos en conjunto)" if share > 1 else "")
                   + ": " + ("el DCF depende casi por completo de la empresa madura, así que es muy sensible al crecimiento "
                             "estable y al margen final." if share > 0.85 else "una proporción habitual en una empresa en crecimiento."))
    pr = c.projection
    if pr is not None and not pr.empty and "FCFF" in pr:
        neg = pr[pr["FCFF"] < 0]
        if len(neg):
            first_pos = pr[pr["FCFF"] >= 0]["Año"]
            out.append(f"El flujo de caja libre es negativo durante {len(neg)} de los 10 años"
                       + (f" y pasa a positivo en el año {int(first_pos.iloc[0])}" if len(first_pos) else " y no llega a ser positivo")
                       + ": hasta entonces la empresa consume capital y depende de nuevas rondas.")
        if "Margen operativo" in pr:
            m_pos = pr[pr["Margen operativo"] > 0]["Año"]
            if len(m_pos) and _ok(c.current_margin) and c.current_margin <= 0:
                out.append(f"El margen operativo se vuelve positivo en el año {int(m_pos.iloc[0])}.")
    if _ok(c.cost_of_capital):
        out.append(f"La tasa de descuento inicial ({fmt_pct(c.cost_of_capital)}) "
                   + ("es alta: con beta total se cobra el riesgo de una cartera poco diversificada; con beta de mercado el "
                      "DCF subiría." if c.cost_of_capital > 0.25 else "está en línea con el riesgo de una startup de esta etapa."))
    if _ok(c.dcf_survival_value) and c.dcf_survival_value + (c.cash or 0) - (c.debt or 0) < 0:
        out.append("Antes del suelo en cero el equity del DCF sería negativo: con estos supuestos el plan destruye valor y "
                   "el resultado lo sostienen solo la caja y la responsabilidad limitada.")
    return out


def read_vc(c: MemoContext) -> list[str]:
    out = []
    if _ok(c.required_stake, c.stake):
        if c.stake >= c.required_stake:
            out.append(f"Al precio propuesto el fondo obtiene un {fmt_pct(c.stake)}, más que el {fmt_pct(c.required_stake)} que "
                       f"necesita para su rentabilidad objetivo: el precio deja un margen de {fmt_num((c.stake - c.required_stake) * 100, 1)} "
                       "puntos de participación.")
        else:
            out.append(f"Al precio propuesto el fondo obtiene un {fmt_pct(c.stake)}, menos que el {fmt_pct(c.required_stake)} que "
                       f"necesita: la pre-money que cumple el objetivo es {_m(c, c.vc_pre_money)}"
                       + (f", un {fmt_pct(1 - c.vc_pre_money / c.pre_money, 0)} menos que la propuesta." if _ok(c.pre_money)
                          and c.pre_money > 0 and c.vc_pre_money > 0 else "."))
    if _ok(c.exit_value, c.exit_multiple, c.revenue) and c.exit_basis == "EV/Sales" and c.exit_multiple > 0 and c.revenue > 0:
        exit_rev = c.exit_value / c.exit_multiple
        out.append(f"La salida supone ingresos de {_m(c, exit_rev)} en el año {c.exit_year}, {fmt_num(exit_rev / c.revenue, 1)} "
                   "veces los actuales.")
    if _ok(c.exit_multiple, c.small_cap_multiple) and c.exit_basis == "EV/Sales" and c.small_cap_multiple > 0:
        ratio = c.exit_multiple / c.small_cap_multiple
        if ratio > 2:
            out.append(f"El múltiplo de salida es {fmt_num(ratio, 1)} veces el de las cotizadas pequeñas: el método VC asume una "
                       "salida al precio de una empresa grande.")
    return out


# ---------------------------------------------------------------- retorno, escenarios y caja


def read_montecarlo(c: MemoContext) -> list[str]:
    out = []
    pct = c.mc_moic_pct or {}
    p10, p50, p90 = (pct.get(k, math.nan) for k in ("P10", "P50", "P90"))
    if _ok(p10, p50, p90) and p50 > 0:
        out.append(f"Si hay salida, la mitad de los escenarios dan menos de {fmt_mult(p50)} y uno de cada diez más de "
                   f"{fmt_mult(p90)}" + (": la distribución tiene una cola larga, típica de VC, y el resultado medio lo "
                                         "explican pocos casos muy buenos." if p90 / p50 > 2.5 else "."))
    if _ok(c.mc_prob_loss):
        out.append(f"La probabilidad de recuperar menos de lo invertido es del {fmt_pct(c.mc_prob_loss, 0)}, incluidos los fracasos.")
    if _ok(c.mc_moic_mean, c.moic) and c.moic > 0:
        out.append(f"El MOIC medio de la simulación ({fmt_mult(c.mc_moic_mean)}) frente al del caso base si hay salida "
                   f"({fmt_mult(c.moic)}) mide cuánto pesa el riesgo de fracaso en el retorno.")
    return out


def read_scenarios(c: MemoContext) -> list[str]:
    out = []
    df = c.scenarios_raw
    if df is None or df.empty or not _ok(c.pre_money):
        return out
    methods = [m for m in ("DCF", "Método VC (pre-money)", "Múltiplos (EV/Sales)") if m in df]
    if "DCF" in df and len(df) >= 2:
        lo, hi = float(df["DCF"].min()), float(df["DCF"].max())
        out.append(f"El DCF va de {_m(c, lo)} a {_m(c, hi)} entre escenarios: "
                   + ("una horquilla muy amplia, el valor depende de qué escenario se crea." if lo > 0 and hi / lo > 3 else
                      "el pesimista llega a cero, así que el valor intrínseco depende del escenario favorable." if lo <= 0 else
                      "una horquilla moderada."))
    hits = [(r["Escenario"], m) for _, r in df.iterrows() for m in methods if _ok(r[m]) and r[m] >= c.pre_money]
    total = len(df) * len(methods)
    if total:
        worst = df.loc[df["DCF"].idxmin(), "Escenario"] if "DCF" in df else None
        out.append(f"{len(hits)} de {total} combinaciones de escenario y método sostienen la pre-money propuesta"
                   + (f"; en el escenario {str(worst).lower()} ningún método lo hace." if worst is not None and not any(
                       e == worst for e, _ in hits) else "."))
    return out


def read_cash(c: MemoContext) -> list[str]:
    out = []
    cp = c.cash_projection
    lead = "Con esta ronda," if c.cash_includes_round else "Sin contar esta ronda,"
    if cp is not None and not cp.empty:
        neg = cp[cp["Caja"] < 0]
        be = cp[cp["Consumo de caja"] <= 0]
        if len(neg):
            out.append(f"{lead} la caja se agota en el mes {int(neg['Mes'].iloc[0])}"
                       + (f", antes de llegar al punto de equilibrio (mes {int(be['Mes'].iloc[0])})." if len(be) and
                          be["Mes"].iloc[0] > neg["Mes"].iloc[0] else ".")
                       + " La próxima ronda debería cerrarse al menos seis meses antes.")
        elif len(be):
            out.append(f"{lead} la empresa llega al punto de equilibrio de caja en el mes {int(be['Mes'].iloc[0])} sin "
                       f"quedarse sin caja en los {int(cp['Mes'].iloc[-1])} meses proyectados, con el crecimiento y el "
                       "consumo actuales.")
        else:
            out.append(f"{lead} la caja dura los {int(cp['Mes'].iloc[-1])} meses proyectados, aunque la empresa aún no "
                       "alcanza el equilibrio.")
    if _ok(c.funding_gap, c.implied_dilution) and c.funding_gap > 0:
        out.append(f"El plan del DCF, más ambicioso (crece más y reinvierte para crecer), necesita {_m(c, c.funding_gap)} más "
                   f"en 10 años; levantados hoy, supondrían una dilución adicional del {fmt_pct(c.implied_dilution)}.")
    return out
