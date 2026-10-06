"""Gráficos estáticos (PNG) para el memo de inversión, con matplotlib.

Misma paleta que la app (verde tinta para los datos, latón para la referencia, granate para negativos), un eje por
gráfico, rejilla discreta y sin decoración. Funciona en el servidor sin navegador.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo genera imágenes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRIMARY = "#1E5641"
PRIMARY_LIGHT = "#A2C6B0"
REFERENCE = "#B7862A"
NEGATIVE = "#9C3535"
SLATE = "#3D5A80"
INK = "#18211D"
INK_2 = "#4B524E"
GRID = "#DDE3DA"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def _scale(values) -> tuple[float, str]:
    m = float(np.nanmax(np.abs(np.asarray(values, dtype=float)))) if len(values) else 0.0
    if m >= 1e9:
        return 1e9, "mil M"
    if m >= 1e6:
        return 1e6, "M"
    if m >= 1e3:
        return 1e3, "k"
    return 1.0, ""


def _num(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def football_field(rows: list[dict], pre_money: float, sym: str) -> bytes:
    """Rango bajo a alto y valor central de cada método frente a la pre-money propuesta."""
    sc, unit = _scale([v for r in rows for v in (r["low"], r["high"], r["mid"])] + [pre_money])
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.0))
    for i, r in enumerate(rows[::-1]):
        lo, hi = sorted((r["low"] / sc, r["high"] / sc))
        ax.plot([lo, hi], [i, i], color=PRIMARY_LIGHT, linewidth=9, solid_capstyle="round", zorder=1)
        ax.scatter([r["mid"] / sc], [i], color=PRIMARY, s=55, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(_num(r["mid"] / sc), (r["mid"] / sc, i), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.axvline(pre_money / sc, color=REFERENCE, linewidth=1.8, zorder=2)
    ax.annotate(f"Pre-money propuesta: {_num(pre_money / sc)} {unit}", (pre_money / sc, len(rows) - 0.4),
                ha="left", fontsize=8, color=INK_2, xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(range(len(rows)), [r["method"] for r in rows[::-1]])
    ax.set_xlabel(f"{unit} {sym}")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_title("Valor por método frente a la pre-money propuesta")
    return _png(fig)


def projection(years, revenue, margin, fcff, sym: str) -> bytes:
    """Ingresos, margen operativo y FCFF proyectados: tres gráficos pequeños, cada uno con su eje."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    sc, unit = _scale(revenue)
    axes[0].bar(years, np.asarray(revenue) / sc, color=PRIMARY, width=0.6)
    axes[0].set_title("Ingresos")
    axes[0].set_ylabel(f"{unit} {sym}")
    axes[1].plot(years, np.asarray(margin) * 100, color=PRIMARY, marker="o", markersize=3.5, linewidth=1.6)
    axes[1].axhline(0, color=GRID, linewidth=0.8)
    axes[1].set_title("Margen operativo")
    axes[1].set_ylabel("%")
    sc2, unit2 = _scale(fcff)
    axes[2].bar(years, np.asarray(fcff) / sc2, color=[PRIMARY if v >= 0 else NEGATIVE for v in fcff], width=0.6)
    axes[2].axhline(0, color=GRID, linewidth=0.8)
    axes[2].set_title("Flujo de caja libre")
    axes[2].set_ylabel(f"{unit2} {sym}")
    for ax in axes:
        ax.set_xticks(list(years)[::2])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return _png(fig)


def moic_distribution(moic: np.ndarray, survived: np.ndarray, target: float) -> bytes:
    """Distribución del MOIC en los escenarios con salida, con P10, P50, P90 y el objetivo."""
    m = moic[survived] if survived.sum() >= 10 else moic
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    hi = float(np.percentile(m, 99)) if len(m) else 1.0
    ax.hist(np.clip(m, 0, hi), bins=50, color=PRIMARY, edgecolor="white", linewidth=0.5)
    for name, q in zip(("P10", "P50", "P90"), np.percentile(m, [10, 50, 90]) if len(m) else [0, 0, 0]):
        ax.axvline(q, color=INK_2, linewidth=0.9)
        ax.annotate(f"{name} {_num(q)}x", (q, ax.get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points",
                    fontsize=7.5, color=INK_2)
    if target <= hi:
        ax.axvline(target, color=REFERENCE, linewidth=1.6)
    ax.set_title("MOIC si hay salida (Monte Carlo)")
    ax.set_xlabel("MOIC")
    ax.set_ylabel("Simulaciones")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return _png(fig)


def dcf_bridge(pv_fcff: float, pv_terminal: float, survival_adj: float, net_cash: float, equity: float, sym: str) -> bytes:
    """Del valor de los flujos al equity: cascada con cada paso del DCF."""
    steps = [("Flujos 10 años", pv_fcff), ("Valor terminal", pv_terminal), ("Ajuste por fracaso", survival_adj),
             ("Caja menos deuda", net_cash)]
    sc, unit = _scale([abs(v) for _, v in steps] + [abs(equity), abs(pv_fcff + pv_terminal)])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    run = 0.0
    for i, (name, v) in enumerate(steps):
        ax.bar(i, v / sc, bottom=run / sc, color=PRIMARY if v >= 0 else NEGATIVE, width=0.6)
        ax.annotate(_num(v / sc), (i, (run + max(v, 0)) / sc), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        run += v
    ax.bar(len(steps), equity / sc, color=REFERENCE, width=0.6)
    ax.annotate(_num(equity / sc), (len(steps), max(equity, 0) / sc), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color=INK_2)
    if abs(run - equity) > 1e-6 * max(1.0, abs(run)):  # suelo de 0 en el equity (responsabilidad limitada)
        ax.annotate(f"suelo en 0 (antes {_num(run / sc)})", (len(steps), 0), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7.5, color=NEGATIVE)
    ax.axhline(0, color=GRID, linewidth=0.8)
    tops = np.cumsum([v for _, v in steps]).tolist() + [equity, 0.0]
    lo, hi = min(tops + [0.0]) / sc, max(tops + [pv_fcff + max(pv_terminal, 0)]) / sc
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.2 * (hi - lo))  # sitio para las etiquetas sobre las barras
    ax.set_xticks(range(len(steps) + 1), [n for n, _ in steps] + ["Equity DCF"], fontsize=8)
    ax.set_ylabel(f"{unit} {sym}")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Del DCF al valor del equity")
    return _png(fig)


def scenarios(names: list[str], series: dict[str, list[float]], pre_money: float, sym: str) -> bytes:
    """Valor por escenario y método (barras agrupadas) frente a la pre-money propuesta."""
    sc, unit = _scale([v for vs in series.values() for v in vs if np.isfinite(v)] + [pre_money])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    colors = [PRIMARY, PRIMARY_LIGHT, SLATE]
    w = 0.8 / max(len(series), 1)
    x = np.arange(len(names))
    for k, (label, vals) in enumerate(series.items()):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, np.nan_to_num(np.asarray(vals, dtype=float)) / sc, width=w,
               color=colors[k % len(colors)], label=label)
    ax.axhline(pre_money / sc, color=REFERENCE, linewidth=1.6, label="Pre-money propuesta")
    ax.set_xticks(x, names)
    ax.set_ylabel(f"{unit} {sym}")
    ax.legend(fontsize=7.5, frameon=False, ncol=len(series) + 1, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Valor por escenario")
    return _png(fig)


def cash_runway(months, cash, sym: str, round_month: int | None = None, breakeven: int | None = None) -> bytes:
    """Caja mes a mes y el momento en que se agota."""
    months, cash = np.asarray(months), np.asarray(cash, dtype=float)
    sc, unit = _scale(cash)
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.plot(months, cash / sc, color=PRIMARY, linewidth=1.8)
    ax.fill_between(months, cash / sc, 0, where=cash < 0, color=NEGATIVE, alpha=0.15, linewidth=0)
    ax.axhline(0, color=INK_2, linewidth=0.9)
    out = np.flatnonzero(cash < 0)
    if len(out):
        m0 = int(months[out[0]])
        ax.axvline(m0, color=NEGATIVE, linewidth=1, linestyle="--")
        ax.annotate(f"Sin caja en el mes {m0}", (m0, ax.get_ylim()[1] * 0.85), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=NEGATIVE)
    if breakeven is not None and not len(out):
        i = int(np.argmin(np.abs(months - breakeven)))
        ax.scatter([months[i]], [cash[i] / sc], color=REFERENCE, s=30, zorder=3)
        ax.annotate(f"Equilibrio de caja: mes {breakeven}", (months[i], cash[i] / sc), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("Meses desde hoy")
    ax.set_ylabel(f"{unit} {sym}")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Caja proyectada" + (" (incluye esta ronda)" if round_month is not None else ""))
    return _png(fig)


def is_png(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and data[:8] == b"\x89PNG\r\n\x1a\n"
