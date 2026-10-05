"""Gráficos estáticos (PNG) para el memo de inversión, con matplotlib.

Misma paleta que la app (azul de acento, naranja para la referencia, rojo para negativos), un eje por
gráfico, rejilla discreta y sin decoración. Funciona en el servidor sin navegador.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo genera imágenes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"
RED = "#e34948"
INK = "#111827"
INK_2 = "#52514e"
GRID = "#e5e7eb"

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
        ax.plot([lo, hi], [i, i], color=BLUE_LIGHT, linewidth=9, solid_capstyle="round", zorder=1)
        ax.scatter([r["mid"] / sc], [i], color=BLUE, s=55, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(_num(r["mid"] / sc), (r["mid"] / sc, i), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.axvline(pre_money / sc, color=ORANGE, linewidth=1.8, zorder=2)
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
    axes[0].bar(years, np.asarray(revenue) / sc, color=BLUE, width=0.6)
    axes[0].set_title("Ingresos")
    axes[0].set_ylabel(f"{unit} {sym}")
    axes[1].plot(years, np.asarray(margin) * 100, color=BLUE, marker="o", markersize=3.5, linewidth=1.6)
    axes[1].axhline(0, color=GRID, linewidth=0.8)
    axes[1].set_title("Margen operativo")
    axes[1].set_ylabel("%")
    sc2, unit2 = _scale(fcff)
    axes[2].bar(years, np.asarray(fcff) / sc2, color=[BLUE if v >= 0 else RED for v in fcff], width=0.6)
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
    ax.hist(np.clip(m, 0, hi), bins=50, color=BLUE, edgecolor="white", linewidth=0.5)
    for name, q in zip(("P10", "P50", "P90"), np.percentile(m, [10, 50, 90]) if len(m) else [0, 0, 0]):
        ax.axvline(q, color=INK_2, linewidth=0.9)
        ax.annotate(f"{name} {_num(q)}x", (q, ax.get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points",
                    fontsize=7.5, color=INK_2)
    if target <= hi:
        ax.axvline(target, color=ORANGE, linewidth=1.6)
    ax.set_title("MOIC si hay salida (Monte Carlo)")
    ax.set_xlabel("MOIC")
    ax.set_ylabel("Simulaciones")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return _png(fig)


def is_png(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and data[:8] == b"\x89PNG\r\n\x1a\n"
