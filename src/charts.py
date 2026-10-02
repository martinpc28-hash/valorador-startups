"""Gráficos plotly y formato de números en español.

Paleta: slots categóricos validados (azul, naranja, aqua), rampa secuencial azul y par
divergente azul ↔ rojo. Un solo eje por gráfico; marcas finas; rejilla discreta.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK_MUTED = "#898781"
INK_SECONDARY = "#52514e"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"
SEQ_BLUE = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
ORDINAL_BLUE = ["#86b6ef", "#2a78d6", "#104281"]  # pesimista, base, optimista
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# "USD" en vez de "US$": en Markdown de Streamlit el "$" abre una fórmula.
CURRENCY_SYMBOL = {"USD": "USD", "EUR": "€"}


# ---------------------------------------------------------------- formato


def _es(text: str) -> str:
    """1,234.5 -> 1.234,5"""
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_num(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—" if not (isinstance(x, float) and math.isinf(x)) else "∞"
    return _es(f"{x:,.{decimals}f}")


def fmt_money(x: float, currency: str = "USD", decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    sym = CURRENCY_SYMBOL.get(currency, currency)
    a = abs(x)
    if a >= 1e9:
        body = f"{fmt_num(x / 1e9, decimals)} mil M"
    elif a >= 1e6:
        body = f"{fmt_num(x / 1e6, decimals)} M"
    elif a >= 1e3:
        body = f"{fmt_num(x / 1e3, 1)} k"
    else:
        body = fmt_num(x, 0)
    return f"{sym} {body}"


def fmt_pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{fmt_num(x * 100, decimals)} %"


def fmt_mult(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{fmt_num(x, decimals)}x"


def money_scale(values) -> tuple[float, str]:
    m = np.nanmax(np.abs(np.asarray(values, dtype=float))) if len(values) else 0
    if m >= 1e9:
        return 1e9, "mil M"
    if m >= 1e6:
        return 1e6, "M"
    if m >= 1e3:
        return 1e3, "k"
    return 1.0, ""


# ---------------------------------------------------------------- base


def _layout(fig: go.Figure, title: str = "", height: int = 360, **kw) -> go.Figure:
    kw.setdefault("bargap", 0.35)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        font=dict(family=FONT, size=12),
        hoverlabel=dict(font_family=FONT),
        separators=",.",
        **kw,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="")
    fig.update_yaxes(gridwidth=1, zeroline=True, zerolinewidth=1, ticks="")
    return fig


# ---------------------------------------------------------------- resumen


def football_field(rows: list[dict], pre_money: float, currency: str) -> go.Figure:
    """Rango (bajo–alto) y punto central de cada método frente a la pre-money propuesta."""
    scale, unit = money_scale([v for r in rows for v in (r["low"], r["high"], r["mid"])] + [pre_money])
    sym = CURRENCY_SYMBOL[currency]
    names = [r["method"] for r in rows][::-1]
    fig = go.Figure()
    for r in rows[::-1]:
        lo, hi = sorted((r["low"], r["high"]))
        fig.add_trace(go.Scatter(
            x=[lo / scale, hi / scale], y=[r["method"]] * 2, mode="lines",
            line=dict(color="#86b6ef", width=14), showlegend=False,
            hovertemplate=f"{r['method']}<br>{r['range_label']}: %{{x:,.2f}} {unit} {sym}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[r["mid"] / scale for r in rows[::-1]], y=names, mode="markers",
        marker=dict(color=BLUE, size=12, line=dict(color="white", width=2)),
        name="Valor central",
        hovertemplate="%{y}<br>Central: %{x:,.2f} " + f"{unit} {sym}<extra></extra>",
    ))
    fig.add_vline(x=pre_money / scale, line=dict(color=ORANGE, width=2))
    fig.add_annotation(
        x=pre_money / scale, y=1.02, yref="paper", text=f"Pre-money propuesta: {fmt_money(pre_money, currency)}",
        showarrow=False, font=dict(color=INK_SECONDARY), xanchor="left", yanchor="bottom",
    )
    _layout(fig, height=130 + 56 * len(rows), showlegend=False)
    fig.update_layout(margin=dict(t=40))
    fig.update_xaxes(title=f"{unit} {sym}", showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# ---------------------------------------------------------------- DCF


def bars(x, y, title: str, ytitle: str, color: str = BLUE, signed: bool = False, fmt: str = ",.2f") -> go.Figure:
    y = np.asarray(y, dtype=float)
    colors = [BLUE if v >= 0 else RED for v in y] if signed else color
    fig = go.Figure(go.Bar(
        x=x, y=y, marker=dict(color=colors, cornerradius=4),
        hovertemplate="Año %{x}<br>%{y:" + fmt + "}<extra></extra>",
    ))
    _layout(fig, title, height=300, showlegend=False)
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(title="Año", dtick=1)
    return fig


def line(x, y, title: str, ytitle: str, pct: bool = False, ref: float | None = None, ref_label: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=x, y=np.asarray(y) * (100 if pct else 1), mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
        hovertemplate="Año %{x}<br>%{y:,.1f}" + (" %" if pct else "") + "<extra></extra>",
    ))
    if ref is not None:
        fig.add_hline(y=ref * (100 if pct else 1), line=dict(color=INK_MUTED, width=1))
        fig.add_annotation(x=1, xref="paper", y=ref * (100 if pct else 1), text=ref_label,
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=INK_SECONDARY))
    _layout(fig, title, height=300, showlegend=False)
    fig.update_yaxes(title=ytitle, ticksuffix=" %" if pct else "")
    fig.update_xaxes(title="Año", dtick=1)
    return fig


def heatmap(z: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, xtitle: str, ytitle: str, unit: str) -> go.Figure:
    text = [[fmt_num(v, 1) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, colorscale=SEQ_BLUE, text=text, texttemplate="%{text}",
        xgap=2, ygap=2, colorbar=dict(title=unit, thickness=10),
        hovertemplate=f"{xtitle}: %{{x}}<br>{ytitle}: %{{y}}<br>Valor: %{{text}} {unit}<extra></extra>",
    ))
    _layout(fig, title, height=380)
    fig.update_xaxes(title=xtitle, type="category")
    fig.update_yaxes(title=ytitle, type="category", showgrid=False, zeroline=False)
    return fig


def tornado(df: pd.DataFrame, base: float, title: str, unit: str) -> go.Figure:
    """df: variable, low, high (valor del resultado con la variable en su extremo bajo/alto)."""
    df = df.assign(span=(df["high"] - df["low"]).abs()).sort_values("span")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["variable"], x=df["low"] - base, base=base, orientation="h", name="Variable baja",
        marker=dict(color=RED, cornerradius=4), customdata=df["low"],
        hovertemplate="%{y} baja<br>Valor: %{customdata:,.2f} " + unit + "<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["variable"], x=df["high"] - base, base=base, orientation="h", name="Variable alta",
        marker=dict(color=BLUE, cornerradius=4), customdata=df["high"],
        hovertemplate="%{y} alta<br>Valor: %{customdata:,.2f} " + unit + "<extra></extra>",
    ))
    fig.add_vline(x=base, line=dict(color=INK_SECONDARY, width=1))
    _layout(fig, title, height=110 + 48 * len(df), barmode="overlay",
            legend=dict(orientation="h", y=-0.15, x=0))
    fig.update_xaxes(title=unit, showgrid=True)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# ---------------------------------------------------------------- escenarios y Monte Carlo


def scenario_bars(df: pd.DataFrame, currency: str) -> go.Figure:
    """df: Escenario, Método, Valor."""
    scale, unit = money_scale(df["Valor"].values)
    fig = go.Figure()
    for color, (name, grp) in zip(ORDINAL_BLUE, df.groupby("Escenario", sort=False)):
        fig.add_trace(go.Bar(
            x=grp["Método"], y=grp["Valor"] / scale, name=name, marker=dict(color=color, cornerradius=4),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:,.2f}} {unit} {CURRENCY_SYMBOL[currency]}<extra></extra>",
        ))
    _layout(fig, "Valor por escenario y método", height=360, barmode="group", bargroupgap=0.08,
            legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(title=f"{unit} {CURRENCY_SYMBOL[currency]}")
    return fig


def histogram(values: np.ndarray, title: str, xtitle: str, markers: dict[str, float], target: tuple[str, float] | None = None, scale: float = 1.0) -> go.Figure:
    v = values / scale
    lo, hi = np.percentile(v, [0.5, 99])
    fig = go.Figure(go.Histogram(
        x=np.clip(v, lo, hi), nbinsx=60, marker=dict(color=BLUE, line=dict(color="white", width=1)),
        hovertemplate=xtitle + ": %{x}<br>Simulaciones: %{y}<extra></extra>",
    ))
    for i, (name, x) in enumerate(markers.items()):
        fig.add_vline(x=x / scale, line=dict(color=INK_SECONDARY, width=1))
        fig.add_annotation(x=x / scale, y=0.97 - 0.09 * i, yref="paper", text=f"{name}: {fmt_num(x / scale, 2)}",
                           showarrow=False, xanchor="left", xshift=3, font=dict(color=INK_SECONDARY, size=11),
                           bgcolor="rgba(255,255,255,0.7)")
    if target and lo <= target[1] / scale <= hi:
        fig.add_vline(x=target[1] / scale, line=dict(color=ORANGE, width=2))
        fig.add_annotation(x=target[1] / scale, y=1.0, yref="paper", text=target[0], showarrow=False,
                           xanchor="right", yanchor="bottom", xshift=-3, font=dict(color=INK_SECONDARY, size=11))
    _layout(fig, title, height=340, showlegend=False, bargap=0.02)
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title="Simulaciones")
    return fig


def cash_chart(df: pd.DataFrame, currency: str) -> go.Figure:
    scale, unit = money_scale(df["Caja"].values)
    fig = go.Figure(go.Scatter(
        x=df["Mes"], y=df["Caja"] / scale, mode="lines", line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.12)",
        hovertemplate="Mes %{x}<br>Caja: %{y:,.2f} " + f"{unit} {CURRENCY_SYMBOL[currency]}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=RED, width=1))
    _layout(fig, "Caja proyectada (mensual)", height=320, showlegend=False)
    fig.update_xaxes(title="Mes")
    fig.update_yaxes(title=f"{unit} {CURRENCY_SYMBOL[currency]}")
    return fig


def jcurve_chart(df: pd.DataFrame, currency: str) -> go.Figure:
    scale, unit = money_scale(df[["Flujo neto acumulado", "Valor total (con NAV)"]].values.ravel())
    fig = go.Figure()
    for col, color in (("Flujo neto acumulado", BLUE), ("Valor total (con NAV)", ORANGE)):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col] / scale, mode="lines+markers", name=col,
            line=dict(color=color, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
            hovertemplate=col + "<br>%{x|%Y-%m-%d}: %{y:,.2f} " + f"{unit}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=INK_MUTED, width=1))
    _layout(fig, "Curva J", height=340, legend=dict(orientation="h", y=1.1, x=0))
    fig.update_yaxes(title=f"{unit} {CURRENCY_SYMBOL[currency]}")
    return fig
