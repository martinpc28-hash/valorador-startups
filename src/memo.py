"""Memo de inversión para un comité de VC: contenido, riesgos automáticos y exportación a PDF, Word y Excel.

El analista escribe la parte cualitativa (recomendación, tesis, descripción, uso de fondos, riesgos y próximos
pasos). La app rellena las cifras, tablas y gráficos con los resultados del modelo. Los tres formatos salen
de la misma estructura (`Memo`), así que dicen exactamente lo mismo.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import os
from dataclasses import dataclass, field

import matplotlib
import numpy as np
import pandas as pd

from src import memo_charts as mc
from src.charts import fmt_money, fmt_mult, fmt_num, fmt_pct

RECOMMENDATIONS = ["Invertir", "Invertir con condiciones", "Seguir analizando", "No invertir"]
DEFAULT_NEXT_STEPS = [
    "Revisar las cuentas anuales depositadas y conciliarlas con las cifras de gestión.",
    "Validar los ingresos recurrentes, la retención de clientes y el coste de adquisición.",
    "Llamadas de referencia con clientes y con inversores anteriores.",
    "Revisar la tabla de capitalización y los derechos de las acciones preferentes.",
]
DISCLAIMER = ("Documento preparado con el Valorador de Startups. Las cifras dependen de los supuestos indicados en el anexo y de "
              "las fuentes citadas. Herramienta educativa y de análisis: no constituye asesoramiento de inversión.")


# ---------------------------------------------------------------- contexto (lo que calcula la app)


@dataclass
class MemoContext:
    # Cualitativo (lo escribe el analista)
    company: str
    analyst: str = ""
    fund: str = ""
    date: str = ""
    recommendation: str = "Seguir analizando"
    thesis: str = ""
    description: str = ""
    use_of_funds: str = ""
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    # Contexto
    currency: str = "USD"
    industry: str = ""
    sector: str = ""
    stage: str = ""
    # Empresa
    revenue: float = math.nan
    growth: float = math.nan
    current_margin: float = math.nan
    target_margin: float = math.nan
    margin_year: int = 7
    burn: float = math.nan
    cash: float = math.nan
    debt: float = 0.0
    runway: float = math.nan
    # Ronda y supuestos de etapa
    investment: float = math.nan
    pre_money: float = math.nan
    post_money: float = math.nan
    stake: float = math.nan
    future_dilution: float = math.nan
    survival_prob: float = math.nan
    target_irr: float = math.nan
    # Veredicto y rango
    verdict_title: str = ""
    lo_mid: float = math.nan
    hi_mid: float = math.nan
    ff_rows: list[dict] = field(default_factory=list)
    # DCF
    dcf_equity: float = math.nan
    dcf_operating: float = math.nan
    dcf_pv_fcff: float = math.nan
    dcf_pv_terminal: float = math.nan
    cost_of_equity: float = math.nan
    cost_of_capital: float = math.nan
    mature_coc: float = math.nan
    beta_used: float = math.nan
    beta_type: str = ""
    stable_growth: float = math.nan
    projection: pd.DataFrame | None = None
    top_sensitivity: str = ""
    # Método VC y retorno
    exit_year: int = 6
    exit_basis: str = "EV/Sales"
    exit_multiple: float = math.nan
    small_cap_multiple: float = math.nan
    exit_value: float = math.nan
    vc_rate: float = math.nan
    vc_mode: str = ""
    vc_pre_money: float = math.nan
    vc_post_money: float = math.nan
    required_stake: float = math.nan
    moic: float = math.nan
    irr: float = math.nan
    expected_moic: float = math.nan
    stake_exit: float = math.nan
    proceeds: float = math.nan
    # Múltiplos, escenarios y Monte Carlo
    multiples: pd.DataFrame | None = None
    illiquidity_discount: float = math.nan
    scenarios: pd.DataFrame | None = None
    mc_prob_fail: float = math.nan
    mc_prob_target: float = math.nan
    mc_prob_loss: float = math.nan
    mc_moic_mean: float = math.nan
    mc_moic_pct: dict = field(default_factory=dict)
    mc_target: float = 3.0
    mc_moic: np.ndarray | None = None
    mc_survived: np.ndarray | None = None
    mc_sims: int = 10_000
    # Caja
    capital_need: float = math.nan
    funding_gap: float = math.nan
    implied_dilution: float = math.nan
    # Sector y comparables
    spain: pd.DataFrame | None = None
    spain_label: str = ""
    comparables: pd.DataFrame | None = None
    # Anexo
    assumptions: pd.DataFrame | None = None
    provenance: pd.DataFrame | None = None
    data_warnings: list[str] = field(default_factory=list)


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


# ---------------------------------------------------------------- riesgos automáticos


def auto_risks(c: MemoContext) -> list[str]:
    """Riesgos detectados en el análisis, redactados para el comité. El analista puede editarlos."""
    m = lambda x: fmt_money(x, c.currency)  # noqa: E731
    out = []
    if _ok(c.runway) and c.runway < 12:
        out.append(f"Runway de {fmt_num(c.runway, 1)} meses sin contar esta ronda: la empresa necesita financiación en menos de un año.")
    if _ok(c.funding_gap) and c.funding_gap > 0:
        out.append(f"El plan consume {m(c.capital_need)} y, tras la caja y esta ronda, faltan {m(c.funding_gap)}: "
                   f"habrá más rondas y más dilución (hasta un {fmt_pct(c.implied_dilution)} adicional si se levantara hoy).")
    if _ok(c.pre_money) and _ok(c.hi_mid) and c.pre_money > c.hi_mid:
        out.append(f"La pre-money propuesta ({m(c.pre_money)}) supera el valor central de todos los métodos "
                   f"({m(c.lo_mid)} a {m(c.hi_mid)}): el precio exige que el plan se cumpla casi por completo.")
    if _ok(c.mc_prob_fail) and c.mc_prob_fail >= 0.5:
        out.append(f"Riesgo de fracaso del {fmt_pct(c.mc_prob_fail, 0)} en la simulación (supuesto de etapa): el retorno depende de "
                   "pocos escenarios muy buenos.")
    if _ok(c.mc_prob_target) and c.mc_prob_target < 0.3:
        out.append(f"Solo un {fmt_pct(c.mc_prob_target, 0)} de probabilidad de alcanzar un MOIC de {fmt_mult(c.mc_target, 1)}.")
    if _ok(c.exit_multiple) and _ok(c.small_cap_multiple) and c.exit_basis == "EV/Sales" and c.exit_multiple > 2 * c.small_cap_multiple:
        out.append(f"El múltiplo de salida ({fmt_mult(c.exit_multiple)} ventas) es más del doble del de las cotizadas más pequeñas "
                   f"({fmt_mult(c.small_cap_multiple)}): la salida supone una valoración de empresa grande.")
    if _ok(c.current_margin) and _ok(c.target_margin) and c.target_margin - c.current_margin > 0.3:
        out.append(f"El plan supone pasar de un margen operativo del {fmt_pct(c.current_margin)} al {fmt_pct(c.target_margin)} "
                   f"en {c.margin_year} años.")
    if _ok(c.growth) and c.growth > 1.0:
        out.append(f"Crecimiento supuesto del {fmt_pct(c.growth, 0)} anual: conviene contrastarlo con el histórico real.")
    if c.top_sensitivity:
        out.append(f"La variable que más mueve el DCF es: {c.top_sensitivity}.")
    if c.data_warnings:
        out.append("Datos de industria con avisos (recortes de valores extremos o reemplazos de fuente): ver el anexo.")
    out.append("Los supuestos por etapa (IRR objetivo, supervivencia y dilución) son ilustrativos, no proceden de una fuente verificada.")
    return out


# ---------------------------------------------------------------- estructura


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]]
    widths: list[float] | None = None  # proporciones; None = iguales
    note: str = ""


@dataclass
class Sub:
    """Subtítulo dentro de una sección."""

    text: str


@dataclass
class Figure:
    png: bytes
    caption: str = ""


@dataclass
class Section:
    title: str
    blocks: list = field(default_factory=list)  # str (párrafo) | Sub | list[str] (viñetas) | Table | Figure


@dataclass
class Memo:
    title: str
    subtitle: str
    meta: dict
    sections: list[Section]


def _kv(pairs: list[tuple[str, str]]) -> Table:
    return Table(["Concepto", "Valor"], [[k, v] for k, v in pairs], widths=[0.45, 0.55])


def _df_table(df: pd.DataFrame, widths=None, note: str = "") -> Table:
    return Table([str(c) for c in df.columns], [[("" if pd.isna(v) else str(v)) for v in r] for r in df.itertuples(index=False)],
                 widths=widths, note=note)


def build_memo(c: MemoContext) -> Memo:
    m = lambda x: fmt_money(x, c.currency) if _ok(x) else "n/d"  # noqa: E731
    p = lambda x, d=1: fmt_pct(x, d) if _ok(x) else "n/d"  # noqa: E731
    x = lambda v: fmt_mult(v) if _ok(v) else "n/d"  # noqa: E731
    sym = "USD" if c.currency == "USD" else "€"
    date = c.date or dt.date.today().strftime("%d/%m/%Y")
    sections = []

    # 1. Resumen ejecutivo
    s = Section("1. Resumen ejecutivo")
    s.blocks.append(f"Recomendación: {c.recommendation}. {c.verdict_title}: la pre-money propuesta es {m(c.pre_money)} y los "
                    f"valores centrales de los métodos de valoración van de {m(c.lo_mid)} a {m(c.hi_mid)}.")
    s.blocks.append(_kv([
        ("Ronda", f"{m(c.investment)} con pre-money de {m(c.pre_money)} (post-money {m(c.post_money)})"),
        ("Participación", p(c.stake)),
        ("Rango de valor (centrales)", f"{m(c.lo_mid)} a {m(c.hi_mid)}"),
        ("MOIC e IRR si hay salida", f"{x(c.moic)} y {p(c.irr)} anual en el año {c.exit_year}"),
        ("MOIC esperado (con supervivencia)", x(c.expected_moic)),
        (f"Probabilidad de MOIC de {fmt_mult(c.mc_target, 1)} o más", p(c.mc_prob_target, 0)),
        ("Runway actual", f"{fmt_num(c.runway, 1)} meses" if _ok(c.runway) else "n/d"),
    ]))
    if c.thesis.strip():
        s.blocks.append(Sub("Tesis de inversión"))
        s.blocks.append(c.thesis.strip())
    sections.append(s)

    # 2. La empresa
    s = Section("2. La empresa")
    if c.description.strip():
        s.blocks.append(c.description.strip())
    s.blocks.append(_kv([
        ("Industria", f"{c.industry} ({c.sector})"), ("Etapa", c.stage),
        ("Ingresos últimos 12 meses", m(c.revenue)), ("Crecimiento anual", p(c.growth)),
        ("Margen operativo actual", p(c.current_margin)), ("Burn rate mensual", m(c.burn)),
        ("Caja", m(c.cash)), ("Deuda financiera", m(c.debt)),
    ]))
    sections.append(s)

    # 3. La ronda
    s = Section("3. La ronda")
    s.blocks.append(_kv([
        ("Inversión", m(c.investment)), ("Pre-money", m(c.pre_money)), ("Post-money", m(c.post_money)),
        ("Participación al entrar", p(c.stake)), ("Dilución futura supuesta hasta la salida", p(c.future_dilution)),
        ("Participación a la salida", p(c.stake_exit)),
    ]))
    if c.use_of_funds.strip():
        s.blocks.append(Sub("Uso de los fondos"))
        s.blocks.append(c.use_of_funds.strip())
    sections.append(s)

    # 4. Valoración
    s = Section("4. Valoración")
    s.blocks.append(Table(["Método", "Bajo", "Central", "Alto", "Rango"],
                          [[r["method"], m(r["low"]), m(r["mid"]), m(r["high"]), r["range_label"]] for r in c.ff_rows],
                          widths=[0.28, 0.18, 0.18, 0.18, 0.18]))
    if c.ff_rows and _ok(c.pre_money):
        s.blocks.append(Figure(mc.football_field(c.ff_rows, c.pre_money, sym),
                               "Rango de valor de cada método; el punto es el valor central y la línea naranja, la pre-money propuesta."))
    s.blocks.append(f"DCF. Equity de {m(c.dcf_equity)}: valor operativo de {m(c.dcf_operating)} (flujos de 10 años {m(c.dcf_pv_fcff)} "
                    f"y valor terminal {m(c.dcf_pv_terminal)}), ajustado por una probabilidad de supervivencia del {p(c.survival_prob, 0)}. "
                    f"Tasa de descuento inicial del {p(c.cost_of_capital)} (beta {c.beta_type} de {fmt_num(c.beta_used, 2) if _ok(c.beta_used) else 'n/d'}, "
                    f"coste del equity {p(c.cost_of_equity)}) que converge al {p(c.mature_coc)}; crecimiento estable del {p(c.stable_growth)} "
                    f"y margen objetivo del {p(c.target_margin)} en el año {c.margin_year}.")
    if c.projection is not None and not c.projection.empty:
        pr = c.projection
        s.blocks.append(Figure(mc.projection(pr["Año"].tolist(), pr["Ingresos"].tolist(), pr["Margen operativo"].tolist(),
                                             pr["FCFF"].tolist(), sym), "Proyección del DCF a 10 años."))
    s.blocks.append(f"Método VC. Salida en el año {c.exit_year} por {m(c.exit_value)} ({c.exit_basis} de {x(c.exit_multiple)}), "
                    f"descontada al {p(c.vc_rate)} ({c.vc_mode}) con una dilución futura del {p(c.future_dilution)}: post-money de "
                    f"{m(c.vc_post_money)} y pre-money de {m(c.vc_pre_money)}. Para lograr la rentabilidad objetivo, el fondo necesitaría "
                    f"hoy un {p(c.required_stake)} de la empresa (se ofrece un {p(c.stake)}).")
    if c.multiples is not None and not c.multiples.empty:
        s.blocks.append(f"Múltiplos comparables, con un descuento por iliquidez del {p(c.illiquidity_discount, 0)}:")
        s.blocks.append(_df_table(c.multiples, widths=[0.46, 0.12, 0.18, 0.24]))
    sections.append(s)

    # 5. Retorno y riesgo
    s = Section("5. Retorno y riesgo")
    s.blocks.append(_kv([
        ("Cobro a la salida", m(c.proceeds)), ("MOIC si hay salida", x(c.moic)), ("IRR anual si hay salida", p(c.irr)),
        ("MOIC esperado (MOIC × supervivencia)", x(c.expected_moic)),
    ]))
    pct = c.mc_moic_pct or {}
    s.blocks.append(Sub("Monte Carlo"))
    s.blocks.append(f"{c.mc_sims:,} simulaciones con semilla fija; crecimiento, margen y múltiplo de salida correlacionados.".replace(",", "."))
    s.blocks.append(_kv([
        ("Probabilidad de fracaso", p(c.mc_prob_fail, 0)),
        (f"Probabilidad de MOIC de {fmt_mult(c.mc_target, 1)} o más", p(c.mc_prob_target, 0)),
        ("Probabilidad de perder dinero (MOIC menor de 1x)", p(c.mc_prob_loss, 0)),
        ("MOIC medio (incluye fracasos)", x(c.mc_moic_mean)),
        ("MOIC si hay salida: P10, P50 y P90", " · ".join(x(pct.get(k, math.nan)) for k in ("P10", "P50", "P90"))),
    ]))
    if c.mc_moic is not None and c.mc_survived is not None and len(c.mc_moic):
        s.blocks.append(Figure(mc.moic_distribution(c.mc_moic, c.mc_survived, c.mc_target),
                               "Solo escenarios con salida; la línea naranja es el MOIC objetivo y la última barra agrupa "
                               "los valores extremos (por encima del percentil 99)."))
    if c.scenarios is not None and not c.scenarios.empty:
        s.blocks.append(Sub("Escenarios"))
        s.blocks.append(_df_table(c.scenarios))
    s.blocks.append(_kv([
        ("Caja que consume el plan", m(c.capital_need)), ("Déficit tras la caja y esta ronda", m(c.funding_gap)),
        ("Dilución adicional implícita", p(c.implied_dilution)),
    ]))
    sections.append(s)

    # 6. Sector y comparables
    if (c.spain is not None and not c.spain.empty) or (c.comparables is not None and not c.comparables.empty):
        s = Section("6. Sector y comparables")
        if c.spain is not None and not c.spain.empty:
            s.blocks.append(f"Empresas españolas del sector ({c.spain_label}), Banco de España, Central de Balances:")
            s.blocks.append(_df_table(c.spain, widths=[0.3, 0.12, 0.12, 0.12, 0.14, 0.2]))
        if c.comparables is not None and not c.comparables.empty:
            s.blocks.append("Comparables seleccionados:")
            s.blocks.append(_df_table(c.comparables))
        sections.append(s)

    # 7. Riesgos y 8. Próximos pasos
    n = len(sections) + 1
    sections.append(Section(f"{n}. Riesgos y mitigantes", [list(c.risks) or ["Sin riesgos indicados."]]))
    sections.append(Section(f"{n + 1}. Próximos pasos", [list(c.next_steps) or ["Sin próximos pasos indicados."]]))

    # Anexo
    s = Section("Anexo: supuestos, fórmulas y fuentes")
    if c.assumptions is not None and not c.assumptions.empty:
        s.blocks.append(_df_table(c.assumptions, widths=[0.42, 0.22, 0.36]))
    s.blocks.append([
        "Participación = inversión / (pre-money + inversión).",
        "Coste del equity = tasa libre de riesgo + beta × prima de riesgo + prima de tamaño + prima de iliquidez.",
        "FCFF = EBIT − impuestos (tras compensar pérdidas) − reinversión; reinversión = aumento de ingresos / (ventas / capital).",
        "Valor terminal = EBIT del año 11 × (1 − t) × (1 − g / ROC) / (tasa madura − g).",
        "Equity DCF = máx(p × valor operativo + (1 − p) × recuperación + caja − deuda, 0).",
        "Post-money (método VC) = valor de salida × (1 − dilución) / (1 + r)^T; pre-money = post-money − inversión.",
        "MOIC = participación × (1 − dilución) × valor de salida / inversión; IRR anual = MOIC^(1 / T) − 1.",
    ])
    if c.provenance is not None and not c.provenance.empty:
        s.blocks.append("Datos de la industria usados y su procedencia:")
        s.blocks.append(_df_table(c.provenance, widths=[0.34, 0.14, 0.14, 0.22, 0.16]))
    if c.data_warnings:
        s.blocks.append("Avisos sobre los datos:")
        s.blocks.append(list(dict.fromkeys(c.data_warnings)))
    s.blocks.append(DISCLAIMER)
    sections.append(s)

    meta = {"Empresa": c.company, "Fecha": date}
    if c.analyst:
        meta["Analista"] = c.analyst
    if c.fund:
        meta["Fondo"] = c.fund
    meta["Recomendación"] = c.recommendation
    return Memo(f"Memo de inversión: {c.company}", f"{c.industry} · {c.stage} · {c.currency}", meta, sections)


# ---------------------------------------------------------------- PDF


def _font_dir() -> str:
    return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def to_pdf(memo: Memo) -> bytes:
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    accent, ink2, line, fill = (42, 120, 214), (82, 81, 78), (229, 231, 235), (243, 244, 246)

    class PDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("DejaVu", "", 7.5)
            self.set_text_color(*ink2)
            self.cell(0, 6, memo.title, align="L")
            self.ln(8)

        def footer(self):
            self.set_y(-12)
            self.set_font("DejaVu", "", 7)
            self.set_text_color(*ink2)
            self.cell(0, 5, f"Página {self.page_no()} de {{nb}} · No constituye asesoramiento de inversión", align="C")

    pdf = PDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    d = _font_dir()
    pdf.add_font("DejaVu", "", os.path.join(d, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(d, "DejaVuSans-Bold.ttf"))
    pdf.alias_nb_pages()
    pdf.add_page()
    w = pdf.w - pdf.l_margin - pdf.r_margin

    # Portada
    pdf.set_fill_color(*accent)
    pdf.rect(0, 0, pdf.w, 3, "F")
    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 19)
    pdf.set_text_color(17, 24, 39)
    pdf.multi_cell(w, 9, memo.title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*ink2)
    pdf.multi_cell(w, 6, memo.subtitle, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 9)
    for k, v in memo.meta.items():
        pdf.set_text_color(*ink2)
        pdf.cell(32, 5.5, k)
        pdf.set_text_color(17, 24, 39)
        pdf.cell(0, 5.5, str(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    def table(t: Table):
        widths = t.widths or [1 / len(t.columns)] * len(t.columns)
        pdf.set_font("DejaVu", "", 8)
        pdf.set_text_color(17, 24, 39)
        pdf.set_draw_color(*line)
        pdf.set_fill_color(255, 255, 255)
        with pdf.table(col_widths=[x * 100 for x in widths], width=w, line_height=4.6, padding=1.4,
                       headings_style=FontFace(emphasis="BOLD", color=(17, 24, 39), fill_color=fill),
                       cell_fill_color=(249, 250, 251), cell_fill_mode="EVEN_ROWS",
                       borders_layout="HORIZONTAL_LINES", text_align="LEFT") as tb:
            row = tb.row()
            for col in t.columns:
                row.cell(col)
            for r in t.rows:
                row = tb.row()
                for v in r:
                    row.cell(v)
        pdf.ln(2)

    for sec in memo.sections:
        if pdf.get_y() > pdf.h - 45:
            pdf.add_page()
        pdf.ln(2)
        pdf.set_font("DejaVu", "B", 12.5)
        pdf.set_text_color(*accent)
        pdf.multi_cell(w, 7, sec.title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*line)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + w, pdf.get_y())
        pdf.ln(2)
        for b in sec.blocks:
            if isinstance(b, Sub):
                pdf.set_font("DejaVu", "B", 9.8)
                pdf.set_text_color(17, 24, 39)
                pdf.multi_cell(w, 5.5, b.text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(0.5)
            elif isinstance(b, str):
                pdf.set_font("DejaVu", "", 9)
                pdf.set_text_color(17, 24, 39)
                pdf.multi_cell(w, 5, b, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1.5)
            elif isinstance(b, list):
                pdf.set_font("DejaVu", "", 9)
                pdf.set_text_color(17, 24, 39)
                for item in b:
                    pdf.cell(4, 5, "•")
                    pdf.multi_cell(w - 4, 5, item, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1.5)
            elif isinstance(b, Table):
                table(b)
            elif isinstance(b, Figure):
                img = io.BytesIO(b.png)
                if pdf.get_y() > pdf.h - 85:
                    pdf.add_page()
                pdf.image(img, w=w)
                if b.caption:
                    pdf.set_font("DejaVu", "", 7.5)
                    pdf.set_text_color(*ink2)
                    pdf.multi_cell(w, 4, b.caption, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
    return bytes(pdf.output())


# ---------------------------------------------------------------- Word


def to_docx(memo: Memo) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    base = doc.styles["Normal"]
    base.font.name = "Calibri"
    base.font.size = Pt(10)
    accent = RGBColor(0x2A, 0x78, 0xD6)

    t = doc.add_heading(memo.title, level=0)
    t.runs[0].font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    sub = doc.add_paragraph(memo.subtitle)
    sub.runs[0].font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
    for k, v in memo.meta.items():
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(0)
        r = para.add_run(f"{k}: ")
        r.bold = True
        para.add_run(str(v))

    def table(tb: Table):
        table = doc.add_table(rows=1, cols=len(tb.columns))
        table.style = "Light List Accent 1"
        for i, col in enumerate(tb.columns):
            table.rows[0].cells[i].text = col
        for r in tb.rows:
            cells = table.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = v
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(9)
        doc.add_paragraph()

    for s in memo.sections:
        h = doc.add_heading(s.title, level=1)
        h.runs[0].font.color.rgb = accent
        for b in s.blocks:
            if isinstance(b, Sub):
                doc.add_heading(b.text, level=2)
            elif isinstance(b, str):
                doc.add_paragraph(b)
            elif isinstance(b, list):
                for item in b:
                    doc.add_paragraph(item, style="List Bullet")
            elif isinstance(b, Table):
                table(b)
            elif isinstance(b, Figure):
                doc.add_picture(io.BytesIO(b.png), width=Cm(17))
                if b.caption:
                    cap = doc.add_paragraph(b.caption)
                    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    cap.runs[0].font.size = Pt(8)
                    cap.runs[0].font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- Excel


def to_xlsx(c: MemoContext, memo: Memo) -> bytes:
    """Cuaderno para rehacer los números: resumen, proyección, métodos, escenarios, supuestos y fuentes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        summary = pd.DataFrame(
            [("Empresa", c.company), ("Fecha", memo.meta.get("Fecha")), ("Recomendación", c.recommendation),
             ("Moneda", c.currency), ("Industria", c.industry), ("Etapa", c.stage), ("Inversión", c.investment),
             ("Pre-money", c.pre_money), ("Post-money", c.post_money), ("Participación", c.stake),
             ("DCF (equity)", c.dcf_equity), ("Método VC (pre-money)", c.vc_pre_money), ("Valor de salida", c.exit_value),
             ("MOIC si hay salida", c.moic), ("IRR anual si hay salida", c.irr), ("MOIC esperado", c.expected_moic),
             ("Prob. de fracaso (Monte Carlo)", c.mc_prob_fail), (f"Prob. de MOIC >= {c.mc_target}", c.mc_prob_target),
             ("Runway (meses)", c.runway)],
            columns=["Concepto", "Valor"])
        summary.to_excel(xl, sheet_name="Resumen", index=False)
        pd.DataFrame([{"Método": r["method"], "Bajo": r["low"], "Central": r["mid"], "Alto": r["high"], "Rango": r["range_label"]}
                      for r in c.ff_rows]).to_excel(xl, sheet_name="Métodos", index=False)
        if c.projection is not None:
            c.projection.to_excel(xl, sheet_name="Proyección DCF", index=False)
        if c.scenarios is not None:
            c.scenarios.to_excel(xl, sheet_name="Escenarios", index=False)
        if c.multiples is not None:
            c.multiples.to_excel(xl, sheet_name="Múltiplos", index=False)
        if c.assumptions is not None:
            c.assumptions.to_excel(xl, sheet_name="Supuestos", index=False)
        if c.provenance is not None:
            c.provenance.to_excel(xl, sheet_name="Fuentes", index=False)
        pd.DataFrame({"Riesgos": c.risks or [""]}).to_excel(xl, sheet_name="Riesgos", index=False)
        for ws in xl.book.worksheets:  # ancho de columnas legible
            for col in ws.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 60)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def memo_text(memo: Memo) -> str:
    """Todo el texto del memo (para pruebas y búsquedas)."""
    parts = [memo.title, memo.subtitle, *map(str, memo.meta.values())]
    for s in memo.sections:
        parts.append(s.title)
        for b in s.blocks:
            if isinstance(b, Sub):
                parts.append(b.text)
            elif isinstance(b, str):
                parts.append(b)
            elif isinstance(b, list):
                parts.extend(b)
            elif isinstance(b, Table):
                parts.extend(b.columns)
                parts.extend(v for r in b.rows for v in r)
            elif isinstance(b, Figure):
                parts.append(b.caption)
    return "\n".join(parts)
