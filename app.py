"""Valorador de startups con múltiples fuentes de datos (Streamlit).

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from src import charts as ch
from src.charts import fmt_money, fmt_mult, fmt_num, fmt_pct
from src.data import (
    MetricResolver,
    industry_table,
    latest_fx,
    latest_series,
    load_crosswalk,
    load_fx,
    load_industry_metrics,
    load_market_metrics,
    load_metric_definitions,
    load_size_metrics,
    load_sources,
    load_stage_assumptions,
    source_order,
)
from src import comparables as cmp
from src import company_store as cs
from src import statements as stm
from src.fund import fund_metrics, j_curve, project_cash_flows
from src.montecarlo import MCSettings, simulate
from src.valuation import (
    DEFAULT_SCENARIOS,
    DCFInputs,
    ScenarioFactors,
    dcf,
    deal_returns,
    discount_rate,
    exit_ebitda_margin,
    implied_dilution,
    monthly_cash_projection,
    multiple_value,
    relever_beta,
    resolve_round,
    runway_months,
    scale_margin,
    scenario_inputs,
    vc_method,
    wacc,
)

st.set_page_config(page_title="Valorador de Startups", page_icon="📈", layout="wide")


# ======================================================================= datos


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    metrics = load_industry_metrics()
    crosswalk = load_crosswalk()
    return {
        "metrics": metrics,
        "defs": load_metric_definitions(),
        "crosswalk": crosswalk,
        "industries": industry_table(crosswalk),
        "sources": load_sources(),
        "stages": load_stage_assumptions(),
        "size": load_size_metrics(),
        "market": load_market_metrics(),
        "fx": load_fx(),
    }


D = load_data()
industries = D["industries"]
sector_of = dict(zip(industries["industry_std"], industries["sector"]))
stages = D["stages"]
source_names = dict(zip(D["sources"]["source_id"], D["sources"]["name"]))

us_rf = latest_series(D["market"], "us_tbond_10y")
ea_rf = latest_series(D["market"], "ea_aaa_10y")
erp_row = latest_series(D["market"], "implied_erp_fcfe")
fx_row = latest_fx(D["fx"])


# Los valores por defecto viven en st.session_state (no en el parámetro `value`), para que la
# precarga desde la pestaña de comparables pueda reescribirlos antes de dibujar los widgets.

def _default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def pct_input(label: str, value: float, key: str, min_value: float = -100.0, max_value: float = 500.0,
              step: float = 0.5, help: str | None = None) -> float:
    """Entrada en porcentaje que devuelve un decimal."""
    _default(key, float(round(value * 100, 2)))
    return st.number_input(label, min_value, max_value, step=step, format="%.2f", key=key, help=help) / 100.0


def money_input(label: str, value: float, key: str, help: str | None = None, min_value: float = 0.0) -> float:
    _default(key, float(value))
    return st.number_input(label, min_value, None, step=50_000.0, format="%.0f", key=key, help=help)


def usd_rate(cur: str) -> float:
    """USD por unidad de la moneda base."""
    return 1.0 if cur == "USD" else float(fx_row["rate"])


def apply_prefill(values: dict, cur: str) -> None:
    """Callback del botón de precarga: escribe en las entradas de la barra lateral (importes en USD)."""
    r = usd_rate(cur)
    if "revenue" in values:
        st.session_state["rev0"] = round(values["revenue"] / r)
    if "growth" in values:
        st.session_state["growth"] = round(values["growth"] * 100, 2)
    if "current_margin" in values:
        st.session_state["cm"] = round(values["current_margin"] * 100, 2)
    if "cash" in values:
        st.session_state["cash"] = round(values["cash"] / r)
    if "investment" in values:
        st.session_state["inv"] = round(values["investment"] / r)
        if st.session_state.get("solve_for") == "Inversión":
            st.session_state["solve_for"] = "Participación"
    if "industry" in values and values["industry"] in set(industries["industry_std"]):
        st.session_state["industry"] = values["industry"]
    st.session_state["prefill_msg"] = values.get("_label", "")


# ----------------------------------------------------------------------- cuentas y biblioteca


@st.cache_resource(show_spinner=False)
def company_store():
    return cs.get_store()


# Biblioteca compartida y sin cuentas (decisión del usuario): todos ven las mismas empresas.
SHARED_OWNER = "compartida"


def current_user() -> str:
    return SHARED_OWNER


def my_companies() -> list[dict]:
    """Biblioteca compartida, cacheada en la sesión hasta que se guarda o borra algo."""
    if "lib" not in st.session_state:
        try:
            st.session_state["lib"] = company_store().list(SHARED_OWNER)
        except Exception as e:  # noqa: BLE001: se muestra al usuario en lugar de romper la app
            st.session_state["lib"] = []
            st.session_state["lib_error"] = str(e)
    return st.session_state["lib"]


def refresh_library() -> None:
    st.session_state.pop("lib", None)


def load_company(company: dict) -> None:
    """Callback: carga una empresa guardada en las entradas del modelo (en su propia moneda)."""
    inp = company.get("inputs", {})
    if company.get("currency") in ("USD", "EUR"):
        st.session_state["currency"] = company["currency"]
    money_keys = {"revenue": "rev0", "burn": "burn", "cash": "cash", "investment": "inv", "pre_money": "pre",
                  "debt": "debt", "nol": "nol"}
    for field, key in money_keys.items():
        # Inversión y pre-money a 0 = sin dato: no se cargan (la ronda exige importes positivos)
        if inp.get(field) is not None and not (field in ("investment", "pre_money") and not inp[field]):
            st.session_state[key] = float(inp[field])
    for field, key in {"growth": "growth", "current_margin": "cm"}.items():
        if inp.get(field) is not None:
            st.session_state[key] = round(float(inp[field]) * 100, 2)
    if inp.get("investment") and inp.get("pre_money"):
        st.session_state["solve_for"] = "Participación"
    if company.get("industry") in set(industries["industry_std"]):
        st.session_state["industry"] = company["industry"]
    if company.get("stage") in set(stages["stage_label"]):
        st.session_state["stage"] = company["stage"]
    st.session_state["prefill_msg"] = company.get("name", "empresa guardada")


# ======================================================================= barra lateral

with st.sidebar:
    st.title("Valorador de Startups")
    st.caption("Rellena los pasos 1 a 3. Todo se recalcula al instante; el veredicto está en la pestaña **Resumen**.")
    if st.session_state.get("prefill_msg"):
        msg = f"Datos precargados de {st.session_state.pop('prefill_msg').rstrip('.')}. Revísalos antes de usarlos."
        st.success(msg)
        st.toast(msg, icon="✅")

    st.subheader("📁 Mis empresas")
    lib = my_companies()
    if lib:
        by_id = {c["id"]: c for c in lib}
        pick = st.selectbox("Cargar empresa guardada", list(by_id), format_func=lambda i: by_id[i]["name"], key="lib_pick",
                            label_visibility="collapsed")
        b1, b2 = st.columns([3, 1])
        b1.button("Cargar en el modelo", on_click=load_company, args=(by_id[pick],), key="lib_load_btn", type="primary",
                  help="Sustituye las entradas de la barra lateral por las de la empresa guardada.", width="stretch")
        b2.button("🔄", on_click=refresh_library, key="lib_refresh_btn", width="stretch",
                  help="Actualizar la lista (por si alguien guardó una empresa desde otra sesión).")
    else:
        st.caption("Aún no hay empresas guardadas. Créalas en la pestaña **Mis empresas** y pulsa **💾 Guardar**.")
        st.button("🔄 Actualizar lista", on_click=refresh_library, key="lib_refresh_btn")
    st.divider()

    st.subheader("1. Contexto")
    _default("currency", "USD")
    currency = st.radio("Moneda base", ["USD", "EUR"], horizontal=True, key="currency",
                        help="Los importes se introducen y se muestran en esta moneda.")
    sym = ch.CURRENCY_SYMBOL[currency]
    options = industries["industry_std"].tolist()
    _default("industry", "Software (System & Application)")
    industry = st.selectbox(
        "Industria", options, key="industry",
        format_func=lambda i: f"{i} · {sector_of[i]}",
        help="Taxonomía propia (industry_std) con equivalencias desde la clasificación de cada fuente.",
    )
    _default("stage", stages["stage_label"].iloc[1])
    stage_label = st.selectbox("Etapa", stages["stage_label"].tolist(), key="stage")

    st.subheader("2. Tu empresa")
    revenue0 = money_input(f"Ingresos últimos 12 meses ({sym})", 1_500_000, "rev0")
    growth = pct_input("Crecimiento anual de ingresos (%)", 0.80, "growth",
                       help="Se mantiene durante los años de alto crecimiento y luego converge a la tasa estable.")
    current_margin = pct_input("Margen operativo actual (%)", -0.60, "cm", min_value=-1000.0, max_value=90.0,
                               help="Negativo si la empresa pierde dinero, lo normal en una startup.")
    burn = money_input(f"Burn rate mensual ({sym})", 150_000, "burn", help="Caja que consume la empresa cada mes.")
    cash = money_input(f"Caja disponible ({sym})", 1_200_000, "cash")

    st.subheader("3. La ronda")
    solve_for = st.radio("Calcular", ["Participación", "Pre-money", "Inversión"], horizontal=True, key="solve_for",
                         help="Participación = inversión / (pre-money + inversión). Introduce dos y se calcula la tercera.")
    inv_in = money_input(f"Inversión ({sym})", 3_000_000, "inv") if solve_for != "Inversión" else None
    pre_in = money_input(f"Pre-money propuesta ({sym})", 12_000_000, "pre") if solve_for != "Pre-money" else None
    stake_in = pct_input("Participación buscada (%)", 0.20, "stake", 0.1, 99.0) if solve_for != "Participación" else None
    try:
        terms = resolve_round(inv_in, pre_in, stake_in)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.caption(
        f"Inversión {fmt_money(terms.investment, currency)} · pre-money {fmt_money(terms.pre_money, currency)} · "
        f"post-money {fmt_money(terms.post_money, currency)} · participación {fmt_pct(terms.stake)}"
    )

    st.subheader("Ajustes avanzados")
    st.caption("Opcionales: los valores por defecto salen de los datos de la industria y de la etapa.")
    with st.expander("Proyección y salida"):
        hg_years = st.slider("Años de alto crecimiento", 1, 9, 5)
        exit_year = st.slider("Año de salida", 2, 10, 6)
        exit_basis = st.radio("Múltiplo de salida", ["EV/Sales", "EV/EBITDA"], horizontal=True)

    with st.expander("Tasas, primas y beta"):
        beta_type = st.radio("Beta", ["Total", "De mercado"], horizontal=True,
                             help="Beta total = beta de mercado / correlación. Supone un inversor no diversificado (fundador, VC concentrado).")
        rf_default = (us_rf["value"] if currency == "USD" else ea_rf["value"])
        rf = pct_input("Tasa libre de riesgo (%)", rf_default, f"rf_{currency}", 0.0, 20.0, 0.05,
                       help=("Bono del Tesoro a 10 años según Damodaran (histimpl)" if currency == "USD"
                             else "Curva AAA de la zona euro a 10 años (BCE)"))
        erp = pct_input("Prima de riesgo del mercado (%)", erp_row["value"], "erp", 0.0, 20.0, 0.05,
                        help=f"Prima implícita de EE. UU. de Damodaran al {erp_row['date']}.")
        stage_row0 = stages[stages["stage_label"] == stage_label].iloc[0]
        size_prem = pct_input("Prima por tamaño (%)", 0.0, "size", 0.0, 20.0, 0.25)
        illiq_prem = pct_input("Prima por iliquidez (%)", stage_row0["illiquidity_premium"], f"illiq_{stage_label}", 0.0, 20.0, 0.25,
                               help="Por defecto, el supuesto de la etapa.")
    with st.expander("Estructura de capital e impuestos"):
        de_ratio = pct_input("D/E de la startup (%)", 0.0, "de", 0.0, 500.0, 5.0)
        kd = pct_input("Costo de la deuda antes de impuestos (%)", 0.08, "kd", 0.0, 40.0, 0.25)
        tax = pct_input("Tasa marginal de impuestos (%)", 0.25, "tax", 0.0, 60.0, 0.5)
        nol0 = money_input(f"Pérdidas fiscales acumuladas ({sym})", 0, "nol")
        debt = money_input(f"Deuda financiera ({sym})", 0, "debt")
    with st.expander("Método VC"):
        vc_mode_label = st.radio(
            "Tratamiento del riesgo de fracaso",
            ["IRR objetivo (ya incluye el fracaso)", "Costo del equity × supervivencia"],
            help="Nunca se aplican las dos cosas a la vez: sería contar el fracaso dos veces.",
        )
    vc_mode = "irr" if vc_mode_label.startswith("IRR") else "survival"
    with st.expander("Fuentes de datos"):
        available = source_order(D["metrics"])
        priority = st.multiselect(
            "Orden de prioridad", available, default=available,
            format_func=lambda s: source_names.get(s, s),
            help="La primera es la fuente por defecto. Si le falta una métrica se usa la siguiente y se avisa. "
                 "Pon el Banco de España primero para usar ratios de empresas españolas donde existan.",
        )
        apply_caps = st.checkbox("Recortar valores extremos (topes en metric_definitions.csv)", True)


# ======================================================================= pestañas (controles primero)

st.title("Valorador de Startups")
st.caption(f"{industry} · {sector_of[industry]} · etapa {stage_label} · moneda {currency}")
warn_box = st.container()

tab_names = ["Read Me", "Resumen", "DCF", "Método VC", "Múltiplos", "Escenarios", "Monte Carlo", "Caja y ronda",
             "Comparables", "Ratios España", "Mis empresas", "Fondos", "Supuestos", "Datos y fuentes"]
TAB_ICONS = {"Read Me": "📖", "Resumen": "🎯", "DCF": "📈", "Método VC": "🚀", "Múltiplos": "✖️", "Escenarios": "🔀",
             "Monte Carlo": "🎲", "Caja y ronda": "💧", "Comparables": "🔎", "Ratios España": "📊", "Mis empresas": "📁",
             "Fondos": "🏦", "Supuestos": "⚙️", "Datos y fuentes": "🗂️"}
T = dict(zip(tab_names, st.tabs([f"{TAB_ICONS.get(n, '')} {n}".strip() for n in tab_names])))

# ======================================================================= Read Me


def _sec_range(dataset: str) -> str:
    try:
        d = pd.read_csv(cmp.DATA / {"sec_form_d": "sec_form_d.csv.gz", "sec_form_c": "sec_form_c.csv.gz",
                                    "sec_s1": "sec_s1.csv.gz"}[dataset], usecols=["filing_date"])
        return f"{d['filing_date'].min()} a {d['filing_date'].max()} ({len(d):,} empresas)".replace(",", ".")
    except Exception:  # noqa: BLE001: el Read Me no debe romperse si falta un archivo
        return "n/d"


with T["Read Me"]:
    damo_date = D["metrics"][D["metrics"]["source"] == "damodaran"]["as_of"].iloc[0]
    st.header("Cómo funciona el Valorador de Startups")
    st.markdown(
        "Esta aplicación estima cuánto vale una startup a partir de los datos que introduces en la barra lateral. "
        "Combina cuatro métodos de valoración y los compara con la pre-money que se propone en la ronda. "
        "Todos los supuestos que vienen de fuentes externas muestran su procedencia: pasa el cursor por el icono "
        "**?** de cada número o abre la pestaña **Datos y fuentes**."
    )
    st.warning("Es una herramienta educativa y de análisis. No es asesoramiento de inversión.", icon="⚠️")

    st.subheader("Qué hace este servidor")
    st.markdown(
        "- La app está escrita en Python con Streamlit y se ejecuta en **Google Cloud Run**, en la región de Madrid "
        "(europe-southwest1). Cuando nadie la usa se apaga sola, así que no genera coste en reposo.\n"
        "- Los datos de mercado e industria están guardados como archivos CSV dentro del propio proyecto. "
        "La app **no descarga nada mientras la usas**: funciona igual sin conexión a internet.\n"
        "- Las empresas que guardas en **Mis empresas** se almacenan en **Firestore**, la base de datos de Google "
        "Cloud del mismo proyecto. Por ahora la biblioteca es compartida: cualquiera con el enlace puede verla.\n"
        "- De los estados financieros que subes solo se guardan las cifras extraídas, nunca el archivo.\n"
        "- El código está en GitHub. Cada cambio en la rama principal pasa las pruebas automáticas y, si todas "
        "pasan, se publica solo una nueva versión."
    )

    st.subheader("Cómo empezar")
    st.markdown(
        "1. En la barra lateral elige la **industria**, la **etapa** y la **moneda**.\n"
        "2. Introduce los datos de la empresa: ingresos, crecimiento, margen, burn rate y caja.\n"
        "3. Introduce la ronda: dos de estos tres datos (inversión, pre-money o participación) y la app calcula el tercero.\n"
        "4. Mira el veredicto en **Resumen** y entra en cada pestaña para ver el detalle.\n"
        "5. Si quieres, guarda la empresa en **Mis empresas** o compárala con empresas reales en **Comparables SEC**."
    )

    st.subheader("Las pestañas")
    tabs_doc = [
        ("Resumen", "Muestra el rango de valor de cada método (DCF, método VC, múltiplos y Monte Carlo) frente a la "
         "pre-money propuesta, y dice si la propuesta queda por debajo, dentro o por encima de ese rango."),
        ("DCF", "Descuenta los flujos de caja futuros. Toma la beta de la industria, la ajusta a la deuda de la "
         "startup y calcula el costo de capital. Proyecta 10 años: los ingresos crecen y convergen a una tasa estable, "
         "el margen converge al de la industria y la reinversión sale del ratio ventas / capital. Tiene en cuenta las "
         "pérdidas fiscales acumuladas y la probabilidad de que la empresa sobreviva. Incluye un mapa de sensibilidad "
         "y un gráfico de las variables que más mueven el valor."),
        ("Método VC", "Calcula el valor de salida (ingresos o EBITDA del año de salida por un múltiplo) y lo trae a hoy "
         "con la rentabilidad objetivo del inversor y la dilución futura. Da la pre-money que justifica, la participación "
         "necesaria y el MOIC e IRR con las condiciones propuestas. El riesgo de fracaso se cuenta una sola vez: o con "
         "una IRR alta, o con el costo del equity multiplicado por la probabilidad de supervivencia."),
        ("Múltiplos", "Valora la empresa con EV/Sales y EV/EBITDA de su industria y de las cotizadas más pequeñas, "
         "con un descuento por iliquidez. Compara las empresas rentables con el conjunto, que incluye las que pierden dinero."),
        ("Escenarios", "Repite la valoración en un caso pesimista, uno base y uno optimista. Los factores sobre "
         "crecimiento, margen y múltiplo son editables."),
        ("Monte Carlo", "Hace 10.000 simulaciones con semilla fija, así que el resultado siempre es el mismo. "
         "Crecimiento, margen y múltiplo se mueven a la vez y de forma correlacionada, y también se simula el fracaso. "
         "Muestra los percentiles P10, P50 y P90 y la probabilidad de alcanzar el MOIC objetivo."),
        ("Caja y ronda", "Calcula cuántos meses de caja quedan (runway), proyecta la caja mes a mes, estima el "
         "capital que necesita el plan y la dilución adicional que implicaría."),
        ("Comparables", "Buscador de empresas reales. De EE. UU., las que presentaron documentos ante la SEC: rondas "
         "privadas (Form D), startups pequeñas con estados financieros (Form C) y salidas a bolsa (S-1). De España, las "
         "sociedades con constituciones o ampliaciones de capital publicadas en el BORME, que se buscan por nombre u objeto "
         "social. Selecciona una o varias para compararlas con tu startup y ver en qué percentil queda tu ronda o tus "
         "ingresos. Un botón carga los datos de la empresa elegida en el modelo."),
        ("Ratios España", "Compara tu empresa con las empresas españolas de su sector (CNAE) y tamaño, con los datos de "
         "la Central de Balances del Banco de España: crecimiento de ventas, margen EBITDA, rentabilidad, endeudamiento, "
         "coste de la deuda, periodos de cobro y pago y productividad. Muestra el cuartil inferior, la mediana y el cuartil "
         "superior, dónde queda tu empresa y la evolución de 2020 a 2024."),
        ("Mis empresas", "Biblioteca de empresas guardadas. Puedes crear una a mano o subir sus estados financieros "
         "en Excel, CSV o PDF: la app reconoce las partidas en español o inglés y la escala (miles o millones) y rellena "
         "los campos. Revisa lo detectado y pulsa **💾 Guardar** al final de la pestaña. Una vez guardada, la empresa "
         "aparece arriba en la barra lateral, en **📁 Mis empresas**, donde eliges **Cargar en el modelo**."),
        ("Fondos", "Analiza un fondo de VC desde el punto de vista del inversor: DPI, RVPI, TVPI, MOIC e IRR, curva J "
         "y una proyección simple de flujos. Compara el tamaño de tu fondo con los vehículos de VC que presentaron Form D."),
        ("Supuestos", "Tabla editable con los supuestos por etapa: IRR objetivo, probabilidad de supervivencia, "
         "dilución futura, prima de iliquidez y descuento por iliquidez. Son supuestos propios e ilustrativos."),
        ("Datos y fuentes", "Lista cada número de la industria que se está usando, con su fuente, fecha, URL y avisos "
         "(si se recortó un valor extremo o se usó un dato de reemplazo). También muestra el perfil completo de la "
         "industria y el registro de fuentes."),
    ]
    for name, text in tabs_doc:
        st.markdown(f"**{name}.** {text}")

    st.subheader("La barra lateral")
    st.markdown(
        "Ahí están todas las entradas del modelo, en tres pasos: **1. Contexto** (moneda, industria y etapa), "
        "**2. Tu empresa** y **3. La ronda**. Cualquier cambio recalcula todas las pestañas al instante. Arriba, en "
        "**📁 Mis empresas**, cargas una empresa guardada. Los **ajustes avanzados** son opcionales y vienen cerrados: "
        "proyección y salida, tasas y primas, estructura de capital, método VC, fuentes de datos (por ejemplo, poner el "
        "Banco de España primero) y los supuestos anclados en la industria (margen objetivo, ventas / capital y múltiplo "
        "de salida). Los valores marcados con **?** explican de dónde salen."
    )

    st.subheader("Fuentes de datos")
    fx_date = fx_row["date"] if fx_row is not None else "n/d"
    sources_doc = pd.DataFrame([
        {"Fuente": "Aswath Damodaran (NYU Stern)", "Qué aporta": "Betas, costo de capital, márgenes, múltiplos, crecimiento "
         "y reinversión de 94 industrias de EE. UU.; riesgo y múltiplos por tamaño; prima de riesgo implícita",
         "Fecha de los datos": damo_date, "Condiciones": "Uso libre, solo agregados por industria",
         "Enlace": "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html"},
        {"Fuente": "Banco Central Europeo", "Qué aporta": "Tipo de cambio EUR/USD y tasa libre de riesgo en euros "
         "(curva AAA a 10 años)", "Fecha de los datos": fx_date, "Condiciones": "Uso libre citando al BCE",
         "Enlace": "https://data.ecb.europa.eu/"},
        {"Fuente": "SEC EDGAR, Form D", "Qué aporta": "Rondas privadas: importe, inversores, industria y rango de ingresos",
         "Fecha de los datos": _sec_range("sec_form_d"), "Condiciones": "Información pública, redistribuible",
         "Enlace": "https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets"},
        {"Fuente": "SEC EDGAR, Form C", "Qué aporta": "Estados financieros de startups pequeñas (crowdfunding)",
         "Fecha de los datos": _sec_range("sec_form_c"), "Condiciones": "Información pública, redistribuible",
         "Enlace": "https://www.sec.gov/data-research/sec-markets-data/crowdfunding-offerings-data-sets"},
        {"Fuente": "SEC EDGAR, S-1 y XBRL", "Qué aporta": "Empresas que solicitaron salir a bolsa, con sus ingresos y márgenes",
         "Fecha de los datos": _sec_range("sec_s1"), "Condiciones": "Información pública, redistribuible",
         "Enlace": "https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data"},
        {"Fuente": "Banco de España, Central de Balances", "Qué aporta": "Ratios de empresas españolas por sector CNAE y "
         "tamaño (cuartiles), de 2020 a 2024", "Fecha de los datos": "ejercicios 2020 a 2024",
         "Condiciones": "Reutilización libre citando al Banco de España",
         "Enlace": "https://app.bde.es/gnt_spa/rse/es/"},
        {"Fuente": "BORME (Agencia Estatal BOE)", "Qué aporta": "Sociedades españolas con constituciones o ampliaciones de "
         "capital: provincia, objeto social, CNAE y capital nominal. Sin datos de personas",
         "Fecha de los datos": "últimos 12 meses", "Condiciones": "Reutilización libre citando al BOE y respetando el RGPD",
         "Enlace": "https://www.boe.es/datosabiertos/api/api.php"},
        {"Fuente": "Supuestos propios", "Qué aporta": "Parámetros por etapa (IRR objetivo, supervivencia, dilución, iliquidez)",
         "Fecha de los datos": "n/d", "Condiciones": "Ilustrativos y editables", "Enlace": None},
    ])
    st.dataframe(sources_doc, hide_index=True, width="stretch",
                 column_config={"Enlace": st.column_config.LinkColumn("Enlace", display_text="abrir")})
    st.markdown(
        "Las fuentes que se investigaron pero aún no se integran (FRED, Kenneth French, Pablo Fernandez, Kroll, Carta, "
        "PitchBook) están en la pestaña **Datos y fuentes**, con el motivo. Nunca se promedian dos fuentes en silencio: "
        "si una métrica falta en la fuente preferida se usa la siguiente y la app lo avisa."
    )

    st.subheader("Limitaciones que conviene conocer")
    st.markdown(
        "- Los datos de Damodaran son de empresas cotizadas de EE. UU. Para una startup europea son una referencia, no una medida exacta.\n"
        "- Los múltiplos de la industria vienen de empresas grandes. Para una startup suelen ser altos: compáralos con los de las cotizadas más pequeñas.\n"
        "- Form D y Form C no informan la valoración de las empresas.\n"
        "- En el BORME el capital y las ampliaciones son nominales: sin la prima de emisión no miden el tamaño de una ronda.\n"
        "- Los ratios del Banco de España son agregados por sector y tamaño, no datos de empresas concretas.\n"
        "- Los supuestos por etapa son ilustrativos hasta que se integre una fuente verificada con datos por etapa.\n"
        "- La lectura de estados financieros en PDF es la menos fiable: revisa siempre lo detectado."
    )

with T["Supuestos"]:
    st.subheader("Supuestos por etapa")
    st.info(
        "Supuestos **ilustrativos** del autor, editables. Se sustituirán por una fuente verificada con datos por "
        "etapa cuando se integre (ver `data/sources.csv`). Los cambios afectan a todos los cálculos."
    )
    stage_cols = ["stage_label", "target_irr", "survival_prob", "future_dilution", "illiquidity_premium", "illiquidity_discount"]
    edited = st.data_editor(
        stages[stage_cols], key="stage_editor", hide_index=True, disabled=["stage_label"], width="stretch",
        column_config={
            "stage_label": "Etapa",
            "target_irr": st.column_config.NumberColumn("IRR objetivo", format="percent", min_value=0.0, max_value=2.0),
            "survival_prob": st.column_config.NumberColumn("Prob. de supervivencia", format="percent", min_value=0.01, max_value=1.0),
            "future_dilution": st.column_config.NumberColumn("Dilución futura", format="percent", min_value=0.0, max_value=0.95),
            "illiquidity_premium": st.column_config.NumberColumn("Prima de iliquidez", format="percent", min_value=0.0, max_value=0.3),
            "illiquidity_discount": st.column_config.NumberColumn("Descuento por iliquidez (múltiplos)", format="percent", min_value=0.0, max_value=0.9),
        },
    )
    st.caption("Fuente: supuesto propio (`data/stage_assumptions.csv`).")

stage = edited[edited["stage_label"] == stage_label].iloc[0]

with T["Escenarios"]:
    st.subheader("Factores por escenario")
    st.caption("Multiplican el crecimiento, el margen objetivo (mejora o empeora) y el múltiplo de salida del caso base.")
    sc_df = st.data_editor(
        pd.DataFrame([vars(s) for s in DEFAULT_SCENARIOS]), key="scenario_editor", hide_index=True,
        disabled=["name"], width="stretch",
        column_config={
            "name": "Escenario",
            "growth": st.column_config.NumberColumn("Crecimiento ×", min_value=0.0, max_value=5.0, step=0.05),
            "margin": st.column_config.NumberColumn("Margen ×", min_value=0.0, max_value=2.0, step=0.05),
            "multiple": st.column_config.NumberColumn("Múltiplo ×", min_value=0.0, max_value=5.0, step=0.05),
        },
    )
    scenarios = [ScenarioFactors(r.name, r.growth, r.margin, r.multiple) for r in sc_df.itertuples()]

with T["Monte Carlo"]:
    st.subheader("Supuestos de la simulación")
    c1, c2, c3, c4 = st.columns(4)
    mc = MCSettings(
        growth_sd=c1.number_input("Desv. del crecimiento (pp)", 0.0, 200.0, 20.0, 1.0) / 100,
        margin_sd=c2.number_input("Desv. del margen objetivo (pp)", 0.0, 50.0, 5.0, 0.5) / 100,
        multiple_sd=c3.number_input("Desv. del log del múltiplo", 0.0, 2.0, 0.35, 0.05),
        moic_target=c4.number_input("MOIC objetivo", 0.5, 100.0, 3.0, 0.5),
    )
    c5, c6, c7, c8 = st.columns(4)
    mc.corr_growth_margin = c5.number_input("Correlación crecimiento y margen", -0.95, 0.95, -0.2, 0.05)
    mc.corr_growth_multiple = c6.number_input("Correlación crecimiento y múltiplo", -0.95, 0.95, 0.5, 0.05)
    mc.corr_margin_multiple = c7.number_input("Correlación margen y múltiplo", -0.95, 0.95, 0.2, 0.05)
    mc.include_failure = c8.checkbox("Incluir fracaso (supervivencia)", True)
    st.caption(f"{mc.n_sims:,} simulaciones con semilla fija {mc.seed}: el resultado es reproducible.".replace(",", "."))

with T["Caja y ronda"]:
    cc1, cc2 = st.columns(2)
    cost_growth = cc1.number_input("Crecimiento anual de los costos (%)", -50.0, 500.0, 30.0, 5.0) / 100
    add_round = cc2.checkbox("Sumar la inversión de esta ronda a la caja", True)

# ======================================================================= resolución de datos

resolver = MetricResolver(D["metrics"], D["defs"], priority or None, apply_caps)
R = {m: resolver.get(industry, m) for m in [
    "unlevered_beta_cash_adj", "correlation_market", "operating_margin", "operating_margin_unadj",
    "ebitda_margin", "sales_to_capital", "cost_of_capital", "roc", "de_ratio", "tax_rate", "cost_of_debt",
    "ev_sales", "ev_ebitda_pos", "ev_ebitda_all", "n_firms",
]}
critical = ["unlevered_beta_cash_adj", "correlation_market", "operating_margin", "sales_to_capital", "ev_sales"]
missing = [R[m].label for m in critical if not R[m].ok]
if missing:
    st.error(f"Faltan datos imprescindibles para {industry}: {', '.join(missing)}. Elige otra industria u otra fuente.")
    st.stop()

# Valores de industria que el usuario puede sustituir
with st.sidebar, st.expander("Supuestos anclados en la industria"):
    target_margin = pct_input("Margen operativo objetivo (%)", R["operating_margin"].value, f"tm_{industry}_{apply_caps}",
                              -100.0, 90.0, help="Por defecto, margen operativo de la industria.\n\n" + R["operating_margin"].provenance())
    margin_year = st.slider("Año en que se alcanza el margen objetivo", 2, 10, 7)
    s2c = st.number_input("Ventas / capital invertido", 0.05, 20.0, float(round(R["sales_to_capital"].value, 2)), 0.05,
                          key=f"s2c_{industry}_{apply_caps}", help=R["sales_to_capital"].provenance())
    mult_res = R["ev_sales"] if exit_basis == "EV/Sales" else R["ev_ebitda_pos"]
    exit_multiple = st.number_input(f"Múltiplo de salida {exit_basis}", 0.1, 200.0, float(round(mult_res.value, 2)), 0.1,
                                    key=f"mult_{industry}_{exit_basis}_{apply_caps}",
                                    help="Por defecto, múltiplo de la industria" + (" (solo empresas con EBITDA positivo)" if exit_basis == "EV/EBITDA" else "")
                                    + ".\n\n" + mult_res.provenance())
    stable_growth = pct_input("Crecimiento estable perpetuo (%)", min(0.03, rf), "g_stable", -5.0, 10.0, 0.25,
                              help="Se limita a la tasa libre de riesgo.")
    distress = pct_input("Recuperación si fracasa (% del valor operativo)", 0.0, "distress", 0.0, 100.0, 5.0)

# ======================================================================= cálculos

use_total = beta_type == "Total"
dr = discount_rate(R["unlevered_beta_cash_adj"].value, R["correlation_market"].value, use_total, rf, erp,
                   size_prem, illiq_prem, de_ratio, tax, kd)
ind_de = R["de_ratio"].value if R["de_ratio"].ok else 0.0
ind_kd = R["cost_of_debt"].value if R["cost_of_debt"].ok else kd
mature_beta = relever_beta(R["unlevered_beta_cash_adj"].value, ind_de, tax)
mature_coc = wacc(rf + mature_beta * erp, ind_kd, tax, ind_de)
terminal_roc = R["roc"].value if R["roc"].ok else mature_coc

base = DCFInputs(
    revenue0=revenue0, growth_high=growth, stable_growth=stable_growth, current_margin=current_margin,
    target_margin=target_margin, margin_year=margin_year, tax_rate=tax, sales_to_capital=s2c,
    cost_of_capital=dr.cost_of_capital, mature_cost_of_capital=mature_coc, terminal_roc=terminal_roc,
    risk_free=rf, survival_prob=float(stage["survival_prob"]), distress_proceeds=distress, nol0=nol0,
    cash=cash, debt=debt, years=10, high_growth_years=hg_years,
)
dcf_res = dcf(base)
proj = dcf_res.projection

da_margin = max((R["ebitda_margin"].value if R["ebitda_margin"].ok else 0) - (R["operating_margin_unadj"].value if R["operating_margin_unadj"].ok else 0), 0.0)


def exit_metric(projection: pd.DataFrame, basis: str) -> tuple[float, float]:
    """(ingresos, EBITDA) en el año de salida."""
    row = projection.iloc[min(exit_year, len(projection)) - 1]
    ebitda = row["Ingresos"] * (row["Margen operativo"] + da_margin)
    return row["Ingresos"], ebitda


def vc_for(projection: pd.DataFrame, multiple: float):
    rev_exit, ebitda_exit = exit_metric(projection, exit_basis)
    base_metric = rev_exit if exit_basis == "EV/Sales" else ebitda_exit
    ev_exit = base_metric * multiple if base_metric > 0 else math.nan
    rate = float(stage["target_irr"]) if vc_mode == "irr" else dr.cost_of_equity
    res = vc_method(ev_exit, exit_year, terms.investment, rate, float(stage["future_dilution"]),
                    float(stage["survival_prob"]), vc_mode)
    return res, rev_exit, ebitda_exit


vc_res, rev_exit, ebitda_exit = vc_for(proj, exit_multiple)
deal = deal_returns(vc_res.exit_value, exit_year, terms, float(stage["future_dilution"]), float(stage["survival_prob"]))

# Múltiplos actuales
illiq_disc = float(stage["illiquidity_discount"])
ebitda_now = revenue0 * (current_margin + da_margin)
rev_fwd = revenue0 * (1 + growth)
size = D["size"]
size_val = lambda cls, m: size[(size.size_class == cls) & (size.metric == m)]["value"].iloc[0]  # noqa: E731
size_url = size[size.metric == "ev_sales"]["url"].iloc[0]
mult_rows = [
    ("EV/Sales industria (trailing)", R["ev_sales"].value, revenue0, "Ingresos 12 m", R["ev_sales"]),
    ("EV/Sales industria (forward, descontado 1 año)", R["ev_sales"].value / (1 + dr.cost_of_equity), rev_fwd, "Ingresos próximos 12 m", R["ev_sales"]),
    ("EV/EBITDA industria (EBITDA positivo)", R["ev_ebitda_pos"].value, ebitda_now, "EBITDA 12 m", R["ev_ebitda_pos"]),
    ("EV/EBITDA industria (todas)", R["ev_ebitda_all"].value, ebitda_now, "EBITDA 12 m", R["ev_ebitda_all"]),
    ("EV/Sales decil de menor capitalización (EE. UU.)", size_val("Bottom decile", "ev_sales"), revenue0, "Ingresos 12 m", None),
    ("EV/Sales todas las cotizadas (EE. UU.)", size_val("All firms", "ev_sales"), revenue0, "Ingresos 12 m", None),
]
mult_table = []
for name, m, base_metric, metric_name, res in mult_rows:
    ev = multiple_value(base_metric, m, illiq_disc)
    eq = ev + cash - debt if not math.isnan(ev) else math.nan
    mult_table.append({
        "Método": name, "Múltiplo": m, "Métrica": metric_name, "Valor de la métrica": base_metric,
        "Valor del equity": eq,
        "Fuente": (f"{res.source} · {res.industry_used} · {res.as_of}" if res else f"damodaran · mktcapmult · {size['as_of'].iloc[0]}"),
        "URL": res.url if res else size_url,
    })
mult_df = pd.DataFrame(mult_table)
valid_mult = mult_df["Valor del equity"].dropna()

# Escenarios
sc_rows = []
for f in scenarios:
    inp = scenario_inputs(base, f)
    r = dcf(inp)
    v, _, _ = vc_for(r.projection, exit_multiple * f.multiple)
    m_val = multiple_value(revenue0, R["ev_sales"].value * f.multiple, illiq_disc) + cash - debt
    sc_rows.append({"Escenario": f.name, "DCF": r.equity_value, "Método VC (pre-money)": v.pre_money,
                    "Múltiplos (EV/Sales)": m_val, "Ingresos año de salida": r.projection.iloc[exit_year - 1]["Ingresos"],
                    "Margen objetivo": inp.target_margin, "Crecimiento": inp.growth_high})
sc_res = pd.DataFrame(sc_rows)

# Monte Carlo
mc_multiple = exit_multiple if exit_basis == "EV/Sales" else R["ev_sales"].value
mc_res = simulate(base, mc_multiple, exit_year, terms.investment, terms.stake, float(stage["future_dilution"]), mc)


def sc_value(name: str, col: str) -> float:
    s = sc_res[sc_res["Escenario"] == name][col]
    return float(s.iloc[0]) if not s.empty else math.nan


# Rango por método
ff_rows = [
    {"method": "DCF", "low": sc_res["DCF"].min(), "high": sc_res["DCF"].max(), "mid": dcf_res.equity_value,
     "range_label": "Escenarios"},
    {"method": "Método VC", "low": sc_res["Método VC (pre-money)"].min(), "high": sc_res["Método VC (pre-money)"].max(),
     "mid": vc_res.pre_money, "range_label": "Escenarios"},
]
if not valid_mult.empty:
    ff_rows.append({"method": "Múltiplos", "low": valid_mult.min(), "high": valid_mult.max(),
                    "mid": float(mult_df["Valor del equity"].iloc[0]), "range_label": "Mín a máx"})
p = mc_res.percentiles["Valor DCF"]
# Central = media: con fracaso incluido la mediana puede caer en un escenario de quiebra,
# y la media es la magnitud comparable con el DCF (valor esperado).
ff_rows.append({"method": "Monte Carlo (DCF)", "low": p["P10"], "high": p["P90"], "mid": p["Media"], "range_label": "P10 a P90"})
ff_rows = [r for r in ff_rows if not any(math.isnan(r[k]) for k in ("low", "high", "mid"))]

mids = [r["mid"] for r in ff_rows]
lo_mid, hi_mid = min(mids), max(mids)
if terms.pre_money < lo_mid:
    verdict = ("Por debajo del rango de valoraciones", "La pre-money propuesta es menor que el valor central de todos los métodos.", "good")
elif terms.pre_money > hi_mid:
    verdict = ("Por encima del rango de valoraciones", "La pre-money propuesta supera el valor central de todos los métodos.", "critical")
else:
    verdict = ("Dentro del rango de valoraciones", "La pre-money propuesta está entre los valores centrales de los métodos.", "warning")

# Caja
runway = runway_months(cash, burn)
cash_proj = monthly_cash_projection(cash, revenue0, burn, growth, cost_growth, 48, terms.investment if add_round else 0.0)
funding_gap = max(dcf_res.capital_need - cash - terms.investment, 0.0)
dil_implied = implied_dilution(funding_gap, terms.post_money)

# ======================================================================= avisos

warnings = []
for res in resolver.used.values():
    warnings += res.warnings()
warnings += dcf_res.warnings
if exit_basis == "EV/EBITDA" and not (ebitda_exit > 0):
    warnings.append("El EBITDA proyectado en el año de salida no es positivo: el método VC por EV/EBITDA no aplica. Usa EV/Sales.")
with warn_box:
    if warnings:
        with st.expander(f"⚠️ Avisos sobre los datos ({len(warnings)})", expanded=False):
            for w in dict.fromkeys(warnings):
                st.markdown(f"- {w}")


def metric(col, label: str, value: str, res=None, help: str | None = None, delta: str | None = None):
    """Tarjeta con la procedencia en el tooltip. La moneda pasa a la etiqueta para que el valor quepa."""
    h = res.provenance() if res is not None else help
    if value.startswith(sym + " "):
        value, label = value[len(sym) + 1:], f"{label} ({sym})"
    col.metric(label, value, help=h)
    if delta:  # comparación como texto: la flecha de st.metric sugiere una variación que no existe
        col.caption(delta)


# ======================================================================= Resumen

with T["Resumen"]:
    icon = {"good": "🟢", "warning": "🟡", "critical": "🔴"}[verdict[2]]
    st.subheader(f"{icon} {verdict[0]}")
    st.markdown(
        f"La pre-money propuesta es **{fmt_money(terms.pre_money, currency)}** y los valores centrales de los métodos van "
        f"de **{fmt_money(lo_mid, currency)}** a **{fmt_money(hi_mid, currency)}**."
    )
    st.caption("Detalle de cada método en sus pestañas; sensibilidad en DCF y probabilidades en Monte Carlo. "
               "Es una comparación con la pre-money propuesta, no una recomendación de inversión.")
    c = st.columns(5)
    metric(c[0], "Pre-money propuesta", fmt_money(terms.pre_money, currency),
           help=f"Post-money {fmt_money(terms.post_money, currency)}; participación {fmt_pct(terms.stake)}")
    metric(c[1], "DCF (equity)", fmt_money(dcf_res.equity_value, currency),
           help="Valor esperado con probabilidad de supervivencia, más caja y menos deuda.")
    metric(c[2], "Método VC (pre-money)", fmt_money(vc_res.pre_money, currency),
           help=f"Salida {fmt_money(vc_res.exit_value, currency)} en el año {exit_year}, descontada a {fmt_pct(vc_res.discount_rate)}.")
    metric(c[3], "MOIC si hay salida", fmt_mult(deal.moic),
           help=f"Esperado con supervivencia ({fmt_pct(stage['survival_prob'])}): {fmt_mult(deal.expected_moic)}")
    metric(c[4], "Runway", f"{fmt_num(runway, 1)} meses", help="Caja disponible / burn rate mensual (sin la ronda).")

    st.plotly_chart(ch.football_field(ff_rows, terms.pre_money, currency), width="stretch")
    st.dataframe(pd.DataFrame([{
        "Método": r["method"], "Bajo": fmt_money(r["low"], currency), "Central": fmt_money(r["mid"], currency),
        "Alto": fmt_money(r["high"], currency), "Rango": r["range_label"],
    } for r in ff_rows]), hide_index=True, width="stretch")

    other = "EUR" if currency == "USD" else "USD"
    if fx_row is not None:
        rate = fx_row["rate"] if currency == "EUR" else 1 / fx_row["rate"]
        st.caption(
            f"Pre-money propuesta en {other}: {fmt_money(terms.pre_money * rate, other)} "
            f"(tipo de referencia BCE {fmt_num(fx_row['rate'], 4)} USD/EUR del {fx_row['date']})."
        )

# ======================================================================= DCF

with T["DCF"]:
    st.subheader("Costo de capital")
    c = st.columns(5)
    metric(c[0], "Beta desapalancada (industria)", fmt_num(R["unlevered_beta_cash_adj"].value, 2), R["unlevered_beta_cash_adj"])
    metric(c[1], "Correlación con el mercado", fmt_pct(R["correlation_market"].value), R["correlation_market"])
    metric(c[2], f"Beta usada ({dr.beta_type})", fmt_num(dr.beta_used, 2),
           help="Beta desapalancada corregida por efectivo" + (" / correlación" if use_total else "")
           + f", reapalancada con D/E {fmt_pct(de_ratio)}.")
    metric(c[3], "Costo del equity", fmt_pct(dr.cost_of_equity),
           help=f"{fmt_pct(rf)} + {fmt_num(dr.beta_used, 2)} × {fmt_pct(erp)} + {fmt_pct(size_prem + illiq_prem)} de primas")
    metric(c[4], "Tasa madura (año 10)", fmt_pct(mature_coc),
           help="WACC con beta de mercado y D/E de la industria: hacia ella converge la tasa de descuento.")

    st.subheader("Valor")
    c = st.columns(5)
    metric(c[0], "VP de los flujos (10 años)", fmt_money(dcf_res.pv_fcff, currency))
    metric(c[1], "VP del valor terminal", fmt_money(dcf_res.pv_terminal, currency),
           help=f"Crecimiento estable {fmt_pct(dcf_res.stable_growth_used)}; ROC terminal {fmt_pct(terminal_roc)}")
    metric(c[2], "Valor operativo en marcha", fmt_money(dcf_res.operating_value, currency))
    metric(c[3], "Ajustado por supervivencia", fmt_money(dcf_res.survival_adjusted_value, currency),
           help=f"Probabilidad de supervivencia {fmt_pct(base.survival_prob)} (supuesto de etapa)")
    metric(c[4], "Valor del equity", fmt_money(dcf_res.equity_value, currency), help="Más caja, menos deuda")

    g1, g2 = st.columns(2)
    sc, unit = ch.money_scale(proj["Ingresos"])
    g1.plotly_chart(ch.bars(proj["Año"], proj["Ingresos"] / sc, "Ingresos proyectados", f"{unit} {sym}"), width="stretch")
    g2.plotly_chart(ch.line(proj["Año"], proj["Margen operativo"], "Margen operativo", "%", pct=True,
                            ref=R["operating_margin"].value, ref_label="Industria"), width="stretch")
    g3, g4 = st.columns(2)
    sc2, unit2 = ch.money_scale(proj["FCFF"])
    g3.plotly_chart(ch.bars(proj["Año"], proj["FCFF"] / sc2, "Flujo de caja libre (FCFF)", f"{unit2} {sym}", signed=True), width="stretch")
    g4.plotly_chart(ch.line(proj["Año"], proj["Tasa de descuento"], "Tasa de descuento", "%", pct=True), width="stretch")

    with st.expander("Tabla de proyección"):
        fmt = proj.copy()
        for col in ["Ingresos", "EBIT", "Impuestos", "Reinversión", "FCFF", "VP del FCFF"]:
            fmt[col] = fmt[col].map(lambda v: fmt_money(v, currency))
        for col in ["Crecimiento", "Margen operativo", "Tasa de descuento"]:
            fmt[col] = fmt[col].map(fmt_pct)
        fmt["Factor de descuento"] = fmt["Factor de descuento"].map(lambda v: fmt_num(v, 3))
        st.dataframe(fmt, hide_index=True, width="stretch")

    st.subheader("Sensibilidad")
    s1, s2 = st.columns(2)
    deltas = [-0.04, -0.02, 0.0, 0.02, 0.04]
    gs = sorted({g for g in [0.0, 0.01, 0.02, 0.03, rf] if g <= rf})
    z = np.array([[dcf(DCFInputs(**{**vars(base), "cost_of_capital": base.cost_of_capital + d, "stable_growth": g})).equity_value
                   for d in deltas] for g in gs])
    zs, zu = ch.money_scale(z.ravel())
    s1.plotly_chart(ch.heatmap(z / zs, [fmt_pct(base.cost_of_capital + d) for d in deltas], [fmt_pct(g) for g in gs],
                               "Equity según tasa de descuento y crecimiento estable", "Tasa de descuento inicial",
                               "Crecimiento estable", f"{zu} {sym}"), width="stretch")

    def eq_with(**kw) -> float:
        return dcf(DCFInputs(**{**vars(base), **kw})).equity_value

    tor = pd.DataFrame([
        {"variable": "Crecimiento ±30 %", "low": eq_with(growth_high=growth * 0.7), "high": eq_with(growth_high=growth * 1.3)},
        {"variable": "Margen objetivo ±5 pp", "low": eq_with(target_margin=target_margin - 0.05), "high": eq_with(target_margin=target_margin + 0.05)},
        {"variable": "Tasa de descuento ∓2 pp", "low": eq_with(cost_of_capital=base.cost_of_capital + 0.02), "high": eq_with(cost_of_capital=max(base.cost_of_capital - 0.02, 0.01))},
        {"variable": "Ventas/capital ±30 %", "low": eq_with(sales_to_capital=s2c * 0.7), "high": eq_with(sales_to_capital=s2c * 1.3)},
        {"variable": "Supervivencia ±10 pp", "low": eq_with(survival_prob=max(base.survival_prob - 0.1, 0.01)), "high": eq_with(survival_prob=min(base.survival_prob + 0.1, 1.0))},
        {"variable": "Año del margen ±2", "low": eq_with(margin_year=min(margin_year + 2, 10)), "high": eq_with(margin_year=max(margin_year - 2, 1))},
    ])
    ts, tu = ch.money_scale(np.r_[tor["low"], tor["high"]])
    s2.plotly_chart(ch.tornado(tor.assign(low=tor["low"] / ts, high=tor["high"] / ts), dcf_res.equity_value / ts,
                               "Qué variables mueven más el valor", f"{tu} {sym}"), width="stretch")

# ======================================================================= Método VC

with T["Método VC"]:
    c = st.columns(4)
    metric(c[0], f"Ingresos año {exit_year}", fmt_money(rev_exit, currency))
    if exit_basis == "EV/EBITDA":
        metric(c[1], f"EBITDA año {exit_year}", fmt_money(ebitda_exit, currency),
               help=f"Margen operativo proyectado + D&A/ventas de la industria ({fmt_pct(da_margin)})")
    metric(c[2 if exit_basis == "EV/EBITDA" else 1], f"Múltiplo {exit_basis}", fmt_mult(exit_multiple), mult_res)
    metric(c[3 if exit_basis == "EV/EBITDA" else 2], "Valor de salida", fmt_money(vc_res.exit_value, currency))

    st.markdown(
        f"**Tratamiento del fracaso:** {vc_mode_label}. Tasa de descuento {fmt_pct(vc_res.discount_rate)}"
        + (f" (IRR objetivo de la etapa {stage_label})" if vc_mode == "irr" else f" (costo del equity) × supervivencia {fmt_pct(vc_res.survival_prob)}")
        + f"; dilución futura {fmt_pct(stage['future_dilution'])}."
    )
    st.latex(r"\text{Post-money} = \frac{\text{Valor de salida} \times (1-\text{dilución})" + (r"\times p" if vc_mode == "survival" else "")
             + r"}{(1+r)^{T}}")
    c = st.columns(4)
    metric(c[0], "Post-money (método VC)", fmt_money(vc_res.post_money, currency))
    metric(c[1], "Pre-money (método VC)", fmt_money(vc_res.pre_money, currency),
           delta=f"Propuesta: {fmt_money(terms.pre_money, currency)}")
    metric(c[2], "Participación necesaria hoy", fmt_pct(vc_res.required_stake),
           delta=f"Ofrecida: {fmt_pct(terms.stake)}")
    metric(c[3], "Participación a la salida", fmt_pct(vc_res.required_stake_at_exit))

    st.subheader("Retorno con las condiciones propuestas")
    c = st.columns(4)
    metric(c[0], "Participación a la salida", fmt_pct(deal.stake_exit))
    metric(c[1], "Cobro a la salida", fmt_money(deal.proceeds, currency))
    metric(c[2], "MOIC / IRR si hay salida", f"{fmt_mult(deal.moic)} / {fmt_pct(deal.irr)}")
    metric(c[3], "MOIC esperado (× supervivencia)", fmt_mult(deal.expected_moic),
           help=f"Probabilidad de supervivencia {fmt_pct(stage['survival_prob'])}")

# ======================================================================= Múltiplos

with T["Múltiplos"]:
    st.caption(f"Descuento por iliquidez aplicado: {fmt_pct(illiq_disc)} (supuesto de etapa, editable en «Supuestos»). "
               "El EBITDA actual se estima como ingresos × (margen operativo + D&A/ventas de la industria).")
    show = mult_df.copy()
    show["Múltiplo"] = show["Múltiplo"].map(fmt_mult)
    show["Valor de la métrica"] = show["Valor de la métrica"].map(lambda v: fmt_money(v, currency))
    show["Valor del equity"] = show["Valor del equity"].map(
        lambda v: fmt_money(v, currency) if not math.isnan(v) else "No aplica (métrica ≤ 0)")
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"URL": st.column_config.LinkColumn("URL", display_text="fuente")})
    if math.isnan(mult_df["Valor del equity"].iloc[2]):
        st.info("El EBITDA actual es negativo: los múltiplos de EBITDA no aplican hoy. Es lo normal en una startup temprana; "
                "usa EV/Sales o el EBITDA del año de salida en el método VC.")

    st.subheader("Madurez: rentables frente a todas las empresas")
    comp = pd.DataFrame({
        "Grupo": ["Solo EBITDA positivo (maduras)", "Todas las empresas (incluye pérdidas)"],
        "EV/EBITDA": [fmt_mult(R["ev_ebitda_pos"].value), fmt_mult(R["ev_ebitda_all"].value)],
    })
    st.dataframe(comp, hide_index=True)
    st.caption("Las clases de capitalización de Damodaran (decil más pequeño) sirven como proxy de empresas jóvenes.")
    sz = size[size.metric.isin(["ev_sales", "ev_ebitda", "operating_margin", "share_operating_loss", "total_beta"])]
    sz = sz.pivot_table(index=["size_rank", "size_class"], columns="metric", values="value").reset_index().drop(columns="size_rank")
    st.dataframe(sz.rename(columns={"size_class": "Clase", "ev_sales": "EV/Sales", "ev_ebitda": "EV/EBITDA",
                                    "operating_margin": "Margen operativo (mediana)", "share_operating_loss": "% con pérdida operativa",
                                    "total_beta": "Beta total"}),
                 hide_index=True, width="stretch",
                 column_config={"Margen operativo (mediana)": st.column_config.NumberColumn(format="percent"),
                                "% con pérdida operativa": st.column_config.NumberColumn(format="percent")})

# ======================================================================= Escenarios

with T["Escenarios"]:
    long = sc_res.melt(id_vars="Escenario", value_vars=["DCF", "Método VC (pre-money)", "Múltiplos (EV/Sales)"],
                       var_name="Método", value_name="Valor")
    st.plotly_chart(ch.scenario_bars(long, currency), width="stretch")
    view = sc_res.copy()
    for col in ["DCF", "Método VC (pre-money)", "Múltiplos (EV/Sales)", "Ingresos año de salida"]:
        view[col] = view[col].map(lambda v: fmt_money(v, currency))
    for col in ["Margen objetivo", "Crecimiento"]:
        view[col] = view[col].map(fmt_pct)
    st.dataframe(view, hide_index=True, width="stretch")

# ======================================================================= Monte Carlo

with T["Monte Carlo"]:
    fail = 1 - mc_res.survived.mean()
    surv = mc_res.survived if mc_res.survived.sum() >= 10 else np.ones_like(mc_res.survived)
    pm =mc_res.percentiles.get("MOIC si hay salida", mc_res.percentiles["MOIC"])
    pv = mc_res.percentiles.get("Valor DCF si sobrevive", mc_res.percentiles["Valor DCF"])
    c = st.columns(5)
    metric(c[0], "Prob. de fracaso", fmt_pct(fail), help="1 − probabilidad de supervivencia de la etapa, simulada.")
    metric(c[1], f"Prob. de MOIC ≥ {fmt_mult(mc.moic_target, 1)}", fmt_pct(mc_res.prob_moic_target), help="Incluye los fracasos.")
    metric(c[2], "Prob. de perder dinero", fmt_pct(mc_res.prob_loss), help="MOIC < 1x, incluidos los fracasos.")
    metric(c[3], "MOIC medio", fmt_mult(mc_res.percentiles["MOIC"]["Media"]), help="Incluye los fracasos (MOIC 0x).")
    metric(c[4], "MOIC P50 si hay salida", fmt_mult(pm["P50"]),
           help=f"P10 {fmt_mult(pm['P10'])} · P90 {fmt_mult(pm['P90'])}")
    if exit_basis == "EV/EBITDA":
        st.caption("La simulación del valor de salida usa EV/Sales de la industria (el EBITDA de salida puede ser negativo en muchas simulaciones).")
    st.caption("Las distribuciones muestran solo los escenarios en que la empresa sobrevive; los fracasos se resumen en las tarjetas y en la tabla.")
    h1, h2 = st.columns(2)
    h1.plotly_chart(ch.histogram(mc_res.moic[surv], "MOIC si hay salida", "MOIC",
                                 {k: pm[k] for k in ("P10", "P50", "P90")}, (f"Objetivo {fmt_mult(mc.moic_target, 1)}", mc.moic_target)),
                    width="stretch")
    vs, vu = ch.money_scale([pv["P90"]])  # escala por P90: la cola extrema no debe fijar las unidades
    h2.plotly_chart(ch.histogram(mc_res.dcf_equity[surv], "Equity (DCF) si sobrevive", f"{vu} {sym}",
                                 {k: pv[k] for k in ("P10", "P50", "P90")},
                                 ("Pre-money propuesta", terms.pre_money), scale=vs), width="stretch")
    st.dataframe(pd.DataFrame({
        k: {pk: (fmt_mult(v) if k.startswith("MOIC") else fmt_money(v, currency)) for pk, v in d.items()}
        for k, d in mc_res.percentiles.items()
    }), width="stretch")

# ======================================================================= Caja y ronda

with T["Caja y ronda"]:
    c = st.columns(4)
    metric(c[0], "Runway actual", f"{fmt_num(runway, 1)} meses")
    metric(c[1], "Caja que consume el plan (DCF)", fmt_money(dcf_res.capital_need, currency),
           help="Mínimo del FCFF acumulado: dinero necesario hasta que la empresa genera caja.")
    metric(c[2], "Déficit tras caja y ronda", fmt_money(funding_gap, currency))
    metric(c[3], "Dilución adicional implícita", fmt_pct(dil_implied),
           help="Si el déficit se levantara hoy al post-money propuesto. Cota superior: las rondas futuras suelen tener mayor valoración. "
                f"Compárala con la dilución futura del supuesto de etapa ({fmt_pct(stage['future_dilution'])}).")
    st.plotly_chart(ch.cash_chart(cash_proj, currency), width="stretch")
    out = cash_proj[cash_proj["Caja"] < 0]
    be = cash_proj[cash_proj["Consumo de caja"] <= 0]
    st.caption(
        ("La caja se agota en el mes " + str(int(out["Mes"].iloc[0])) if not out.empty else "La caja no se agota en 48 meses")
        + ("; el break-even mensual llega en el mes " + str(int(be["Mes"].iloc[0])) if not be.empty else "; no hay break-even en 48 meses")
        + ". Las tasas anuales se convierten a mensuales con (1 + g)^(1/12) − 1."
    )
    st.info(
        "La proyección mensual solo usa ingresos y costos operativos (burn rate). La caja que consume el plan del DCF "
        "es mayor porque añade la reinversión necesaria para crecer, calculada con el ratio ventas / capital de la "
        "industria. Si tu negocio necesita menos capital para crecer, sube ese ratio en la barra lateral."
    )

# ======================================================================= Datos y fuentes

with T["Datos y fuentes"]:
    st.subheader("Números de la industria usados en esta valoración")
    used = pd.DataFrame([{
        "Métrica": r.label, "Valor": r.value, "Valor original": r.raw_value, "Fuente": r.source,
        "Industria usada": r.industry_used, "Fecha de los datos": r.as_of, "Extraído": r.retrieved_at,
        "Avisos": " ".join(r.warnings()), "URL": r.url,
    } for r in resolver.used.values() if not r.missing])
    st.dataframe(used, hide_index=True, width="stretch",
                 column_config={"URL": st.column_config.LinkColumn("URL", display_text="fuente"),
                                "Valor": st.column_config.NumberColumn(format="%.4f"),
                                "Valor original": st.column_config.NumberColumn(format="%.4f")})

    st.subheader(f"Perfil completo de {industry}")
    st.caption("Si varias fuentes tienen la misma métrica, se muestran en columnas separadas, sin promediar.")
    prof = D["metrics"][D["metrics"]["industry_std"] == industry]
    piv = prof.pivot_table(index="metric", columns="source", values="value", aggfunc="first")
    piv.insert(0, "Métrica", [D["defs"].loc[m, "label_es"] if m in D["defs"].index else m for m in piv.index])
    piv.insert(1, "Unidad", [D["defs"].loc[m, "unit"] if m in D["defs"].index else "" for m in piv.index])
    st.dataframe(piv.reset_index(drop=True), hide_index=True, width="stretch")

    st.subheader("Etapa de crecimiento del sector")
    grow_metrics = ["cagr_revenue_5y", "exp_revenue_growth_2y", "exp_revenue_growth_5y", "fundamental_ebit_growth", "reinvestment_rate", "roc"]
    gr = [resolver.get(industry, m) for m in grow_metrics]
    st.dataframe(pd.DataFrame([{"Indicador": r.label, "Valor": fmt_pct(r.value), "Original": fmt_pct(r.raw_value),
                                "Recortado": "sí" if r.capped else "", "Fuente": r.source} for r in gr]),
                 hide_index=True, width="stretch")

    st.subheader("Datos de mercado")
    mk = pd.DataFrame([
        {"Serie": "Bono del Tesoro EE. UU. 10 años", "Valor": fmt_pct(us_rf["value"], 2), "Fecha": us_rf["date"], "Fuente": us_rf["source"], "URL": us_rf["url"]},
        {"Serie": "Prima de riesgo implícita EE. UU.", "Valor": fmt_pct(erp_row["value"], 2), "Fecha": erp_row["date"], "Fuente": erp_row["source"], "URL": erp_row["url"]},
        {"Serie": "Curva AAA zona euro 10 años", "Valor": fmt_pct(ea_rf["value"], 2), "Fecha": ea_rf["date"], "Fuente": ea_rf["source"], "URL": ea_rf["url"]},
        {"Serie": "Tipo de cambio USD por EUR", "Valor": fmt_num(fx_row["rate"], 4), "Fecha": fx_row["date"], "Fuente": fx_row["source"], "URL": fx_row["url"]},
    ])
    st.dataframe(mk, hide_index=True, width="stretch", column_config={"URL": st.column_config.LinkColumn("URL", display_text="fuente")})

    st.subheader("Registro de fuentes")
    st.dataframe(D["sources"], hide_index=True, width="stretch")

# ======================================================================= Comparables SEC


@st.cache_data(show_spinner=False)
def load_sec(dataset: str) -> pd.DataFrame:
    return cmp.load(dataset)


with T["Comparables"]:
    st.caption(
        "Empresas reales con datos públicos: presentaciones ante la SEC (EE. UU., últimos 4 trimestres) y actos "
        "inscritos en el Registro Mercantil publicados en el BORME (España, últimos 12 meses). Los importes se muestran "
        "en la moneda base. Selecciona una o varias filas para compararlas con tu startup y, si quieres, precargar sus "
        "datos en el modelo."
    )
    ds = st.radio("Fuente", list(cmp.DATASETS), format_func=cmp.DATASETS.get, horizontal=True, key="sec_ds")
    # Unidades de la moneda de la fuente por unidad de la moneda base: x / rate pasa a la moneda base
    rate = usd_rate(currency) / usd_rate(cmp.CURRENCY[ds])
    df_all = load_sec(ds)
    if ds == "borme":
        st.info("BORME: el capital y las ampliaciones son importes **nominales**. No incluyen la prima de emisión, así "
                "que no miden el tamaño real de una ronda. No hay ingresos ni beneficios. Fuente: basado en datos de la "
                "Agencia Estatal Boletín Oficial del Estado.", icon="ℹ️")
    f1, f2, f3 = st.columns([2, 2, 1])
    text = f1.text_input("Buscar por nombre" + (" u objeto social" if ds == "borme" else ""), key=f"sec_q_{ds}",
                         placeholder="p. ej. software, inteligencia artificial, biotecnología" if ds == "borme" else "p. ej. robotics, health, AI…")
    if ds == "sec_form_c":
        f2.caption("Form C no informa la industria: filtra por nombre o por ingresos.")
        inds = None
    else:
        ind_opts = sorted(df_all["industry_std"].dropna().unique())
        # BORME: solo una parte de las sociedades declara CNAE, así que no se filtra por industria por defecto
        inds = f2.multiselect("Industria", ind_opts, default=[industry] if industry in ind_opts and ds != "borme" else [],
                              key=f"sec_ind_{ds}_{industry}",
                              help="En el BORME solo las sociedades que indican su CNAE tienen industria; busca mejor por objeto social."
                              if ds == "borme" else None)
    young = f3.checkbox("Constituidas hace < 5 años", True, key="sec_young") if ds == "sec_form_d" else False
    with_rev = f3.checkbox("Solo con ingresos", True, key="sec_rev") if ds == "sec_form_c" else False
    if ds == "borme":
        if f3.checkbox("Solo activas", True, key="borme_alive", help="Excluye las sociedades disueltas o extinguidas."):
            df_all = df_all[~df_all["dissolved"].astype(bool)]
        if f3.checkbox("Con ampliaciones", False, key="borme_ampl",
                       help="Solo sociedades que han ampliado capital: señal de que han levantado fondos."):
            df_all = df_all[df_all["n_capital_increases"] > 0]
    res = cmp.search(df_all, text, inds, young, with_rev)

    cols = {
        "sec_form_d": ["name", "industry_std", "state", "revenue_range", "round_size", "total_sold", "n_investors", "first_sale_date", "filing_date", "url"],
        "sec_form_c": ["name", "state", "revenue", "growth", "net_margin", "cash", "employees", "round_size", "filing_date", "url"],
        "sec_s1": ["name", "industry_std", "state", "fiscal_year", "revenue", "growth", "operating_margin", "filing_date", "url"],
        "borme": ["name", "industry_std", "province", "town", "constitution_date", "capital_latest", "n_capital_increases",
                  "capital_increases_total", "purpose", "last_act_date", "url"],
    }[ds]
    labels = {"name": "Empresa", "industry_std": "Industria", "state": "Estado/país", "revenue_range": "Rango de ingresos",
              "round_size": f"Oferta ({sym})", "total_sold": f"Vendido ({sym})", "n_investors": "Inversores",
              "first_sale_date": "Primera venta", "filing_date": "Presentación", "revenue": f"Ingresos ({sym})",
              "growth": "Crecimiento", "net_margin": "Margen neto", "cash": f"Caja ({sym})", "employees": "Empleados",
              "operating_margin": "Margen operativo", "fiscal_year": "Ejercicio",
              "url": "BORME" if ds == "borme" else "EDGAR", "province": "Provincia", "town": "Municipio",
              "constitution_date": "Constitución", "capital_latest": f"Capital nominal ({sym})", "n_capital_increases": "Ampliaciones",
              "capital_increases_total": f"Ampliado nominal ({sym})", "purpose": "Objeto social", "last_act_date": "Último acto"}
    money_cols = ("round_size", "total_sold", "revenue", "cash", "capital_latest", "capital_increases_total")
    view = res[cols].copy()
    for c in money_cols:
        if c in view:
            view[c] = view[c] / rate
    if "industry_std" in view:
        view["industry_std"] = view["industry_std"].fillna("Sin clasificar")
    n_total = len(cmp.search(df_all, text, inds, young, with_rev, limit=10**7))
    st.caption(f"{n_total:,} empresas coinciden".replace(",", ".") + (" (se muestran las 500 más recientes)." if n_total > 500 else "."))
    event = st.dataframe(
        view.rename(columns=labels), hide_index=True, width="stretch", height=320,
        on_select="rerun", selection_mode="multi-row", key=f"sec_table_{ds}",
        column_config={
            "EDGAR": st.column_config.LinkColumn("EDGAR", display_text="ver filing"),
            "BORME": st.column_config.LinkColumn("BORME", display_text="ver BORME"),
            "Crecimiento": st.column_config.NumberColumn(format="percent"),
            "Margen neto": st.column_config.NumberColumn(format="percent"),
            "Margen operativo": st.column_config.NumberColumn(format="percent"),
            **{labels[c]: st.column_config.NumberColumn(format="compact") for c in money_cols if c in cols},
        },
    )
    picked = res.iloc[event.selection.rows] if event and event.selection.rows else res.iloc[0:0]

    # Contexto: dónde queda tu startup en la distribución de la fuente
    if ds == "borme":
        sample = df_all[df_all["industry_std"].isin(inds)] if inds else df_all
        k1, k2, k3 = st.columns(3)
        metric(k1, "Sociedades en la muestra", f"{len(sample):,}".replace(",", "."))
        metric(k2, "Constituidas en el periodo", f"{int(sample['constitution_date'].notna().sum()):,}".replace(",", "."))
        metric(k3, "Con ampliaciones de capital", f"{int((sample['n_capital_increases'] > 0).sum()):,}".replace(",", "."),
               help="Señal de que la sociedad ha levantado fondos, aunque el importe nominal no mide la ronda.")
        series = pd.Series(dtype=float)
    elif ds == "sec_form_d":
        sample = df_all[df_all["industry_std"].isin(inds)] if inds else df_all
        sample = sample[sample["inc_within_5y"].astype(bool)] if young else sample
        your, series, what = terms.investment * rate, sample["round_size"], "tamaño de ronda"
    else:
        sample = df_all[df_all["industry_std"].isin(inds)] if inds else df_all
        your, series, what = revenue0 * rate, sample["revenue"], "ingresos"
    if ds != "borme":
        pctl = cmp.percentile_of(your, series[series > 0])
        k1, k2, k3 = st.columns(3)
        metric(k1, f"Empresas en la muestra", f"{int((series > 0).sum()):,}".replace(",", "."))
        metric(k2, f"Mediana de {what}", fmt_money(float(series[series > 0].median()) / rate, currency) if (series > 0).any() else "n/d")
        metric(k3, f"Tu {what}: percentil", fmt_pct(pctl, 0) if not math.isnan(pctl) else "n/d",
               help=f"Porcentaje de la muestra con {what} menor que el tuyo ({fmt_money(your / rate, currency)}).")
        if (series > 0).sum() >= 10:
            st.plotly_chart(ch.log_histogram(series[series > 0] / rate, f"Distribución de {what} en la muestra",
                                             f"{what.capitalize()} ({sym}, escala logarítmica)",
                                             {"Tu startup": your / rate}), width="stretch")

    lib_now = my_companies()
    mine = []
    if lib_now:
        lib_by_id = {c["id"]: c for c in lib_now}
        mine_ids = st.multiselect("Añadir mis empresas guardadas a la comparación", list(lib_by_id),
                                  format_func=lambda i: lib_by_id[i]["name"], key="sec_mine")
        mine = [lib_by_id[i] for i in mine_ids]

    if not picked.empty or mine:
        st.subheader("Comparación con tu startup")
        user = {
            "industry_std": industry, "revenue": revenue0, "growth": growth, "operating_margin": current_margin,
            "cash": cash, "round_size": terms.investment, "filing_date": "n/d",
        }
        shown = picked.copy()
        for c in ("revenue", "revenue_est", "cash", "round_size", "total_sold", "capital_latest"):
            if c in shown:
                shown[c] = shown[c] / rate
        own_rows = []
        for c in mine:
            fx_c = usd_rate(c.get("currency") or "USD") / rate  # moneda de la empresa -> moneda base
            inp = c.get("inputs") or {}
            fin = c.get("financials") or {}
            last = fin[max(fin, key=str)] if fin else {}
            m = lambda k: (inp.get(k) or 0) * fx_c if inp.get(k) is not None else np.nan  # noqa: E731
            own_rows.append({
                "name": f"{c['name']} (guardada)", "industry_std": c.get("industry"), "revenue": m("revenue"),
                "growth": inp.get("growth"), "operating_margin": inp.get("current_margin"),
                "net_margin": (last.get("net_income") / last["revenue"]) if last.get("revenue") and last.get("net_income") is not None else np.nan,
                "cash": m("cash"), "round_size": m("investment"), "employees": last.get("employees", np.nan),
                "filing_date": (c.get("updated_at") or "")[:10],
            })
        if own_rows:
            shown = pd.concat([pd.DataFrame(own_rows), shown], ignore_index=True)
        st.dataframe(cmp.comparison_table(user, shown, lambda x: fmt_money(x, currency), fmt_pct),
                     hide_index=True, width="stretch")

    if not picked.empty:

        st.subheader("Precargar en el modelo")
        who = st.selectbox("Empresa", picked.index, format_func=lambda i: picked.loc[i, "name"], key=f"sec_prefill_{ds}")
        pf = cmp.prefill_from(picked.loc[who], ds)
        if pf.values:
            prev = []
            for k, v in pf.values.items():
                shown_v = (fmt_money(v / rate, currency) if k in ("revenue", "cash", "investment")
                           else fmt_pct(v) if k in ("growth", "current_margin") else v)
                prev.append({"Entrada": {"revenue": "Ingresos 12 m", "growth": "Crecimiento anual", "current_margin": "Margen operativo actual",
                                         "cash": "Caja", "investment": "Inversión", "industry": "Industria"}[k],
                             "Valor": shown_v, "Origen": pf.notes[k]})
            st.dataframe(pd.DataFrame(prev), hide_index=True, width="stretch")
            st.button(
                "Precargar estos datos en el modelo", type="primary", key=f"sec_btn_{ds}",
                on_click=apply_prefill, args=({**pf.values, "_label": str(picked.loc[who, "name"])}, currency),
                help="Sustituye esas entradas de la barra lateral; el resto se mantiene. Todo se recalcula al instante.",
            )
        else:
            st.info("Esta presentación no trae datos que se puedan cargar en el modelo.")

# ======================================================================= Ratios España

from src.sources import bde as bde_source  # noqa: E402

BDE_LABELS = {
    "revenue_growth_1y": "Crecimiento de las ventas", "ebitda_margin": "EBITDA sobre ventas",
    "roi_ordinary": "Rentabilidad ordinaria del activo (ROI)", "roe": "Rentabilidad de los recursos propios (ROE)",
    "cost_of_debt": "Coste medio de la financiación", "debt_to_liabilities": "Recursos ajenos sobre pasivo",
    "sales_per_employee": "Ventas por empleado", "personnel_cost_per_employee": "Gasto de personal por empleado",
    "receivable_days": "Periodo medio de cobro (días)", "payable_days": "Periodo medio de pago (días)",
}


@st.cache_data(show_spinner=False)
def load_bde_detail() -> pd.DataFrame:
    return bde_source.load_detail()


size_for_revenue = bde_source.size_for_revenue


def fmt_ratio(v: float, unit: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/d"
    if unit == "decimal":
        return fmt_pct(v)
    if unit == "eur":
        return fmt_money(v, "EUR", 1)
    return fmt_num(v, 0)


def position(v: float, p25: float, p50: float, p75: float) -> str:
    if v is None or any(isinstance(x, float) and math.isnan(x) for x in (v, p25, p50, p75)):
        return "n/d"
    if v < p25:
        return "Por debajo del P25"
    if v < p50:
        return "Entre P25 y la mediana"
    if v < p75:
        return "Entre la mediana y P75"
    return "Por encima del P75"


with T["Ratios España"]:
    det = load_bde_detail()
    if det.empty:
        st.info("Los ratios del Banco de España aún no están disponibles en esta versión.")
    else:
        st.caption("Ratios de empresas españolas no financieras por sector (CNAE) y tamaño: cuartil inferior (P25), mediana "
                   "y cuartil superior (P75). Fuente: elaboración propia con datos extraídos del sitio web del Banco de "
                   "España (www.bde.es), Central de Balances.")
        sec_names = det.drop_duplicates("sector_code").set_index("sector_code")["sector_name"].to_dict()
        bde_xw = D["crosswalk"][D["crosswalk"]["source"] == "bde"].set_index("industry_std")["industry_original"].to_dict()
        default_sector = bde_xw.get(industry, "ZC")
        codes = sorted(sec_names, key=lambda c: (c not in ("Z0", "ZC"), c))
        rev_eur = revenue0 * usd_rate(currency) / usd_rate("EUR")
        sizes = det.drop_duplicates("size_id").set_index("size_id")["size_name"].to_dict()
        r1, r2, r3 = st.columns([3, 2, 1])
        sector = r1.selectbox("Sector CNAE", codes, index=codes.index(default_sector) if default_sector in codes else 0,
                              format_func=lambda c: f"{c} · {sec_names[c]}", key=f"bde_sector_{industry}",
                              help="Por defecto, el sector asociado a la industria elegida en la barra lateral.")
        size_ids = list(sizes)
        default_size = size_for_revenue(rev_eur)
        size_id = r2.selectbox("Tamaño (cifra de negocios)", size_ids, index=size_ids.index(default_size) if default_size in size_ids else 0,
                               format_func=sizes.get, key=f"bde_size_{default_size}",
                               help="Por defecto, el tramo que corresponde a tus ingresos.")
        years = sorted(det["year"].unique(), reverse=True)
        year = r3.selectbox("Ejercicio", years, key="bde_year")
        if sector not in bde_xw.values():
            st.caption(f"La industria «{industry}» no tiene un sector CNAE asociado; se muestra el total de empresas.")

        employees = st.number_input("Empleados de tu empresa (opcional, para ventas por empleado)", 0, 100000, 0, 1, key="bde_emp")
        cut = det[(det["sector_code"] == sector) & (det["size_id"] == size_id)]
        now = cut[cut["year"] == year].set_index("metric")
        mine_vals = {
            "revenue_growth_1y": growth,
            "ebitda_margin": current_margin + da_margin,
            "cost_of_debt": kd if de_ratio > 0 else math.nan,
            "sales_per_employee": rev_eur / employees if employees else math.nan,
        }
        rows = []
        for m, label in BDE_LABELS.items():
            if m not in now.index:
                continue
            r = now.loc[m]
            mv = mine_vals.get(m, math.nan)
            rows.append({"Ratio": label, "P25": fmt_ratio(r["p25"], r["unit"]), "Mediana": fmt_ratio(r["p50"], r["unit"]),
                         "P75": fmt_ratio(r["p75"], r["unit"]), "Tu empresa": fmt_ratio(mv, r["unit"]),
                         "Posición": position(mv, r["p25"], r["p50"], r["p75"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Tu empresa: crecimiento y margen EBITDA (margen operativo más amortización típica de la industria) salen "
                   "de la barra lateral; el coste de la deuda, si tienes deuda. Las ventas se convierten a euros con el tipo "
                   "del BCE. Para una startup en pérdidas lo normal es quedar por debajo del P25 en rentabilidad.")

        pick = st.selectbox("Evolución del ratio", [m for m in BDE_LABELS if m in cut["metric"].unique()],
                            format_func=BDE_LABELS.get, key="bde_trend")
        tr = cut[cut["metric"] == pick].sort_values("year")
        if not tr.empty:
            unit = tr["unit"].iloc[0]
            k = 100 if unit == "decimal" else 1
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tr["year"], y=tr["p75"] * k, mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=tr["year"], y=tr["p25"] * k, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(42,120,214,0.15)", name="P25 a P75",
                                     hovertemplate="%{x}: P25 %{y:,.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=tr["year"], y=tr["p50"] * k, mode="lines+markers", name="Mediana",
                                     line=dict(color=ch.BLUE, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                                     hovertemplate="%{x}: mediana %{y:,.1f}<extra></extra>"))
            ch._layout(fig, f"{BDE_LABELS[pick]} · {sec_names[sector]} · {sizes[size_id]}", height=340,
                       legend=dict(orientation="h", y=1.12, x=0))
            fig.update_xaxes(dtick=1, title="Ejercicio")
            fig.update_yaxes(title="%" if unit == "decimal" else ("euros" if unit == "eur" else "días"),
                             ticksuffix=" %" if unit == "decimal" else "")
            st.plotly_chart(fig, width="stretch")

# ======================================================================= Mis empresas

CO_MONEY =[("revenue", "Ingresos últimos 12 meses"), ("burn", "Burn rate mensual"), ("cash", "Caja disponible"),
            ("debt", "Deuda financiera"), ("nol", "Pérdidas fiscales acumuladas"), ("investment", "Inversión de la ronda"),
            ("pre_money", "Pre-money propuesta")]
CO_PCT = [("growth", "Crecimiento anual de ingresos (%)"), ("current_margin", "Margen operativo actual (%)")]
ITEM_LABEL = {k: v[0] for k, v in stm.ITEMS.items()}
LABEL_ITEM = {v: k for k, v in ITEM_LABEL.items()}


def _fin_to_editor(fin: dict) -> pd.DataFrame:
    """{año: {partida: valor}} -> tabla editable (filas = partidas, columnas = años)."""
    if not fin:
        return pd.DataFrame({"Partida": [ITEM_LABEL[i] for i in ("revenue", "operating_income", "net_income", "cash")]})
    df = pd.DataFrame(fin).reindex([i for i in stm.ITEM_ORDER if any(i in v for v in fin.values())])
    df.index = [ITEM_LABEL[i] for i in df.index]
    df = df[sorted(df.columns, key=str)]
    return df.rename_axis("Partida").reset_index()


def _editor_to_fin(df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        item = LABEL_ITEM.get(str(row.get("Partida", "")).strip())
        if not item:
            continue
        for col in df.columns:
            if col == "Partida":
                continue
            v = stm.parse_amount(row[col])
            if not math.isnan(v):
                out.setdefault(str(col), {})[item] = v
    return out


def co_select() -> None:
    """Callback del selector: vuelca la empresa elegida (o una nueva) en el formulario."""
    sel = st.session_state.get("co_pick")
    c = next((x for x in my_companies() if x["id"] == sel), None) or cs.new_company()
    st.session_state["co_name"] = c.get("name", "")
    st.session_state["co_industry"] = c.get("industry") or industry
    st.session_state["co_stage"] = c.get("stage") or stage_label
    st.session_state["co_currency"] = c.get("currency") or currency
    st.session_state["co_notes"] = c.get("notes", "")
    inp = c.get("inputs", {})
    for k, _ in CO_MONEY:
        st.session_state[f"co_{k}"] = float(inp.get(k) or 0.0)
    for k, _ in CO_PCT:
        st.session_state[f"co_{k}"] = round(float(inp.get(k) or 0.0) * 100, 2)
    st.session_state["co_fin"] = c.get("financials", {})
    st.session_state["co_files"] = c.get("source_files", [])
    st.session_state["co_editor_v"] = st.session_state.get("co_editor_v", 0) + 1


def co_fill(summary: dict, fin: dict, files: list[str], detected_currency: str | None) -> None:
    """Callback: rellena el formulario con lo detectado en los estados financieros."""
    for k, _ in CO_MONEY:
        if k in summary:
            st.session_state[f"co_{k}"] = float(round(summary[k]))
    for k, _ in CO_PCT:
        if k in summary:
            st.session_state[f"co_{k}"] = round(summary[k] * 100, 2)
    if detected_currency in ("USD", "EUR"):
        st.session_state["co_currency"] = detected_currency
    st.session_state["co_fin"] = fin
    st.session_state["co_files"] = sorted(set(st.session_state.get("co_files", [])) | set(files))
    st.session_state["co_editor_v"] = st.session_state.get("co_editor_v", 0) + 1
    st.session_state["co_msg"] = ("Campos rellenados con los estados financieros. Revísalos y pulsa 💾 Guardar al final "
                                  "de la página: hasta entonces la empresa no está guardada.")


def co_collect(fin_df: pd.DataFrame) -> dict:
    c = {"id": st.session_state.get("co_pick") if st.session_state.get("co_pick") != "__new__" else None,
         "name": st.session_state.get("co_name", "").strip(), "industry": st.session_state.get("co_industry"),
         "stage": st.session_state.get("co_stage"), "currency": st.session_state.get("co_currency"),
         "notes": st.session_state.get("co_notes", ""),
         "inputs": {**{k: st.session_state.get(f"co_{k}") for k, _ in CO_MONEY},
                    **{k: (st.session_state.get(f"co_{k}") or 0) / 100 for k, _ in CO_PCT}},
         "financials": _editor_to_fin(fin_df), "source_files": st.session_state.get("co_files", [])}
    return {k: v for k, v in c.items() if v is not None or k != "id"}


def co_save(fin_df: pd.DataFrame, also_load: bool = False) -> None:
    c = co_collect(fin_df)
    if not c["name"]:
        st.session_state["co_msg_err"] = "Ponle un nombre a la empresa."
        return
    if not c.get("id"):
        c.pop("id", None)
        c = {**cs.new_company(c["name"]), **c}
    try:
        saved = company_store().save(current_user(), c)
    except Exception as e:  # noqa: BLE001: el usuario debe ver por qué no se guardó
        st.session_state["co_msg_err"] = f"No se pudo guardar: {e}"
        return
    refresh_library()
    st.session_state["co_pick"] = saved["id"]
    st.session_state["co_msg"] = f"«{saved['name']}» guardada. Ya puedes cargarla desde la barra lateral (📁 Mis empresas)."
    if also_load:
        load_company(saved)


def co_delete() -> None:
    sel = st.session_state.get("co_pick")
    if sel and sel != "__new__":
        company_store().delete(current_user(), sel)
        refresh_library()
        st.session_state["co_pick"] = "__new__"
        co_select()
        st.session_state["co_msg"] = "Empresa eliminada."


with T["Mis empresas"]:
    st.warning("Biblioteca **compartida y sin contraseña**: cualquiera con el enlace de la app puede ver, editar y "
               "borrar estas empresas. No guardes datos confidenciales.", icon="⚠️")
    if st.session_state.get("lib_error"):
        st.error(f"No se pudo leer la biblioteca: {st.session_state.pop('lib_error')}")
    lib = my_companies()
    by_id = {c["id"]: c for c in lib}
    if "co_pick" not in st.session_state or (st.session_state["co_pick"] != "__new__" and st.session_state["co_pick"] not in by_id):
        st.session_state["co_pick"] = "__new__"
    if "co_name" not in st.session_state:
        co_select()
    st.selectbox("Empresa", ["__new__", *by_id], key="co_pick", on_change=co_select,
                 format_func=lambda i: "➕ Nueva empresa" if i == "__new__" else by_id[i]["name"])
    if st.session_state.get("co_msg"):
        msg = st.session_state.pop("co_msg")
        st.success(msg)
        st.toast(msg, icon="✅")  # visible aunque el botón quede lejos del mensaje
    if st.session_state.get("co_msg_err"):
        msg = st.session_state.pop("co_msg_err")
        st.error(msg)
        st.toast(msg, icon="⚠️")

    st.subheader("1. Estados financieros (opcional)")
    st.caption("Sube la cuenta de resultados, el balance o ambos (Excel, CSV o PDF). Detecto las partidas por su nombre "
               "en español o inglés y la escala (miles, millones). Revisa siempre lo detectado.")
    u1, u2 = st.columns([3, 1])
    files = u1.file_uploader("Archivos", type=["xlsx", "xlsm", "csv", "pdf"], accept_multiple_files=True,
                             key=f"co_upload_{st.session_state['co_pick']}", label_visibility="collapsed")
    u2.download_button("Descargar plantilla (CSV)", stm.TEMPLATE.to_csv(index=False).encode("utf-8-sig"),
                       "plantilla_estados_financieros.csv", "text/csv", key="co_tpl")
    if files:
        results = []
        for f in files:
            try:
                results.append((f.name, stm.parse_statement(f.getvalue(), f.name)))
            except Exception as e:  # noqa: BLE001: archivo ilegible: se informa y se sigue con los demás
                st.error(f"{f.name}: no se pudo leer ({e}).")
        for name, r in results:
            for w in r.warnings:
                st.warning(f"{name}: {w}")
        merged = stm.merge_results([r for _, r in results])
        if not merged.empty:
            fin = {str(y): {i: float(merged.loc[i, y]) for i in merged.index if pd.notna(merged.loc[i, y])} for y in merged.columns}
            summary = stm.summarize(merged)
            cur_detected = next((r.currency for _, r in results if r.currency), None)
            with st.expander(f"Partidas detectadas ({sum(len(r.detections) for _, r in results)})", expanded=True):
                det = pd.DataFrame([{"Partida": ITEM_LABEL[d.item], "Año": str(d.year), "Texto en el archivo": d.label,
                                     "Dónde": f"{n} · {d.where}", "Valor": d.value * (1 if d.item in stm.COUNT_ITEMS else r.scale)}
                                    for n, r in results for d in r.detections])
                st.dataframe(det.sort_values(["Partida", "Año"]), hide_index=True, width="stretch",
                             column_config={"Valor": st.column_config.NumberColumn(format="localized")})
            prev = {"revenue": "Ingresos", "growth": "Crecimiento", "current_margin": "Margen operativo", "net_margin": "Margen neto",
                    "cash": "Caja", "debt": "Deuda", "burn": "Burn mensual (estimado)", "nol": "Pérdidas acumuladas", "employees": "Empleados"}
            st.dataframe(pd.DataFrame([{"Campo": prev[k], "Valor": (fmt_pct(v) if k in ("growth", "current_margin", "net_margin")
                                                                      else fmt_num(v, 0)), "Año": summary.get("year")}
                                       for k, v in summary.items() if k in prev]), hide_index=True, width="stretch")
            st.caption("Burn mensual estimado = flujo de caja operativo negativo / 12 (si no está, beneficio neto + amortización). "
                       "Pérdidas acumuladas = suma de los resultados netos negativos de los años del archivo.")
            st.button("Rellenar los campos con lo detectado", type="primary", key="co_fill_btn",
                      on_click=co_fill, args=(summary, fin, [f.name for f in files], cur_detected))

    st.subheader("2. Datos de la empresa")
    a1, a2, a3, a4 = st.columns([3, 3, 2, 1])
    a1.text_input("Nombre", key="co_name")
    a2.selectbox("Industria", options, key="co_industry", format_func=lambda i: f"{i} · {sector_of[i]}")
    a3.selectbox("Etapa", stages["stage_label"].tolist(), key="co_stage")
    a4.selectbox("Moneda", ["USD", "EUR"], key="co_currency")
    b = st.columns(4)
    for i, (k, label) in enumerate(CO_MONEY):
        b[i % 4].number_input(label, 0.0, None, step=50_000.0, format="%.0f", key=f"co_{k}")
    p = st.columns(4)
    p[0].number_input(CO_PCT[0][1], -100.0, 500.0, step=0.5, format="%.2f", key="co_growth")
    p[1].number_input(CO_PCT[1][1], -1000.0, 90.0, step=0.5, format="%.2f", key="co_current_margin")
    st.text_area("Notas", key="co_notes", height=80)

    with st.expander("Histórico financiero por año (editable, opcional)", expanded=bool(st.session_state.get("co_fin"))):
        fin_df = st.data_editor(_fin_to_editor(st.session_state.get("co_fin", {})), num_rows="dynamic", hide_index=True,
                                width="stretch", key=f"co_fin_editor_{st.session_state.get('co_editor_v', 0)}",
                                column_config={"Partida": st.column_config.SelectboxColumn("Partida", options=list(LABEL_ITEM), required=True)})
        if st.session_state.get("co_files"):
            st.caption("Archivos de origen: " + ", ".join(st.session_state["co_files"]) + " (solo se guardan las cifras, no los archivos).")

    st.subheader("3. Guardar")
    s1, s2, s3 = st.columns([1, 1, 1])
    s1.button("💾 Guardar", type="primary", on_click=co_save, args=(fin_df,), key="co_save_btn", width="stretch")
    s2.button("Guardar y cargar en el modelo", on_click=co_save, args=(fin_df, True), key="co_save_load_btn", width="stretch")
    if st.session_state.get("co_pick") != "__new__":
        s3.button("🗑️ Eliminar", on_click=co_delete, key="co_del_btn", width="stretch")

    if lib:
        st.subheader("Tu biblioteca")
        st.dataframe(pd.DataFrame([{
            "Empresa": c["name"], "Industria": c.get("industry"), "Etapa": c.get("stage"), "Moneda": c.get("currency"),
            "Ingresos": fmt_num((c.get("inputs") or {}).get("revenue") or 0, 0),
            "Crecimiento": fmt_pct((c.get("inputs") or {}).get("growth") or 0),
            "Margen operativo": fmt_pct((c.get("inputs") or {}).get("current_margin") or 0),
            "Actualizada": (c.get("updated_at") or "")[:10],
        } for c in lib]), hide_index=True, width="stretch")
        import json as _json
        st.download_button("Exportar biblioteca (JSON)", _json.dumps(lib, ensure_ascii=False, indent=1).encode("utf-8"),
                           "mis_empresas.json", "application/json", key="co_export",
                           help="Copia de seguridad de tus empresas.")

# ======================================================================= Fondos

EXAMPLE_FUND = pd.DataFrame({
    "date": pd.to_datetime(["2019-03-31", "2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31",
                            "2023-12-31", "2024-12-31", "2025-12-31"]),
    "capital_call": [20.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, 0.0],
    "distribution": [0.0, 0.0, 0.0, 5.0, 10.0, 25.0, 40.0, 30.0],
    "nav": [18.0, 40.0, 62.0, 85.0, 95.0, 92.0, 80.0, 70.0],
})

with T["Fondos"]:
    st.caption("Análisis de un fondo de VC desde el punto de vista del inversor (LP). Importes en la moneda base, "
               "en las unidades que prefieras (p. ej. millones).")
    up = st.file_uploader("Cargar flujos desde CSV (columnas: date, capital_call, distribution, nav)", type="csv", key="fund_csv")
    base_flows = EXAMPLE_FUND
    if up is not None:
        try:
            base_flows = pd.read_csv(up, parse_dates=["date"])[["date", "capital_call", "distribution", "nav"]]
        except Exception as e:  # noqa: BLE001: el usuario debe ver por qué no se pudo leer
            st.error(f"No se pudo leer el CSV: {e}")
    else:
        st.info("Datos de **ejemplo ilustrativo**. Edita la tabla o carga tu CSV.")
    flows = st.data_editor(
        base_flows, num_rows="dynamic", hide_index=True, width="stretch", key=f"fund_editor_{up.name if up else 'ej'}",
        column_config={
            "date": st.column_config.DateColumn("Fecha", required=True),
            "capital_call": st.column_config.NumberColumn("Capital llamado", min_value=0.0),
            "distribution": st.column_config.NumberColumn("Distribución", min_value=0.0),
            "nav": st.column_config.NumberColumn("Valor residual (NAV)", min_value=0.0),
        },
    ).dropna(subset=["date"])

    if len(flows) >= 2 and flows["capital_call"].fillna(0).sum() > 0:
        fm = fund_metrics(flows)
        c = st.columns(6)
        metric(c[0], "Capital desembolsado", fmt_num(fm.paid_in, 1))
        metric(c[1], "DPI", fmt_mult(fm.dpi), help="Distribuciones / capital desembolsado")
        metric(c[2], "RVPI", fmt_mult(fm.rvpi), help="Valor residual (NAV) / capital desembolsado")
        metric(c[3], "TVPI", fmt_mult(fm.tvpi), help="DPI + RVPI")
        metric(c[4], "MOIC", fmt_mult(fm.moic), help="(Distribuciones + NAV) / capital desembolsado")
        metric(c[5], "IRR (XIRR)", fmt_pct(fm.irr), help="Con fechas reales; el último NAV cuenta como valor terminal.")
        st.plotly_chart(ch.jcurve_chart(j_curve(flows), currency), width="stretch")
    else:
        st.warning("Introduce al menos dos fechas y alguna llamada de capital para calcular las métricas.")

    st.subheader("Proyección simple de flujos")
    st.caption("Modelo tipo Takahashi-Alexander (Yale): llamadas según un calendario y distribuciones que crecen con la edad del fondo.")
    p1, p2, p3, p4 = st.columns(4)
    commitment = p1.number_input("Compromiso", 1.0, 1e12, 100.0, 10.0, key="fund_commit")
    years_f = p2.slider("Vida del fondo (años)", 6, 15, 12, key="fund_years")
    growth_f = p3.number_input("Crecimiento anual del NAV (%)", -20.0, 50.0, 12.0, 1.0, key="fund_growth") / 100
    bow = p4.number_input("Factor de distribución (bow)", 0.5, 6.0, 2.5, 0.1, key="fund_bow",
                          help="Mayor = distribuciones más concentradas al final de la vida del fondo.")
    proj_f = project_cash_flows(commitment, years_f, growth=growth_f, bow=bow)
    proj_flows = pd.DataFrame({
        "date": pd.to_datetime([f"{2026 + y}-12-31" for y in proj_f["Año"]]),
        "capital_call": proj_f["Llamadas"], "distribution": proj_f["Distribuciones"], "nav": proj_f["NAV"],
    })
    pm_f = fund_metrics(proj_flows)
    c = st.columns(4)
    metric(c[0], "TVPI proyectado", fmt_mult(pm_f.tvpi))
    metric(c[1], "DPI proyectado", fmt_mult(pm_f.dpi))
    metric(c[2], "IRR proyectada", fmt_pct(pm_f.irr))
    metric(c[3], "NAV final", fmt_num(proj_f["NAV"].iloc[-1], 1))
    st.plotly_chart(ch.jcurve_chart(j_curve(proj_flows), currency), width="stretch")
    with st.expander("Tabla de la proyección"):
        st.dataframe(proj_f.round(2), hide_index=True, width="stretch")

    st.subheader("Contexto: tamaño de vehículos de VC (Form D)")
    funds = load_sec("sec_form_d_funds")
    sizes = funds["total_offering"].fillna(funds["total_sold"])
    sizes = sizes[sizes > 0]
    fund_size = st.number_input(f"Tamaño de tu fondo ({sym})", 0.0, None, 50_000_000.0, 1_000_000.0, format="%.0f", key="fund_size")
    k1, k2, k3 = st.columns(3)
    metric(k1, "Vehículos de VC en la muestra", f"{len(sizes):,}".replace(",", "."))
    metric(k2, "Mediana del tamaño", fmt_money(sizes.median() / usd_rate(currency), currency))
    metric(k3, "Percentil de tu fondo", fmt_pct(cmp.percentile_of(fund_size * usd_rate(currency), sizes), 0))
    st.plotly_chart(ch.log_histogram(sizes / usd_rate(currency), "Tamaño de los vehículos de VC que presentaron Form D",
                                     f"Tamaño ({sym}, escala logarítmica)", {"Tu fondo": fund_size}), width="stretch")
    st.caption("Incluye fondos y vehículos de una sola inversión (SPVs), por eso la mediana es baja. Es contexto de tamaño, "
               "no un benchmark de rentabilidad: no hay todavía una fuente verificada y redistribuible de retornos de fondos.")

# ======================================================================= pie

st.divider()
used_sources = sorted({r.source for r in resolver.used.values() if not r.missing})
foot = [f"**{source_names.get(s, s)}**: datos al {D['metrics'][D['metrics'].source == s]['as_of'].iloc[0]}" for s in used_sources]
foot.append(f"**BCE**: tipo de cambio y curva AAA al {fx_row['date']}")
foot.append("**Supuestos por etapa**: propios e ilustrativos")
st.caption(" · ".join(foot))
st.caption(
    "Herramienta educativa y de análisis. **No constituye asesoramiento de inversión.** "
    "Los resultados dependen de supuestos que el usuario debe revisar."
)
