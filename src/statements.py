"""Lectura de estados financieros subidos por el usuario (Excel, CSV, PDF).

Detecta partidas (ingresos, EBIT, beneficio neto, caja, deuda...) por su nombre en español o
inglés, la fila de años y la escala ("en miles", "in millions"). El resultado siempre se
revisa en la app antes de guardarlo: la detección es heurística.
"""

from __future__ import annotations

import io
import math
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Partida -> (etiqueta, patrones que la identifican, patrones que la excluyen)
ITEMS: dict[str, tuple[str, list[str], list[str]]] = {
    "revenue": ("Ingresos", [
        r"importe neto de la cifra de negocios", r"cifra de negocios", r"^ingresos( totales| de explotacion| ordinarios| por ventas)?$",
        r"^ventas( netas)?$", r"^(total )?revenues?$", r"^net (sales|revenues?)$", r"^total (net )?revenues?$", r"^sales$",
        r"^revenues? from contracts? with customers", r"^ingresos totales", r"^revenue",
    ], [r"otros ingresos", r"other (income|revenue)", r"financier", r"interest", r"diferid", r"deferred", r"cost of", r"costo", r"coste",
        r"growth", r"crecimiento", r"%", r"per (share|employee)"]),
    "cogs": ("Costo de ventas", [
        r"^(costo|coste) de (las )?ventas", r"^aprovisionamientos", r"^cost of (goods sold|sales|revenues?)", r"^cogs$",
        r"^costo de los bienes vendidos",
    ], []),
    # "Resultado bruto de explotación" (PGC español) es EBITDA, no beneficio bruto
    "gross_profit": ("Beneficio bruto", [r"^(beneficio|margen|resultado) bruto", r"^gross (profit|margin)"], [r"explotacion", r"%"]),
    "ebitda": ("EBITDA", [r"^ebitda", r"resultado bruto de explotacion"], [r"margen ebitda", r"ebitda margin", r"%"]),
    "depreciation": ("Amortización y depreciación", [
        r"^amortizacion del inmovilizado", r"^(dotacion a la )?amortizacion", r"^depreciation( and amortization)?",
        r"^d&a$", r"^amortizaciones",
    ], [r"acumulada", r"accumulated"]),
    "operating_income": ("Resultado operativo (EBIT)", [
        r"^resultado de explotacion", r"^resultado operativo", r"^beneficio operativo", r"^(beneficio|perdida) de explotacion",
        r"^ebit$", r"^operating (income|profit|loss)", r"^(income|loss) from operations", r"^total operating income",
    ], [r"margen", r"margin", r"antes de amortiz"]),
    "net_income": ("Beneficio neto", [
        r"^resultado del ejercicio", r"^beneficio neto", r"^resultado neto", r"^(beneficio|perdida) del ejercicio",
        r"^net (income|loss|profit|earnings)", r"^net income \(loss\)", r"^resultado despues de impuestos",
        r"^profit (for the year|after tax)",
    ], [r"por accion", r"per share", r"margen", r"margin", r"atribuible a minorit", r"non-?controlling", r"operaciones interrumpidas"]),
    "cash": ("Caja", [
        r"^efectivo y otros activos liquidos", r"^efectivo y equivalentes", r"^tesoreria$", r"^cash and cash equivalents",
        r"^cash$", r"^caja( y bancos)?$", r"^efectivo$",
    ], [r"flujo", r"flow", r"variacion", r"aumento", r"increase", r"restringid", r"restricted"]),
    "short_term_debt": ("Deuda a corto plazo", [
        r"^deudas? (financieras? )?a corto plazo", r"^deudas con entidades de credito a corto", r"^short[- ]term (debt|borrowings)",
        r"^current portion of long[- ]term debt",
    ], []),
    "long_term_debt": ("Deuda a largo plazo", [
        r"^deudas? (financieras? )?a largo plazo", r"^deudas con entidades de credito a largo", r"^long[- ]term (debt|borrowings)",
        r"^long[- ]term debt, net",
    ], [r"current portion"]),
    "total_debt": ("Deuda financiera total", [r"^deuda financiera( total| bruta)?$", r"^total (debt|borrowings)$", r"^deuda total$"], []),
    "total_assets": ("Activo total", [r"^(total )?activo( total)?$", r"^total assets$", r"^total activo"], [r"corriente", r"current"]),
    "equity": ("Patrimonio neto", [r"^patrimonio neto( total)?$", r"^fondos propios$", r"^total (stockholders'?|shareholders'?) equity", r"^total equity$"], [r"pasivo", r"liabilities"]),
    "operating_cash_flow": ("Flujo de caja operativo", [
        r"flujos? de efectivo de las actividades de explotacion", r"^flujo de caja operativo", r"^net cash (provided by|used in|from) operating",
        r"^cash flows? from operating activities",
    ], []),
    "capex": ("Capex", [r"^(pagos por )?inversiones en inmovilizado", r"^capex$", r"^capital expenditures", r"^purchases? of property"], []),
    "employees": ("Empleados", [r"^(numero de )?empleados", r"^plantilla( media)?$", r"^(number of )?employees", r"^headcount", r"^fte"], []),
}
ITEM_ORDER = list(ITEMS)
COUNT_ITEMS = {"employees"}

SCALE_PATTERNS = [
    (r"en millones|in millions|millones de|\(m(€|\$|eur|usd)\)|€ ?m\b|\$ ?m\b|usd ?m\b", 1e6),
    (r"en miles|in thousands|miles de|\(k?(€|\$)000\)|thousands of|en k€|\(000\)|'000", 1e3),
]


def normalize(text: object) -> str:
    """minúsculas, sin acentos, sin espacios dobles ni numeración inicial ('1. Ventas' -> 'ventas')."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"^\s*([a-z]\)|[ivx]+\.|\d+(\.\d+)*[.)]?|[-–•*])\s+", "", s)
    s = re.sub(r"[:…]+$", "", s)
    return " ".join(s.replace("\xa0", " ").split()).strip(" .-")


def match_item(label: str) -> tuple[str, int] | None:
    """Devuelve (partida, prioridad) o None. Prioridad menor = patrón más específico."""
    norm = normalize(label)
    if not norm or len(norm) > 120:
        return None
    for key, (_, patterns, excludes) in ITEMS.items():
        if any(re.search(x, norm) for x in excludes):
            continue
        for i, p in enumerate(patterns):
            if re.search(p, norm):
                return key, i
    return None


def parse_amount(raw: object) -> float:
    """'1.234.567,89' / '1,234,567.89' / '(1.234)' / '-' -> float. Grupos de 3 dígitos = miles."""
    if raw is None:
        return math.nan
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        return float(raw)
    s = str(raw).strip().replace("\xa0", " ")
    if s in {"", "-", "–", "—", "n/a", "na", "nm", "n.a."}:
        return math.nan
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("−") or s.endswith("-")
    s = re.sub(r"[^\d.,]", "", s)
    if not s or not re.search(r"\d", s):
        return math.nan
    if "." in s and "," in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            s = s.replace(sep, "")  # separador de miles
        else:
            s = s.replace(sep, ".")
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return -v if neg else v


def parse_year(cell: object) -> int | None:
    if isinstance(cell, (pd.Timestamp,)) or hasattr(cell, "year") and not isinstance(cell, (int, float, str)):
        y = getattr(cell, "year", None)
        return y if y and 1990 <= y <= 2100 else None
    if isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(cell, bool):
        if not math.isnan(float(cell)) and float(cell).is_integer() and 1990 <= int(cell) <= 2100:
            return int(cell)
        return None
    m = re.fullmatch(r"\s*(?:fy|ej\.?|ejercicio|año|ano|year|a)?\s*'?(\d{4})\s*(?:[a-z]{1,3})?\s*", normalize(cell))
    if m and 1990 <= int(m[1]) <= 2100:
        return int(m[1])
    m = re.fullmatch(r"\s*(?:\d{1,2}[/.-]\d{1,2}[/.-](\d{4})|(\d{4})[/.-]\d{1,2}[/.-]\d{1,2})\s*", str(cell))
    if m:
        y = int(m[1] or m[2])
        return y if 1990 <= y <= 2100 else None
    return None


def detect_scale(text: str) -> float:
    t = normalize(text)
    for pat, mult in SCALE_PATTERNS:
        if re.search(pat, t):
            return mult
    return 1.0


@dataclass
class Detection:
    item: str
    year: int | str
    value: float
    label: str       # texto original de la partida en el archivo
    where: str       # hoja/página y fila
    priority: int


@dataclass
class StatementResult:
    table: pd.DataFrame                # partidas × años (valores ya escalados)
    detections: list[Detection]
    scale: float
    currency: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def found(self) -> list[str]:
        return [i for i in ITEM_ORDER if i in self.table.index]


# ---------------------------------------------------------------- tablas


def _scan_grid(grid: pd.DataFrame, where: str) -> tuple[list[Detection], str]:
    """Busca la fila de años y las partidas en una cuadrícula sin encabezados."""
    grid = grid.dropna(how="all").dropna(axis=1, how="all")
    if grid.empty:
        return [], ""
    text = " ".join(str(v) for v in grid.values.ravel() if isinstance(v, str))

    header_row, year_cols = None, {}
    for ridx, row in grid.head(40).iterrows():
        yc = {c: parse_year(v) for c, v in row.items()}
        yc = {c: y for c, y in yc.items() if y}
        if len(yc) >= 1 and len(set(yc.values())) == len(yc):
            header_row, year_cols = ridx, yc
            if len(yc) >= 2:
                break

    out: list[Detection] = []
    cols = list(grid.columns)
    for ridx, row in grid.iterrows():
        if ridx == header_row:
            continue
        label_cell = next(((c, v) for c, v in row.items() if isinstance(v, str) and re.search(r"[A-Za-zÁ-ú]", v)), None)
        if not label_cell:
            continue
        m = match_item(label_cell[1])
        if not m:
            continue
        item, prio = m
        start = cols.index(label_cell[0]) + 1
        if year_cols:
            pairs = [(year_cols[c], row[c]) for c in cols[start:] if c in year_cols]
        else:
            first = next((row[c] for c in cols[start:] if not math.isnan(parse_amount(row[c]))), None)
            pairs = [("actual", first)] if first is not None else []
        for year, raw in pairs:
            v = parse_amount(raw)
            if not math.isnan(v):
                out.append(Detection(item, year, v, str(label_cell[1]).strip(), f"{where}, fila {ridx + 1}", prio))
    return out, text


def _read_tabular(data: bytes, filename: str) -> list[tuple[pd.DataFrame, str]]:
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine="openpyxl")
        return [(df, f"hoja «{s}»") for s, df in sheets.items()]
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=str, engine="python", on_bad_lines="skip")
    return [(df, "CSV")]


# Sin espacio como separador de miles: en texto de PDF "400.000 900.000" son dos importes
NUM_TOKEN = r"\(?[-−–]?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\)?-?|\(?[-−–]?\d+(?:[.,]\d+)?\)?-?"


def _read_pdf(data: bytes) -> list[tuple[pd.DataFrame, str]]:
    import pdfplumber

    grids = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for t in page.extract_tables() or []:
                if t and len(t) > 1:
                    grids.append((pd.DataFrame(t), f"página {pno} (tabla)"))
            # Líneas de texto "Partida ..... 1.234 987": útil cuando el PDF no tiene tablas reales
            rows = []
            for line in (page.extract_text() or "").splitlines():
                nums = re.findall(NUM_TOKEN + r"(?=\s|$)", line)
                years = [parse_year(n) for n in re.findall(r"\b(?:19|20)\d{2}\b", line)]
                label = re.split(r"\s(?=" + NUM_TOKEN + r"(?:\s|$))", line, maxsplit=1)[0].strip()
                if years and len(years) >= 2 and len(re.sub(r"[\d\s/.-]", "", line)) < 15:
                    rows.append([""] + [str(y) for y in years])
                elif nums and re.search(r"[A-Za-zÁ-ú]{3}", label):
                    rows.append([label] + [n.strip() for n in nums][-6:])
            if rows:
                width = max(len(r) for r in rows)
                grids.append((pd.DataFrame([r + [None] * (width - len(r)) for r in rows]), f"página {pno} (texto)"))
    return grids


def parse_statement(data: bytes, filename: str) -> StatementResult:
    name = filename.lower()
    grids = _read_pdf(data) if name.endswith(".pdf") else _read_tabular(data, filename)

    detections, texts, warnings = [], [], []
    for grid, where in grids:
        d, text = _scan_grid(grid, where)
        detections += d
        texts.append(text)
    all_text = " ".join(texts)
    scale = detect_scale(all_text)
    currency = ("EUR" if re.search(r"€|\beur\b|euros?", all_text, re.I)
                else "USD" if re.search(r"\$|\busd\b|dollars?|dólares", all_text, re.I) else None)

    # Por partida y año, nos quedamos con el patrón más específico (y el primero que aparezca)
    best: dict[tuple[str, object], Detection] = {}
    for d in detections:
        k = (d.item, d.year)
        if k not in best or d.priority < best[k].priority:
            best[k] = d
    kept = list(best.values())
    if not kept:
        warnings.append("No se reconoció ninguna partida. Revisa que el archivo tenga los nombres de las partidas y una fila de años.")
        return StatementResult(pd.DataFrame(), [], scale, currency, warnings)

    years = sorted({d.year for d in kept}, key=lambda y: (isinstance(y, str), y))
    table = pd.DataFrame(index=[i for i in ITEM_ORDER if any(d.item == i for d in kept)], columns=years, dtype=float)
    for d in kept:
        table.loc[d.item, d.year] = d.value * (1 if d.item in COUNT_ITEMS else scale)
    if scale != 1:
        warnings.append(f"Importes multiplicados por {scale:,.0f} (el archivo indica que están en {'miles' if scale == 1e3 else 'millones'}).".replace(",", "."))
    if "revenue" not in table.index:
        warnings.append("No se encontraron los ingresos: introdúcelos a mano.")
    return StatementResult(table, kept, scale, currency, warnings)


def merge_results(results: list[StatementResult]) -> pd.DataFrame:
    """Combina varios archivos (p. ej. cuenta de resultados y balance). El primero manda si hay solapes."""
    tables = [r.table for r in results if not r.table.empty]
    if not tables:
        return pd.DataFrame()
    out = tables[0]
    for t in tables[1:]:
        out = out.combine_first(t)
    order = [i for i in ITEM_ORDER if i in out.index]
    cols = sorted(out.columns, key=lambda y: (isinstance(y, str), y))
    return out.loc[order, cols]


# ---------------------------------------------------------------- resumen para el modelo


def summarize(table: pd.DataFrame) -> dict[str, float]:
    """Entradas del modelo a partir del último año con ingresos (decimales para tasas)."""
    if table.empty:
        return {}
    t = table.copy()
    years = [c for c in t.columns if not isinstance(c, str)] or list(t.columns)
    get = lambda item, y: float(t.loc[item, y]) if item in t.index and pd.notna(t.loc[item, y]) else math.nan  # noqa: E731
    with_rev = [y for y in years if not math.isnan(get("revenue", y))]
    last = with_rev[-1] if with_rev else years[-1]
    prev = with_rev[-2] if len(with_rev) >= 2 else None
    out: dict[str, float] = {"year": last}

    rev = get("revenue", last)
    if not math.isnan(rev):
        out["revenue"] = rev
    if prev is not None and get("revenue", prev) > 0:
        out["growth"] = rev / get("revenue", prev) - 1
    ebit = get("operating_income", last)
    if math.isnan(ebit) and not math.isnan(get("ebitda", last)):
        ebit = get("ebitda", last) - abs(get("depreciation", last)) if not math.isnan(get("depreciation", last)) else math.nan
    if rev and rev > 0 and not math.isnan(ebit):
        out["current_margin"] = ebit / rev
    ni = get("net_income", last)
    if rev and rev > 0 and not math.isnan(ni):
        out["net_margin"] = ni / rev
    cash = get("cash", last)
    if not math.isnan(cash):
        out["cash"] = cash
    debt = get("total_debt", last)
    if math.isnan(debt):
        parts = [get("short_term_debt", last), get("long_term_debt", last)]
        debt = sum(p for p in parts if not math.isnan(p)) if any(not math.isnan(p) for p in parts) else math.nan
    if not math.isnan(debt):
        out["debt"] = abs(debt)
    # Burn mensual: flujo operativo negativo; si no hay, beneficio neto + amortización
    ocf = get("operating_cash_flow", last)
    if math.isnan(ocf) and not math.isnan(ni):
        da = get("depreciation", last)
        ocf = ni + (abs(da) if not math.isnan(da) else 0.0)
    if not math.isnan(ocf):
        out["burn"] = max(-ocf, 0.0) / 12.0
    emp = get("employees", last)
    if not math.isnan(emp):
        out["employees"] = emp
    # Pérdidas acumuladas aproximadas: suma de resultados netos negativos de los años disponibles
    losses = [get("net_income", y) for y in years if get("net_income", y) < 0]
    if losses:
        out["nol"] = -sum(losses)
    return out


TEMPLATE = pd.DataFrame({
    "Partida": [ITEMS[i][0] for i in ITEM_ORDER],
    "2023": [None] * len(ITEM_ORDER),
    "2024": [None] * len(ITEM_ORDER),
    "2025": [None] * len(ITEM_ORDER),
})
