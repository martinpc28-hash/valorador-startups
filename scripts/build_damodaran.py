"""Genera los CSV de Damodaran a partir de sus páginas HTML (solo EE. UU.).

No hace falta para ejecutar la app: documenta cómo se generaron los datos de `data/`.

Uso:
    python scripts/build_damodaran.py            # usa el HTML en .cache/damodaran si existe
    python scripts/build_damodaran.py --refresh  # vuelve a descargar las páginas

Salidas:
    data/industry_metrics.csv     (reemplaza solo las filas con source == "damodaran")
    data/size_class_metrics.csv   (clases de capitalización: mktcaprisk, mktcapmult)
    data/market_metrics.csv       (prima de riesgo implícita y bono del Tesoro, histimpl)
    data/damodaran_manifest.csv   (página, filas extraídas, hash del HTML)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning import normalize_name, parse_number  # noqa: E402

BASE_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/"
CACHE = ROOT / ".cache" / "damodaran"
DATA = ROOT / "data"

SOURCE = "damodaran"
REGION = "US"
AS_OF = "2026-01-09"  # última actualización completa publicada por Damodaran
LICENSE = (
    "Uso libre según las reglas de uso de Damodaran ('no strings attached', "
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datahistory.html#rules); "
    "solo agregados por industria; se cita la fuente"
)

# Por página: (encabezado esperado, métrica) para cada columna después del nombre.
# None = columna que se ignora porque ya se extrae de otra página.
INDUSTRY_PAGES: dict[str, list[tuple[str, str | None]]] = {
    "Betas": [
        ("Number of firms", "n_firms"),
        ("Beta", "beta"),
        ("D/E Ratio", "de_ratio"),
        ("Effective Tax rate", "tax_rate"),
        ("Unlevered beta", "unlevered_beta"),
        ("Cash/Firm value", "cash_to_firm_value"),
        ("Unlevered beta corrected for cash", "unlevered_beta_cash_adj"),
        ("HiLo Risk", "hilo_risk"),
        ("Standard deviation of equity", "sd_equity"),
        ("Standard deviation in operating income (last 10 years)", "sd_operating_income"),
    ],
    "totalbeta": [
        ("Number of firms", None),
        ("Average Unlevered Beta", None),
        ("Average Levered Beta", None),
        ("Average correlation with the market", "correlation_market"),
        ("Total Unlevered Beta", "total_unlevered_beta"),
        ("Total Levered Beta", "total_levered_beta"),
    ],
    "wacc": [
        ("Number of Firms", None),
        ("Beta", None),
        ("Cost of Equity", "cost_of_equity"),
        ("E/(D+E)", "equity_to_capital"),
        ("Std Dev in Stock", None),
        ("Cost of Debt", "cost_of_debt"),
        ("Tax Rate", None),
        ("After-tax Cost of Debt", "after_tax_cost_of_debt"),
        ("D/(D+E)", "debt_to_capital"),
        ("Cost of Capital", "cost_of_capital"),
    ],
    "margin": [
        ("Number of firms", None),
        ("Gross Margin", "gross_margin"),
        ("Net Margin", "net_margin"),
        ("Pre-tax, Pre-stock compensation Operating Margin", "operating_margin_pre_sbc"),
        ("Pre-tax Unadjusted Operating Margin", "operating_margin_unadj"),
        ("After-tax Unadjusted Operating Margin", "after_tax_operating_margin_unadj"),
        ("Pre-tax Lease adjusted Margin", "operating_margin"),
        ("After-tax Lease Adjusted Margin", "after_tax_operating_margin"),
        ("Pre-tax Lease & R&D adj Margin", "operating_margin_lease_rd_adj"),
        ("After-tax Lease & R&D adj Margin", "after_tax_operating_margin_lease_rd_adj"),
        ("EBITDA/Sales", "ebitda_margin"),
        ("EBITDASG&A/Sales", "ebitdasga_to_sales"),
        ("EBITDAR&D/Sales", "ebitdard_to_sales"),
        ("COGS/Sales", "cogs_to_sales"),
        ("R&D/Sales", "rd_to_sales"),
        ("SG&A/ Sales", "sga_to_sales"),
        ("Stock-Based Compensation/Sales", "sbc_to_sales"),
        ("Lease Expense/Sales", "lease_to_sales"),
    ],
    "psdata": [
        ("Number of firms", None),
        ("Price/Sales", "price_to_sales"),
        ("Net Margin", None),
        ("EV/Sales", "ev_sales"),
        ("Pre-tax Operating Margin", None),
    ],
    "vebitda": [
        ("Number of firms", None),
        ("EV/EBITDAR&D", "ev_ebitdard_pos"),
        ("EV/EBITDA", "ev_ebitda_pos"),
        ("EV/EBIT", "ev_ebit_pos"),
        ("EV/EBIT (1-t)", "ev_ebit_after_tax_pos"),
        ("EV/EBITDAR&D", "ev_ebitdard_all"),
        ("EV/EBITDA", "ev_ebitda_all"),
        ("EV/EBIT", "ev_ebit_all"),
        ("EV/EBIT (1-t)", "ev_ebit_after_tax_all"),
    ],
    "histgr": [
        ("Number of Firms", None),
        ("CAGR in Net Income- Last 5 years", "cagr_net_income_5y"),
        ("CAGR in Revenues- Last 5 years", "cagr_revenue_5y"),
        ("Expected Growth in Revenues - Next 2 years", "exp_revenue_growth_2y"),
        ("Expected Growth in Revenues - Next 5 years", "exp_revenue_growth_5y"),
        ("Expected Growth in EPS - Next 5 years", "exp_eps_growth_5y"),
    ],
    "fundgrEB": [
        ("Number of Firms", None),
        ("ROC", "roc"),
        ("Reinvestment Rate", "reinvestment_rate"),
        ("Expected Growth in EBIT", "fundamental_ebit_growth"),
    ],
    "capex": [
        ("Number of Firms", None),
        ("Capital Expenditures (US $ millions)", "capex_usd_m"),
        ("Depreciation & Amort ((US $ millions)", "da_usd_m"),
        ("Cap Ex/Deprecn", "capex_to_da"),
        ("Acquisitions (US $ millions)", "acquisitions_usd_m"),
        ("Net R&D (US $ millions)", "net_rd_usd_m"),
        ("Net Cap Ex/Sales", "net_capex_to_sales"),
        ("Net Cap Ex/ EBIT (1-t)", "net_capex_to_ebit_after_tax"),
        ("Sales/ Invested Capital (LTM)", "sales_to_capital"),
    ],
}

SIZE_PAGES: dict[str, list[tuple[str, str | None]]] = {
    "mktcaprisk": [
        ("Number of firms", "n_firms"),
        ("Aggregate Market Cap", "aggregate_market_cap_usd_m"),
        ("Cash/Firm Value", "cash_to_firm_value"),
        ("Market Debt to capital ratio (median)", "debt_to_capital_median"),
        ("Median Beta", "beta_median"),
        ("Correlation with market (median)", "correlation_market_median"),
        ("Standard deviation in stock price (median)", "sd_equity_median"),
        ("Total Beta", "total_beta"),
        ("HiL0 Risk Measure (Hi- lo)/ (Hi+Lo) (median)", "hilo_risk_median"),
        ("Interest Coverage Ratio", "interest_coverage"),
    ],
    "mktcapmult": [
        ("Number of firms", None),
        ("Trailing PE", "pe_trailing"),
        ("Forward PE", "pe_forward"),
        ("PEG Ratio", "peg"),
        ("PBV", "pbv"),
        ("Price/Sales", "price_to_sales"),
        ("EV/EBIT", "ev_ebit"),
        ("EV/EBITDA", "ev_ebitda"),
        ("EV/Sales", "ev_sales"),
        ("EV/Invested Capital", "ev_invested_capital"),
        ("ROE", "roe"),
        ("Pre-tax ROIC", "roic_pretax"),
        ("Net Margin", "net_margin"),
        ("Operating Margin", "operating_margin"),
        ("% of companies with Net Income <0", "share_net_loss"),
        ("% of companies with Operating Income <0", "share_operating_loss"),
    ],
}

SIZE_UNITS = {
    "n_firms": "count",
    "aggregate_market_cap_usd_m": "usd_millions",
    "beta_median": "beta",
    "total_beta": "beta",
    "hilo_risk_median": "score",
    "interest_coverage": "multiple",
    "pe_trailing": "multiple",
    "pe_forward": "multiple",
    "peg": "multiple",
    "pbv": "multiple",
    "price_to_sales": "multiple",
    "ev_ebit": "multiple",
    "ev_ebitda": "multiple",
    "ev_sales": "multiple",
    "ev_invested_capital": "multiple",
}

SIZE_ORDER = [
    "Bottom decile", "2nd decile", "3rd decile", "4th decile", "5th decile",
    "6th decile", "7th decile", "8th decile", "9th decile", "Top decile", "All firms",
]


def fetch(page: str, refresh: bool) -> tuple[bytes, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{page}.html"
    if refresh or not path.exists():
        resp = requests.get(
            BASE_URL + f"{page}.html",
            headers={"User-Agent": "Mozilla/5.0 (valorador-startups; research)"},
            timeout=60,
        )
        resp.raise_for_status()
        path.write_bytes(resp.content)
    retrieved = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return path.read_bytes(), retrieved


def cell_text(cell) -> str:
    # get_text() sin separador: algunas celdas parten una palabra con etiquetas internas.
    return normalize_name(cell.get_text())


def parse_table(raw: bytes, columns: list[tuple[str, str | None]], page: str) -> list[tuple[str, dict[str, float]]]:
    """Devuelve [(nombre_original, {métrica: valor})] validando cada encabezado."""
    soup = BeautifulSoup(raw.decode("mac_roman", errors="replace"), "lxml")
    tables = soup.find_all("table")
    if len(tables) != 1:
        raise ValueError(f"{page}: se esperaba 1 tabla, hay {len(tables)}")
    rows = [[cell_text(c) for c in r.find_all(["td", "th"])] for r in tables[0].find_all("tr")]

    expected = [h for h, _ in columns]
    header_idx = next(
        (i for i, r in enumerate(rows) if len(r) > 1 and r[1].lower() == expected[0].lower()),
        None,
    )
    if header_idx is None:
        raise ValueError(f"{page}: no se encontró la fila de encabezado")
    header = rows[header_idx][1 : 1 + len(columns)]
    for got, want in zip(header, expected):
        if got.lower() != want.lower():
            raise ValueError(f"{page}: encabezado inesperado {got!r}, se esperaba {want!r}")

    out = []
    for r in rows[header_idx + 1 :]:
        if not r or not r[0] or not any(r[1:]):
            continue
        if len(r) < 1 + len(columns):
            raise ValueError(f"{page}: fila corta {r!r}")
        values = {
            metric: parse_number(r[1 + i])
            for i, (_, metric) in enumerate(columns)
            if metric is not None
        }
        out.append((r[0], values))
    return out


def build(refresh: bool) -> None:
    crosswalk = pd.read_csv(DATA / "industry_crosswalk.csv")
    xw = crosswalk[crosswalk["source"] == SOURCE].set_index("industry_original")["industry_std"].to_dict()
    units = pd.read_csv(DATA / "metric_definitions.csv").set_index("metric")["unit"].to_dict()

    records, manifest = [], []
    unmapped: set[str] = set()

    for page, columns in INDUSTRY_PAGES.items():
        raw, retrieved = fetch(page, refresh)
        url = BASE_URL + f"{page}.html"
        rows = parse_table(raw, columns, page)
        for original, values in rows:
            if original not in xw:
                unmapped.add(original)
                continue
            for metric, value in values.items():
                records.append({
                    "source": SOURCE, "region": REGION, "industry_std": xw[original],
                    "industry_original": original, "metric": metric, "value": value,
                    "unit": units[metric], "as_of": AS_OF, "retrieved_at": retrieved,
                    "url": url, "license": LICENSE,
                })
        manifest.append({
            "page": page, "url": url, "kind": "industry", "n_rows": len(rows),
            "n_industries": sum(1 for o, _ in rows if not o.lower().startswith("total market")),
            "as_of": AS_OF, "retrieved_at": retrieved, "sha256": hashlib.sha256(raw).hexdigest(),
        })

    if unmapped:
        raise SystemExit(
            "Industrias sin equivalencia en data/industry_crosswalk.csv:\n  " + "\n  ".join(sorted(unmapped))
        )

    new = pd.DataFrame(records)
    target = DATA / "industry_metrics.csv"
    if target.exists():
        old = pd.read_csv(target)
        new = pd.concat([old[old["source"] != SOURCE], new], ignore_index=True)
    new.to_csv(target, index=False)

    # Clases de capitalización
    size_records = []
    for page, columns in SIZE_PAGES.items():
        raw, retrieved = fetch(page, refresh)
        url = BASE_URL + f"{page}.html"
        rows = parse_table(raw, columns, page)
        names = [o for o, _ in rows]
        if names != SIZE_ORDER:
            raise ValueError(f"{page}: clases inesperadas {names}")
        for original, values in rows:
            for metric, value in values.items():
                size_records.append({
                    "source": SOURCE, "region": REGION, "size_class": original,
                    "size_rank": SIZE_ORDER.index(original) + 1, "metric": metric, "value": value,
                    "unit": SIZE_UNITS.get(metric, "decimal"), "as_of": AS_OF,
                    "retrieved_at": retrieved, "url": url, "license": LICENSE,
                })
        manifest.append({
            "page": page, "url": url, "kind": "size_class", "n_rows": len(rows), "n_industries": 0,
            "as_of": AS_OF, "retrieved_at": retrieved, "sha256": hashlib.sha256(raw).hexdigest(),
        })
    pd.DataFrame(size_records).to_csv(DATA / "size_class_metrics.csv", index=False)

    # Prima de riesgo implícita (histimpl): serie anual
    raw, retrieved = fetch("histimpl", refresh)
    url = BASE_URL + "histimpl.html"
    cols = [
        ("Earnings Yield", "earnings_yield"), ("Dividend Yield", "dividend_yield"),
        ("S&P 500", "sp500_level"), ("Earnings*", "sp500_earnings"), ("Dividends*", "sp500_dividends"),
        ("T.Bond Rate", "us_tbond_10y"), ("Smoothed Growth", "smoothed_growth"),
        ("Implied ERP (FCFE)", "implied_erp_fcfe"),
    ]
    soup = BeautifulSoup(raw.decode("mac_roman", errors="replace"), "lxml")
    rows = [[cell_text(c) for c in r.find_all(["td", "th"])] for r in soup.find("table").find_all("tr")]
    header = rows[0]
    if header[0] != "Year" or [h for h, _ in cols] != header[1 : 1 + len(cols)]:
        raise ValueError(f"histimpl: encabezado inesperado {header}")
    market_records = []
    for r in rows[1:]:
        if not r or not r[0].isdigit():
            continue
        year = int(r[0])
        for i, (_, metric) in enumerate(cols):
            market_records.append({
                "source": SOURCE, "region": REGION, "series": metric,
                # La fila del año Y refleja el mercado al cierre de Y (inicio de Y+1)
                "date": f"{year}-12-31", "value": parse_number(r[1 + i]),
                "unit": "index" if metric.startswith("sp500") else "decimal",
                "as_of": AS_OF, "retrieved_at": retrieved, "url": url, "license": LICENSE,
            })
    market = pd.DataFrame(market_records)
    target = DATA / "market_metrics.csv"
    if target.exists():
        old = pd.read_csv(target)
        market = pd.concat([old[old["source"] != SOURCE], market], ignore_index=True)
    market.to_csv(target, index=False)
    manifest.append({
        "page": "histimpl", "url": url, "kind": "market", "n_rows": int(market[market.source == SOURCE].date.nunique()),
        "n_industries": 0, "as_of": AS_OF, "retrieved_at": retrieved, "sha256": hashlib.sha256(raw).hexdigest(),
    })

    pd.DataFrame(manifest).to_csv(DATA / "damodaran_manifest.csv", index=False)
    print(f"industry_metrics: {len(records)} filas de Damodaran; size_class: {len(size_records)}; market: {len(market_records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="volver a descargar las páginas")
    build(parser.parse_args().refresh)
