"""Lectura de estados financieros: Excel (PGC español, en miles), CSV (inglés) y PDF."""

import io
import math

import pandas as pd
import pytest

from src.statements import TEMPLATE, match_item, merge_results, parse_amount, parse_statement, parse_year, summarize


@pytest.mark.parametrize("raw, expected", [
    ("1.234.567,89", 1234567.89), ("1,234,567.89", 1234567.89), ("(1.234)", -1234.0), ("-850", -850.0),
    ("12,5", 12.5), ("1.234", 1234.0), ("1 234 567", 1234567.0), ("—", math.nan), (3.5, 3.5), ("850-", -850.0),
])
def test_parse_amount(raw, expected):
    v = parse_amount(raw)
    assert (math.isnan(v) and math.isnan(expected)) or v == pytest.approx(expected)


@pytest.mark.parametrize("cell, year", [(2024, 2024), ("FY2025", 2025), ("31/12/2023", 2023), ("Ejercicio 2022", 2022),
                                        ("2024-12-31", 2024), ("Ventas", None), (1500000, None)])
def test_parse_year(cell, year):
    assert parse_year(cell) == year


@pytest.mark.parametrize("label, item", [
    ("1. Importe neto de la cifra de negocios", "revenue"), ("Total revenues", "revenue"), ("Otros ingresos de explotación", None),
    ("Revenue growth", None), ("A) RESULTADO DE EXPLOTACIÓN", "operating_income"), ("Resultado bruto de explotación", "ebitda"),
    ("Resultado del ejercicio", "net_income"), ("Net loss per share", None), ("Efectivo y otros activos líquidos equivalentes", "cash"),
    ("Cash flows from operating activities", "operating_cash_flow"), ("Deudas con entidades de crédito a largo plazo", "long_term_debt"),
    ("Gross profit", "gross_profit"), ("Número de empleados", "employees"),
])
def test_match_item(label, item):
    m = match_item(label)
    assert (m[0] if m else None) == item


def _xlsx_spanish() -> bytes:
    rows = [
        ["Cuenta de pérdidas y ganancias (en miles de euros)", None, None],
        [None, "2024", "2025"],
        ["1. Importe neto de la cifra de negocios", "1.200", "2.100"],
        ["Otros ingresos de explotación", "50", "80"],
        ["4. Aprovisionamientos", "(400)", "(650)"],
        ["8. Amortización del inmovilizado", "(30)", "(45)"],
        ["A) RESULTADO DE EXPLOTACIÓN", "(900)", "(600)"],
        ["Resultado del ejercicio", "(950)", "(640)"],
    ]
    balance = [
        ["Balance (en miles de euros)", None, None],
        [None, 2024, 2025],
        ["Efectivo y otros activos líquidos equivalentes", 800, 1500],
        ["Deudas con entidades de crédito a largo plazo", 0, 500],
        ["Número de empleados", 12, 18],
    ]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="PyG", header=False, index=False)
        pd.DataFrame(balance).to_excel(w, sheet_name="Balance", header=False, index=False)
    return buf.getvalue()


def test_excel_spanish_pgc_in_thousands():
    r = parse_statement(_xlsx_spanish(), "cuentas.xlsx")
    t = r.table
    assert r.scale == 1e3 and r.currency == "EUR"
    assert t.loc["revenue", 2025] == 2_100_000
    assert t.loc["operating_income", 2025] == -600_000
    assert t.loc["cash", 2025] == 1_500_000
    assert t.loc["employees", 2025] == 18  # los conteos no se escalan
    assert "Otros ingresos" not in " ".join(d.label for d in r.detections if d.item == "revenue")
    s = summarize(t)
    assert s["year"] == 2025
    assert s["revenue"] == 2_100_000
    assert s["growth"] == pytest.approx(2100 / 1200 - 1)
    assert s["current_margin"] == pytest.approx(-600 / 2100)
    assert s["debt"] == 500_000
    assert s["burn"] == pytest.approx((640_000 - 45_000) / 12)
    assert s["nol"] == pytest.approx(1_590_000)


def test_csv_english_with_dollar_signs():
    csv = ("Income statement (USD),FY2024,FY2025\n"
           "Total revenues,\"$1,000,000\",\"$1,800,000\"\n"
           "Cost of revenues,\"(300,000)\",\"(500,000)\"\n"
           "Operating loss,\"(700,000)\",\"(400,000)\"\n"
           "Net loss,\"(720,000)\",\"(410,000)\"\n"
           "Cash and cash equivalents,\"2,000,000\",\"3,500,000\"\n").encode()
    r = parse_statement(csv, "pyl.csv")
    assert r.currency == "USD" and r.scale == 1
    s = summarize(r.table)
    assert s["revenue"] == 1_800_000
    assert s["growth"] == pytest.approx(0.8)
    assert s["current_margin"] == pytest.approx(-400 / 1800)
    assert s["cash"] == 3_500_000


def test_pdf_text_lines():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in ["Cuenta de resultados (euros)", "2024 2025", "Importe neto de la cifra de negocios 850.000 1.400.000",
                 "Resultado de explotacion (300.000) (150.000)", "Resultado del ejercicio (320.000) (170.000)",
                 "Tesoreria 400.000 900.000"]:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    r = parse_statement(bytes(pdf.output()), "cuentas.pdf")
    t = r.table
    assert t.loc["revenue", 2025] == 1_400_000
    assert t.loc["operating_income", 2024] == -300_000
    assert t.loc["cash", 2025] == 900_000


def test_template_roundtrip_and_merge():
    t = TEMPLATE.copy()
    t.loc[t["Partida"] == "Ingresos", ["2024", "2025"]] = [100, 150]
    t.loc[t["Partida"] == "Caja", ["2025"]] = [80]
    r1 = parse_statement(t.to_csv(index=False).encode(), "plantilla.csv")
    r2 = parse_statement(b"Partida,2025\nEmpleados,7\n", "extra.csv")
    merged = merge_results([r1, r2])
    assert merged.loc["revenue", 2025] == 150 and merged.loc["employees", 2025] == 7


def test_nothing_found_warns():
    r = parse_statement(b"a,b\nhola,adios\n", "x.csv")
    assert r.table.empty and r.warnings
