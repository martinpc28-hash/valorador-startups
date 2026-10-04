"""Genera el listado de sociedades españolas a partir del BORME (Sección Primera, actos inscritos).

No hace falta para ejecutar la app: documenta cómo se generaron los datos de `data/`.

Fuente: API de datos abiertos del BOE (https://www.boe.es/datosabiertos/api/borme/sumario/AAAAMMDD)
y el XML de cada provincia. Condiciones (https://www.boe.es/informacion/aviso_legal/index.php):
reutilización permitida citando "Basado en datos de la Agencia Estatal Boletín Oficial del Estado",
sin desnaturalizar la información ni sugerir carácter oficial, y respetando el RGPD.

Privacidad: solo se guardan datos de la sociedad (nombre, provincia, municipio, objeto social, CNAE,
capital y actos). Nunca nombramientos, ceses, apoderados ni socios. Se descartan las entradas cuyo
nombre no lleva forma societaria (para no recoger empresarios individuales).

Aviso: la ampliación de capital del BORME es el capital NOMINAL. No incluye la prima de emisión,
así que no mide el tamaño real de una ronda de financiación.

Uso:
    python scripts/build_borme.py [--months 12]

Salida:
    data/borme_companies.csv.gz   una fila por sociedad con constitución o ampliación de capital
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "borme"
API = "https://www.boe.es/datosabiertos/api/borme/sumario/"
SOURCE = "borme"
LICENSE = ("Basado en datos de la Agencia Estatal Boletín Oficial del Estado (BORME); "
           "condiciones en https://www.boe.es/informacion/aviso_legal/index.php")

COMPANY_FORM = re.compile(r"\b(SOCIEDAD|S\.?L\.?U?|S\.?A\.?U?|S\.?L\.?L|SLNE|COOPERATIVA|S\.?COOP|AGRUPACION DE INTERES)\b", re.I)
ACTS = ["Constitución", "Ampliación de capital", "Reducción de capital", "Disolución", "Extinción", "Fusión por absorción",
        "Transformación de sociedad", "Cambio de denominación social", "Cambio de objeto social", "Cambio de domicilio social",
        "Situación concursal", "Escisión"]

RE_ENTRY = re.compile(r"^(\d+)\s*-\s*(.+?)\.?\s*$")
RE_CONST = re.compile(r"Constituci[oó]n\.\s*(?:Comienzo de operaciones:\s*([\d.]+)\.)?\s*(?:Objeto social:\s*(.*?)\s*Domicilio:\s*(.*?)\.\s*)?Capital:\s*([\d.,]+)\s*Euros")
RE_AMPL = re.compile(r"Ampliaci[oó]n de capital\.\s*Capital:\s*([\d.,]+)\s*Euros\.(?:\s*Resultante Suscrito:\s*([\d.,]+)\s*Euros)?")
RE_RED = re.compile(r"Reducci[oó]n de capital\.\s*Importe reducci[oó]n:\s*([\d.,]+)\s*Euros\.(?:\s*Resultante Suscrito:\s*([\d.,]+)\s*Euros)?")
RE_CNAE = re.compile(r"CNAE[:\s]*(\d{2,4})")
RE_TOWN = re.compile(r"\(([^()]+)\)\s*$")

# División CNAE (2 dígitos) -> sección (letra), para cruzar con los sectores del Banco de España
CNAE_SECTIONS = [(1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"), (41, 43, "F"), (45, 47, "G"),
                 (49, 53, "H"), (55, 56, "I"), (58, 63, "J"), (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"),
                 (84, 84, "O"), (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S")]


def eur(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        return None


def cnae_division(code: str | None) -> str | None:
    if not isinstance(code, str) or len(code) < 2 or not code[:2].isdigit():
        return None  # sin CNAE (NaN) o mal formado
    div = int(code[:2])
    letter = next((l for lo, hi, l in CNAE_SECTIONS if lo <= div <= hi), None)
    return f"{letter}{div:02d}" if letter else None


def get(session: requests.Session, url: str, cache_name: str, accept: str | None = None) -> bytes | None:
    path = CACHE / cache_name
    if path.exists():
        return path.read_bytes() or None
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60, headers={"Accept": accept} if accept else None)
            time.sleep(0.2)
            if r.status_code == 404:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")  # día sin BORME: se recuerda para no repetir
                return None
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return r.content
        except requests.RequestException:
            time.sleep(2 ** attempt)
    print(f"  aviso: no se pudo descargar {url}", flush=True)
    return None


def parse_province(xml: bytes, date: str, url_pdf: str) -> list[dict]:
    root = ET.fromstring(xml)
    province = (root.findtext("metadatos/titulo") or "").strip()
    out, current = [], None
    for p in root.iter("p"):
        cls, text = p.get("class"), " ".join("".join(p.itertext()).split())
        if cls == "articulo":
            m = RE_ENTRY.match(text)
            current = {"entry": m[1], "name": m[2].strip()} if m else None
            continue
        if cls != "parrafo" or not current or not COMPANY_FORM.search(current["name"]):
            continue
        acts = [a for a in ACTS if a.lower() in text.lower()]
        rec = {"name": current["name"], "entry": current["entry"], "province": province, "date": date,
               "acts": acts, "url": url_pdf}
        if (m := RE_CONST.search(text)):
            rec["start_ops"] = m[1]
            rec["purpose"] = (m[2] or "").strip()[:300] or None
            town = RE_TOWN.search(m[3] or "")
            rec["town"] = town[1].strip().title() if town else None
            rec["capital_initial"] = eur(m[4])
        if (m := RE_AMPL.search(text)):
            rec["capital_increase"] = eur(m[1])
            rec["capital_after"] = eur(m[2])
        if (m := RE_RED.search(text)):
            rec["capital_reduction"] = eur(m[1])
            rec["capital_after"] = eur(m[2]) or rec.get("capital_after")
        cnae = RE_CNAE.search(text)
        rec["cnae"] = cnae[1] if cnae else None
        out.append(rec)
    return out


def build(months: int) -> None:
    today = dt.date.today()
    start = today - dt.timedelta(days=int(months * 30.5))
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (compatible; valorador-startups)"
    records = []
    days = [start + dt.timedelta(days=i) for i in range((today - start).days + 1)]
    for n, day in enumerate(d for d in days if d.weekday() < 5):
        ymd = day.strftime("%Y%m%d")
        raw = get(s, API + ymd, f"sumario/{ymd}.json", accept="application/json")
        if not raw:
            continue
        diario = json.loads(raw)["data"]["sumario"]["diario"]
        for d in diario:
            for sec in d.get("seccion", []):
                if sec.get("codigo") != "A":
                    continue
                for item in sec.get("item", []):
                    xml = get(s, item["url_xml"], f"xml/{item['identificador']}.xml")
                    if xml:
                        records += parse_province(xml, day.isoformat(), item["url_pdf"]["texto"])
        if n % 20 == 0:
            print(f"{day.isoformat()}: {len(records)} actos de sociedades", flush=True)

    acts = pd.DataFrame(records)
    acts = acts[acts["acts"].apply(lambda a: "Constitución" in a or "Ampliación de capital" in a) |
                acts["name"].isin(acts.loc[acts["acts"].apply(lambda a: "Constitución" in a or "Ampliación de capital" in a), "name"])]
    acts = acts.sort_values("date")
    g = acts.groupby("name", sort=False)
    xw = pd.read_csv(DATA / "industry_crosswalk.csv", dtype=str)
    # Si un CNAE sirve a varias industrias, la primera fila de la tabla de equivalencias es la principal
    bde_xw = (xw[xw["source"] == "bde"].drop_duplicates("industry_original", keep="first")
              .set_index("industry_original")["industry_std"].to_dict())

    def first(series):
        s = series.dropna()
        return s.iloc[0] if len(s) else None

    def last(series):
        s = series.dropna()
        return s.iloc[-1] if len(s) else None

    out = pd.DataFrame({
        "name": list(g.groups),
        "province": g["province"].agg(last).values,
        "town": g["town"].agg(first).values if "town" in acts else None,
        "constitution_date": g.apply(lambda x: first(x.loc[x["acts"].apply(lambda a: "Constitución" in a), "date"])).values,
        "start_ops": g["start_ops"].agg(first).values if "start_ops" in acts else None,
        "purpose": g["purpose"].agg(first).values if "purpose" in acts else None,
        "cnae": g["cnae"].agg(first).values,
        "capital_initial": g["capital_initial"].agg(first).values if "capital_initial" in acts else None,
        "n_capital_increases": g.apply(lambda x: int(x["acts"].apply(lambda a: "Ampliación de capital" in a).sum())).values,
        "capital_increases_total": g["capital_increase"].sum(min_count=1).values if "capital_increase" in acts else None,
        "capital_latest": g["capital_after"].agg(last).values if "capital_after" in acts else None,
        "dissolved": g.apply(lambda x: any(("Disolución" in a or "Extinción" in a) for a in x["acts"])).values,
        "last_act_date": g["date"].agg("max").values,
        "acts": g["acts"].apply(lambda s: ", ".join(sorted({a for acts_ in s for a in acts_}))).values,
        "url": g["url"].agg(last).values,
    })
    out["capital_latest"] = out["capital_latest"].fillna(out["capital_initial"])
    out["cnae_division"] = out["cnae"].map(cnae_division)
    out["industry_std"] = out["cnae_division"].map(bde_xw)
    out["source"] = SOURCE
    out["as_of"] = out["last_act_date"]
    out["retrieved_at"] = today.isoformat()
    out["license"] = LICENSE
    out.to_csv(DATA / "borme_companies.csv.gz", index=False)
    print(f"borme_companies: {len(out)} sociedades ({int(out['constitution_date'].notna().sum())} constituidas en el periodo, "
          f"{int((out['n_capital_increases'] > 0).sum())} con ampliaciones de capital)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", type=int, default=12)
    sys.exit(build(parser.parse_args().months))
