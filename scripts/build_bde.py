"""Genera los ratios sectoriales de empresas españolas del Banco de España (Central de Balances).

No hace falta para ejecutar la app: documenta cómo se generaron los datos de `data/`.

Fuente: base RSE (Ratios Sectoriales de las Sociedades no Financieras), servicio público que usa
la aplicación "Compara tu empresa" (https://app.bde.es/gnt_spa/rse/es/). Para cada ratio, sector
CNAE y tamaño (por cifra de negocios) da el cuartil inferior, la mediana y el cuartil superior
de los últimos cinco ejercicios.

Condiciones de uso (https://www.bde.es/wbe/es/estadisticas/recursos/terminos/terminos-de-uso.html):
reutilización libre citando "Fuente: Elaboración propia con datos extraídos del sitio web del
Banco de España (www.bde.es)", sin alterar el sentido de los datos e indicando su fecha.

Solo se descargan los sectores CNAE asociados a una industria en data/industry_crosswalk.csv
(source == "bde").

Uso:
    python scripts/build_bde.py

Salidas:
    data/bde_ratios.csv        detalle: sector × tamaño × ratio × ejercicio (P25, P50, P75)
    data/industry_metrics.csv  reemplaza las filas source == "bde": mediana, todos los tamaños,
                               último ejercicio, con las métricas en la taxonomía de la app
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "bde"
API = "https://app.bde.es/rss_www/api/ratios/"
APP_URL = "https://app.bde.es/gnt_spa/rse/es/"
SOURCE = "bde"
LICENSE = ("Reutilización libre citando: Elaboración propia con datos extraídos del sitio web del Banco de "
           "España (www.bde.es); https://www.bde.es/wbe/es/estadisticas/recursos/terminos/terminos-de-uso.html")

# Código del ratio -> (métrica de la app, unidad, factor de conversión)
RATIOS = {
    "200001": ("revenue_growth_1y", "decimal", 0.01),         # Tasa de variación de las ventas (%)
    "200388": ("sales_per_employee", "eur", 1000.0),          # Ventas por trabajador (miles de euros)
    "200085": ("personnel_cost_per_employee", "eur", 1.0),    # Gastos de personal por trabajador (euros)
    "200395": ("ebitda_margin", "decimal", 0.01),             # EBITDA / ventas (%)
    "200206": ("cost_of_debt", "decimal", 0.01),              # Coste medio de la financiación (%)
    "200212": ("roi_ordinary", "decimal", 0.01),              # Rentabilidad ordinaria del activo (%)
    "200213": ("roe", "decimal", 0.01),                       # Rentabilidad de los recursos propios (%)
    "200217": ("debt_to_liabilities", "decimal", 0.01),       # Recursos ajenos / pasivo (%)
    "200234": ("receivable_days", "days", 1.0),               # Periodo medio de cobro (días)
    "200237": ("payable_days", "days", 1.0),                  # Periodo medio de pago (días)
}
TOTAL_SIZE = "0"


def fix_text(s: str) -> str:
    """El servicio mezcla UTF-8 y Latin-1: se recupera cada texto por separado."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def get_json(session: requests.Session, url: str, cache_name: str):
    path = CACHE / cache_name
    if path.exists():
        return json.loads(path.read_text(encoding="latin-1"))
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 409:  # "No existen datos para la combinación ratio+sector+tamaño"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"years": []}', encoding="latin-1")
                time.sleep(0.25)
                return {"years": []}
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            time.sleep(0.25)  # ritmo moderado: ~4 peticiones por segundo
            return json.loads(r.content.decode("latin-1"))
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"No se pudo descargar {url}")


def build() -> None:
    today = dt.date.today().isoformat()
    s = requests.Session()
    # Solo ASCII: con acentos el cortafuegos del servidor devuelve una página de rechazo
    s.headers["User-Agent"] = "Mozilla/5.0 (compatible; valorador-startups)"
    cfg = get_json(s, API + "config?lang=es", "config_api.json")
    sector_name = {x["id"]: fix_text(x["name"]) for x in cfg["sectors"]}
    size_name = {x["id"]: fix_text(x["name"]) for x in cfg["sizes"]}
    assert set(RATIOS) == {r["id"] for r in cfg["ratios"]}, "La lista de ratios del Banco de España ha cambiado"

    # Un CNAE puede servir a varias industrias (p. ej. J62 para los dos tipos de software)
    xw_rows = pd.read_csv(DATA / "industry_crosswalk.csv", dtype=str)
    xw_rows = xw_rows[xw_rows["source"] == SOURCE]
    xw = xw_rows.groupby("industry_original", sort=False)["industry_std"].apply(list).to_dict()
    missing = set(xw) - set(sector_name)
    if missing:
        raise SystemExit(f"Sectores CNAE de la tabla de equivalencias que ya no existen: {sorted(missing)}")

    rows = []
    total = len(xw) * len(size_name) * len(RATIOS)
    done = 0
    for code in sorted(xw):
        for size in size_name:
            for rid, (metric, unit, factor) in RATIOS.items():
                data = get_json(s, f"{API}percentiles?ratio={rid}&sector={code}&size={size}",
                                f"p/{code}_{size}_{rid}.json")
                for y in data.get("years", []):
                    p = (y.get("percentiles") or []) + [None] * 3
                    vals = [v * factor if isinstance(v, (int, float)) else None for v in p[:3]]
                    for std in xw[code]:
                        rows.append({
                            "source": SOURCE, "region": "ES", "sector_code": code, "sector_name": sector_name[code],
                            "industry_std": std, "size_id": size, "size_name": size_name[size], "metric": metric,
                            "ratio_id": rid, "year": int(y["year"]), "p25": vals[0], "p50": vals[1], "p75": vals[2],
                            "unit": unit, "as_of": f"{int(y['year'])}-12-31", "retrieved_at": today,
                            "url": APP_URL, "license": LICENSE,
                        })
                done += 1
                if done % 250 == 0:
                    print(f"{done}/{total} consultas", flush=True)

    detail = pd.DataFrame(rows)
    # El servicio devuelve 0 en P25, P50 y P75 cuando no publica el dato (pocas empresas, confidencialidad):
    # se guarda como sin dato para no comparar contra ceros
    suppressed = (detail[["p25", "p50", "p75"]] == 0).all(axis=1)
    detail.loc[suppressed, ["p25", "p50", "p75"]] = None
    print(f"{int(suppressed.sum())} filas sin dato publicado (0 en los tres valores) marcadas como vacías")
    detail.to_csv(DATA / "bde_ratios.csv", index=False)

    # Esquema único: mediana, todos los tamaños, último ejercicio con dato
    latest = (detail[(detail["size_id"] == TOTAL_SIZE) & detail["p50"].notna()]
              .sort_values("year").groupby(["sector_code", "industry_std", "metric"]).tail(1))
    long = pd.DataFrame({
        "source": SOURCE, "region": "ES", "industry_std": latest["industry_std"],
        "industry_original": latest["sector_code"], "metric": latest["metric"], "value": latest["p50"],
        "unit": latest["unit"], "as_of": latest["as_of"], "retrieved_at": today,
        "url": APP_URL, "license": LICENSE,
    })
    target = DATA / "industry_metrics.csv"
    old = pd.read_csv(target)
    pd.concat([old[old["source"] != SOURCE], long], ignore_index=True).to_csv(target, index=False)
    print(f"bde_ratios: {len(detail)} filas; industry_metrics: {len(long)} filas de {SOURCE}")


if __name__ == "__main__":
    sys.exit(build())
