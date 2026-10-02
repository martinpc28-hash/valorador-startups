"""Genera los datos del Banco Central Europeo (BCE) usados por la app.

No hace falta para ejecutar la app: documenta cómo se generaron los datos de `data/`.

Series (API pública del ECB Data Portal, sin clave):
    EXR.D.USD.EUR.SP00.A                    tipo de cambio de referencia USD por EUR (diario)
    YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y    curva AAA de la zona euro, tipo spot a 10 años

El tipo spot de la curva está en composición continua. Se guarda convertido a tasa anual
efectiva, exp(r) - 1, para que sea coherente con el resto de tasas de la app.

Uso:
    python scripts/build_ecb.py [--obs 60]

Salidas:
    data/fx_rates.csv       (reemplaza las filas con source == "ecb")
    data/market_metrics.csv (reemplaza las filas con source == "ecb")
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
API = "https://data-api.ecb.europa.eu/service/data/"
SOURCE = "ecb"
LICENSE = (
    "Uso libre citando al BCE como fuente y reproduciendo los datos con exactitud "
    "(https://www.ecb.europa.eu/services/disclaimer/html/index.en.html); "
    "el tipo spot se convierte de composición continua a anual efectiva"
)


def get_series(key: str, obs: int) -> tuple[pd.DataFrame, str]:
    url = f"{API}{key}?lastNObservations={obs}&format=csvdata"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df[["TIME_PERIOD", "OBS_VALUE"]].dropna(), url.split("?")[0]


def replace_rows(path: Path, new: pd.DataFrame) -> None:
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["source"] != SOURCE], new], ignore_index=True)
    new.to_csv(path, index=False)


def build(obs: int) -> None:
    today = dt.date.today().isoformat()

    fx, fx_url = get_series("EXR/D.USD.EUR.SP00.A", obs)
    fx_rows = pd.DataFrame({
        "source": SOURCE, "date": fx["TIME_PERIOD"], "base": "EUR", "quote": "USD",
        "rate": fx["OBS_VALUE"].astype(float), "as_of": fx["TIME_PERIOD"].max(),
        "retrieved_at": today, "url": fx_url, "license": LICENSE,
    })
    replace_rows(DATA / "fx_rates.csv", fx_rows)

    yc, yc_url = get_series("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", obs)
    yc_rows = pd.DataFrame({
        "source": SOURCE, "region": "EA", "series": "ea_aaa_10y",
        "date": yc["TIME_PERIOD"],
        "value": [math.exp(float(v) / 100.0) - 1.0 for v in yc["OBS_VALUE"]],
        "unit": "decimal", "as_of": yc["TIME_PERIOD"].max(), "retrieved_at": today,
        "url": yc_url, "license": LICENSE,
    })
    replace_rows(DATA / "market_metrics.csv", yc_rows)
    print(f"fx: {len(fx_rows)} obs hasta {fx_rows.date.max()}; ea_aaa_10y: {len(yc_rows)} obs hasta {yc_rows.date.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs", type=int, default=60, help="número de observaciones recientes")
    sys.exit(build(parser.parse_args().obs))
