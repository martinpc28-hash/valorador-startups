"""Genera los CSV de empresas individuales a partir de datos públicos de la SEC (EDGAR).

No hace falta para ejecutar la app: documenta cómo se generaron los datos de `data/`.

Fuentes (gratuitas; "Information presented on sec.gov is considered public information and
may be copied or further distributed by users of the web site without the SEC's permission"):
    Form D : datasets trimestrales de avisos de ofertas privadas (Reg D)
    Form C : datasets trimestrales de crowdfunding (Reg CF): estados financieros de la empresa
    S-1    : índice de presentaciones de EDGAR + API XBRL "frames" para los financieros

La SEC exige identificarse con nombre y correo de contacto en el User-Agent:
    set SEC_USER_AGENT=valorador-startups tu@correo.com      (Windows)
    export SEC_USER_AGENT="valorador-startups tu@correo.com"  (macOS/Linux)

Uso:
    python scripts/build_sec.py [--quarters 4]

Privacidad: no se guardan personas relacionadas, firmantes, direcciones ni teléfonos;
solo datos de la empresa (nombre, ciudad, estado, cifras de la oferta y financieros).

Salidas (CSV comprimidos, con procedencia por fila):
    data/sec_form_d.csv.gz        última oferta de cada empresa operativa (sin fondos)
    data/sec_form_d_funds.csv.gz  ofertas de fondos de venture capital (tamaño de fondo)
    data/sec_form_c.csv.gz        última presentación con financieros de cada empresa
    data/sec_s1.csv.gz            empresas que presentaron un S-1, con financieros XBRL
    data/sec_manifest.csv         trimestres y archivos usados
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "sec"
SEC = "https://www.sec.gov"
LICENSE = (
    "Información pública de sec.gov: puede copiarse y redistribuirse sin permiso de la SEC "
    "(https://www.sec.gov/about/privacy-information)"
)
FORM_D_PAGE = SEC + "/data-research/sec-markets-data/form-d-data-sets"
FORM_C_PAGE = SEC + "/data-research/sec-markets-data/crowdfunding-offerings-data-sets"
PROV = ["source", "url", "as_of", "retrieved_at", "license"]

SPAC_SIC = {"6770"}
NOT_STARTUP_NAME = re.compile(r"\b(?:trust|etf|fund|acquisition corp|capital corp\.? [ivx]+)\b", re.I)


class Sec:
    """Cliente con User-Agent identificado, caché en disco y máximo ~8 peticiones/s."""

    def __init__(self, user_agent: str):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.last = 0.0

    def get(self, url: str, cache_name: str | None = None) -> bytes:
        path = CACHE / cache_name if cache_name else None
        if path and path.exists():
            return path.read_bytes()
        wait = 0.125 - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        resp = self.s.get(url, timeout=120)
        self.last = time.time()
        resp.raise_for_status()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
        return resp.content


def filing_url(cik, accession: str) -> str:
    return f"{SEC}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_tsv(zbytes: bytes, table: str) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    name = next(n for n in z.namelist() if n.endswith("/" + table) or n == table)
    return pd.read_csv(io.BytesIO(z.read(name)), sep="\t", dtype=str, encoding="latin-1",
                       on_bad_lines="skip", quoting=3)


def zip_links(sec: Sec, page: str, suffix: str, n: int) -> list[str]:
    html = sec.get(page).decode("utf-8", "replace")
    links = re.findall(r'href="([^"]+' + suffix + r'\.zip)"', html)
    return sorted(dict.fromkeys(links), key=lambda u: Path(u).stem, reverse=True)[:n]


# ---------------------------------------------------------------- Form D


def build_form_d(sec: Sec, n_quarters: int, crosswalk: dict[str, str], today: str):
    frames, files = [], []
    for link in zip_links(sec, FORM_D_PAGE, "_d", n_quarters):
        zb = sec.get(SEC + link, Path(link).name)
        files.append(SEC + link)
        sub = read_tsv(zb, "FORMDSUBMISSION.tsv")
        off = read_tsv(zb, "OFFERING.tsv")
        iss = read_tsv(zb, "ISSUERS.tsv")
        iss = iss[iss["IS_PRIMARYISSUER_FLAG"].str.upper() == "YES"]
        df = sub.merge(off, on="ACCESSIONNUMBER").merge(iss, on="ACCESSIONNUMBER")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["TESTORLIVE"].fillna("LIVE").str.upper() == "LIVE"]
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="mixed", errors="coerce").dt.date.astype(str)

    out = pd.DataFrame({
        "cik": num(df["CIK"]).astype("Int64"),
        "name": df["ENTITYNAME"].str.strip(),
        "city": df["CITY"].str.title(),
        "state": df["STATEORCOUNTRYDESCRIPTION"],
        "entity_type": df["ENTITYTYPE"],
        "year_of_inc": num(df["YEAROFINC_VALUE_ENTERED"]).astype("Int64"),
        "inc_within_5y": df["YEAROFINC_TIMESPAN_CHOICE"].eq("withinFiveYears"),
        "industry_group": df["INDUSTRYGROUPTYPE"],
        "fund_type": df["INVESTMENTFUNDTYPE"],
        "revenue_range": df["REVENUERANGE"],
        "sic": df["SIC_CODE"],
        "is_amendment": df["ISAMENDMENT"].str.lower().eq("true"),
        "equity": df["ISEQUITYTYPE"].str.lower().eq("true"),
        "debt": df["ISDEBTTYPE"].str.lower().eq("true"),
        "exemptions": df["FEDERALEXEMPTIONS_ITEMS_LIST"],
        "first_sale_date": df["SALE_DATE"],
        "total_offering": num(df["TOTALOFFERINGAMOUNT"]),  # "Indefinite" -> NaN
        "total_sold": num(df["TOTALAMOUNTSOLD"]),
        "total_remaining": num(df["TOTALREMAINING"]),
        "min_investment": num(df["MINIMUMINVESTMENTACCEPTED"]),
        "n_investors": num(df["TOTALNUMBERALREADYINVESTED"]),
        "filing_date": df["filing_date"],
        "accession": df["ACCESSIONNUMBER"],
    })
    out["industry_std"] = out["industry_group"].map(crosswalk)
    out["source"] = "sec_form_d"
    out["url"] = [filing_url(c, a) if pd.notna(c) else SEC for c, a in zip(out["cik"], out["accession"])]
    out["as_of"] = out["filing_date"]
    out["retrieved_at"] = today
    out["license"] = LICENSE

    # Una fila por empresa: la presentación más reciente (las enmiendas actualizan importes)
    out = out.sort_values(["cik", "filing_date"]).drop_duplicates("cik", keep="last")
    is_fund = out["industry_group"].eq("Pooled Investment Fund")
    companies = out[~is_fund].drop(columns=["fund_type"])
    vc_funds = out[is_fund & out["fund_type"].eq("Venture Capital Fund")]
    vc_funds = vc_funds[["cik", "name", "state", "fund_type", "total_offering", "total_sold", "n_investors",
                         "first_sale_date", "filing_date", "accession"] + PROV]
    unmapped = sorted(set(companies["industry_group"].dropna()) - set(crosswalk))
    return companies, vc_funds, files, unmapped


# ---------------------------------------------------------------- Form C


def build_form_c(sec: Sec, n_quarters: int, today: str):
    frames, files = [], []
    for link in zip_links(sec, FORM_C_PAGE, "_cf", n_quarters):
        zb = sec.get(SEC + link, Path(link).name)
        files.append(SEC + link)
        sub = read_tsv(zb, "FORM_C_SUBMISSION.tsv")
        dis = read_tsv(zb, "FORM_C_DISCLOSURE.tsv")
        inf = read_tsv(zb, "FORM_C_ISSUER_INFORMATION.tsv")
        frames.append(sub.merge(dis, on="ACCESSION_NUMBER").merge(inf, on="ACCESSION_NUMBER"))
    df = pd.concat(frames, ignore_index=True)
    # Ofertas (C, C/A) e informes anuales (C-AR, C-AR/A): las que traen estados financieros
    df = df[df["SUBMISSION_TYPE"].isin(["C", "C/A", "C-AR", "C-AR/A"])]
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="mixed", errors="coerce").dt.date.astype(str)
    is_offer = df["SUBMISSION_TYPE"].isin(["C", "C/A"])

    out = pd.DataFrame({
        "cik": num(df["CIK"]).astype("Int64"),
        "name": df["NAMEOFISSUER"].str.strip(),
        "city": df["CITY"].str.title(),
        "state": df["STATEORCOUNTRY"],
        "legal_status": df["LEGALSTATUSFORM"],
        "date_of_inc": df["DATEINCORPORATION"],
        "website": df["ISSUERWEBSITE"],
        "submission_type": df["SUBMISSION_TYPE"],
        "security_type": df["SECURITYOFFEREDTYPE"].where(is_offer),
        "price": num(df["PRICE"]).where(is_offer),
        "offering_target": num(df["OFFERINGAMOUNT"]).where(is_offer),
        "offering_max": num(df["MAXIMUMOFFERINGAMOUNT"]).where(is_offer),
        "deadline": df["DEADLINEDATE"].where(is_offer),
        "employees": num(df["CURRENTEMPLOYEES"]),
        "total_assets": num(df["TOTALASSETMOSTRECENTFISCALYEAR"]),
        "cash": num(df["CASHEQUIMOSTRECENTFISCALYEAR"]),
        "short_term_debt": num(df["SHORTTERMDEBTMRECENTFISCALYEAR"]),
        "long_term_debt": num(df["LONGTERMDEBTRECENTFISCALYEAR"]),
        "revenue": num(df["REVENUEMOSTRECENTFISCALYEAR"]),
        "revenue_prior": num(df["REVENUEPRIORFISCALYEAR"]),
        "cogs": num(df["COSTGOODSSOLDRECENTFISCALYEAR"]),
        "net_income": num(df["NETINCOMEMOSTRECENTFISCALYEAR"]),
        "net_income_prior": num(df["NETINCOMEPRIORFISCALYEAR"]),
        "filing_date": df["filing_date"],
        "accession": df["ACCESSION_NUMBER"],
    })
    out["source"] = "sec_form_c"
    out["url"] = [filing_url(c, a) if pd.notna(c) else SEC for c, a in zip(out["cik"], out["accession"])]
    out["as_of"] = out["filing_date"]
    out["retrieved_at"] = today
    out["license"] = LICENSE

    # Última oferta de cada empresa, completada con los financieros más recientes
    out = out.sort_values(["cik", "filing_date"])
    latest = out.drop_duplicates("cik", keep="last")
    offers = out[out["submission_type"].isin(["C", "C/A"])].drop_duplicates("cik", keep="last")
    offer_cols = ["security_type", "price", "offering_target", "offering_max", "deadline"]
    latest = latest.drop(columns=offer_cols).merge(offers[["cik"] + offer_cols], on="cik", how="left")
    return latest, files


# ---------------------------------------------------------------- S-1


S1_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
}


def recent_quarters(n: int) -> list[tuple[int, int]]:
    today = dt.date.today()
    y, q = today.year, (today.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        q -= 1
        if q == 0:
            y, q = y - 1, 4
        out.append((y, q))
    return out


def frame(sec: Sec, concept: str, period: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json"
    try:
        data = pd.read_json(io.BytesIO(sec.get(url, f"frames/{concept}_{period}.json")), typ="series")
    except requests.HTTPError:
        return pd.DataFrame(columns=["cik", "val", "end", "accn"])
    return pd.DataFrame(data["data"])[["cik", "val", "end", "accn"]]


def build_s1(sec: Sec, n_quarters: int, sic_xw: dict[str, str], today: str):
    rows = []
    for y, q in recent_quarters(n_quarters):
        url = f"{SEC}/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
        text = sec.get(url, f"form_{y}q{q}.idx").decode("latin-1")
        for line in text.splitlines():
            if line.startswith("S-1 "):
                m = re.match(r"S-1\s+(.+?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)", line)
                if m:
                    rows.append({"name": m[1].strip(), "cik": int(m[2]), "filing_date": m[3], "path": m[4]})
    s1 = pd.DataFrame(rows).sort_values("filing_date").drop_duplicates("cik", keep="first")
    s1 = s1[~s1["name"].str.contains(NOT_STARTUP_NAME)]

    # Financieros: año fiscal más reciente disponible en XBRL (CY2025, si no CY2024)
    fin = pd.DataFrame({"cik": s1["cik"]})
    for field, concepts in S1_CONCEPTS.items():
        for year in (2025, 2024):
            col = f"{field}_{year}"
            vals = pd.concat([frame(sec, c, f"CY{year}") for c in concepts]).drop_duplicates("cik")
            fin = fin.merge(vals[["cik", "val"]].rename(columns={"val": col}), on="cik", how="left")
    for field in S1_CONCEPTS:
        fin[field] = fin[f"{field}_2025"].fillna(fin[f"{field}_2024"])
    fin["fiscal_year"] = np.where(fin["revenue_2025"].notna(), 2025, np.where(fin["revenue_2024"].notna(), 2024, np.nan))
    fin["revenue_prior"] = np.where(fin["revenue_2025"].notna(), fin["revenue_2024"], np.nan)
    s1 = s1.merge(fin[["cik", "fiscal_year", "revenue", "revenue_prior", "operating_income", "net_income"]], on="cik")
    s1 = s1[s1["revenue"].notna()]

    # SIC y estado desde el perfil de cada empresa
    sics, states = [], []
    for cik in s1["cik"]:
        try:
            j = pd.read_json(io.BytesIO(sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                                                f"submissions/{cik}.json")), typ="series")
            sics.append(str(j.get("sic") or ""))
            states.append((j.get("addresses") or {}).get("business", {}).get("stateOrCountryDescription"))
        except requests.HTTPError:
            sics.append("")
            states.append(None)
    s1["sic"] = sics
    s1["state"] = states
    s1 = s1[~s1["sic"].isin(SPAC_SIC)]
    s1["industry_std"] = s1["sic"].map(sic_xw)
    s1["accession"] = s1["path"].str.extract(r"/([\d-]+)\.txt$")[0]
    s1["source"] = "sec_s1"
    s1["url"] = [filing_url(c, a) for c, a in zip(s1["cik"], s1["accession"])]
    s1["as_of"] = s1["filing_date"]
    s1["retrieved_at"] = today
    s1["license"] = LICENSE
    unmapped = sorted(set(s1["sic"]) - set(sic_xw) - {""})
    files = [f"{SEC}/Archives/edgar/full-index/{y}/QTR{q}/form.idx" for y, q in recent_quarters(n_quarters)]
    files.append("https://data.sec.gov/api/xbrl/frames/")
    return s1.drop(columns=["path"]), files, unmapped


# ---------------------------------------------------------------- main


def build(n_quarters: int) -> None:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        raise SystemExit("Define SEC_USER_AGENT con un nombre y un correo de contacto (lo exige la SEC).")
    sec = Sec(ua)
    today = dt.date.today().isoformat()
    xw = pd.read_csv(DATA / "industry_crosswalk.csv", dtype=str)
    d_xw = xw[xw["source"] == "sec_form_d"].set_index("industry_original")["industry_std"].to_dict()
    sic_xw = xw[xw["source"] == "sec_sic"].set_index("industry_original")["industry_std"].to_dict()

    companies, vc_funds, d_files, d_unmapped = build_form_d(sec, n_quarters, d_xw, today)
    form_c, c_files = build_form_c(sec, n_quarters, today)
    s1, s1_files, s1_unmapped = build_s1(sec, n_quarters, sic_xw, today)

    companies.to_csv(DATA / "sec_form_d.csv.gz", index=False)
    vc_funds.to_csv(DATA / "sec_form_d_funds.csv.gz", index=False)
    form_c.to_csv(DATA / "sec_form_c.csv.gz", index=False)
    s1.to_csv(DATA / "sec_s1.csv.gz", index=False)
    pd.DataFrame(
        [{"dataset": "sec_form_d", "file": f, "retrieved_at": today} for f in d_files]
        + [{"dataset": "sec_form_c", "file": f, "retrieved_at": today} for f in c_files]
        + [{"dataset": "sec_s1", "file": f, "retrieved_at": today} for f in s1_files]
    ).to_csv(DATA / "sec_manifest.csv", index=False)

    print(f"Form D: {len(companies)} empresas, {len(vc_funds)} fondos VC | Form C: {len(form_c)} | S-1: {len(s1)}")
    if d_unmapped:
        print("Grupos de Form D sin equivalencia:", d_unmapped)
    if s1_unmapped:
        print(f"SIC de S-1 sin equivalencia ({len(s1_unmapped)}):", s1_unmapped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quarters", type=int, default=4, help="trimestres más recientes a incluir")
    sys.exit(build(parser.parse_args().quarters))
