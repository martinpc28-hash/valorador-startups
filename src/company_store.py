"""Biblioteca de empresas del usuario (plantillas para cargar en el modelo y comparar).

Por ahora la app no tiene cuentas: usa un único propietario ("compartida") y todos ven la misma
biblioteca. La interfaz acepta un propietario para poder separar usuarios más adelante.

Dos almacenes con la misma interfaz:
- FirestoreStore: en producción (Cloud Run). Colección por propietario (hash del nombre).
- LocalStore: en local y en pruebas, un JSON por usuario en `.local_companies/` (no se versiona).

La misma interfaz guarda otras colecciones (p. ej. "funds", los flujos de fondos que sube el usuario).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Protocol

from src.paths import ROOT

LOCAL_DIR = Path(os.environ.get("VALORADOR_LOCAL_DIR", ROOT / ".local_companies"))


def owner_key(owner: str) -> str:
    return hashlib.sha256(owner.strip().lower().encode()).hexdigest()[:32]


def _clean(obj):
    """Firestore no admite NaN ni claves numéricas: limpiamos antes de guardar."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, "item"):  # escalares de numpy
        return _clean(obj.item())
    return obj


def new_company(name: str = "") -> dict:
    return {"id": uuid.uuid4().hex[:12], "name": name, "industry": None, "stage": None, "currency": "USD",
            "notes": "", "inputs": {}, "financials": {}, "source_files": []}


class CompanyStore(Protocol):
    def list(self, owner: str) -> list[dict]: ...
    def save(self, owner: str, company: dict) -> dict: ...
    def delete(self, owner: str, company_id: str) -> None: ...


def _stamp(company: dict) -> dict:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    c = _clean(dict(company))
    c.setdefault("id", uuid.uuid4().hex[:12])
    c.setdefault("created_at", now)
    c["updated_at"] = now
    return c


class LocalStore:
    def __init__(self, folder: Path | None = None, collection: str = "companies"):
        self._folder = folder
        self.collection = collection

    @property
    def folder(self) -> Path:
        # Se resuelve en cada uso: la instancia puede quedar en caché (st.cache_resource)
        return self._folder or Path(os.environ.get("VALORADOR_LOCAL_DIR", ROOT / ".local_companies"))

    def _path(self, owner: str) -> Path:
        suffix = "" if self.collection == "companies" else f"_{self.collection}"  # compatibilidad con lo ya guardado
        return self.folder / f"{owner_key(owner)}{suffix}.json"

    def _read(self, owner: str) -> dict[str, dict]:
        p = self._path(owner)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _write(self, owner: str, data: dict[str, dict]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._path(owner).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def list(self, owner: str) -> list[dict]:
        return sorted(self._read(owner).values(), key=lambda c: c.get("name", "").lower())

    def save(self, owner: str, company: dict) -> dict:
        data = self._read(owner)
        c = _stamp(company)
        c["created_at"] = data.get(c["id"], {}).get("created_at", c["created_at"])
        data[c["id"]] = c
        self._write(owner, data)
        return c

    def delete(self, owner: str, company_id: str) -> None:
        data = self._read(owner)
        data.pop(company_id, None)
        self._write(owner, data)


class FirestoreStore:
    def __init__(self, project: str | None = None, collection: str = "companies"):
        from google.cloud import firestore

        self.db = firestore.Client(project=project)
        self.collection = collection

    def _col(self, owner: str):
        return self.db.collection("users").document(owner_key(owner)).collection(self.collection)

    def list(self, owner: str) -> list[dict]:
        return sorted((d.to_dict() for d in self._col(owner).stream()), key=lambda c: c.get("name", "").lower())

    def save(self, owner: str, company: dict) -> dict:
        ref = self._col(owner).document(company.get("id") or uuid.uuid4().hex[:12])
        old = ref.get()
        c = _stamp({**company, "id": ref.id})
        if old.exists:
            c["created_at"] = old.to_dict().get("created_at", c["created_at"])
        ref.set(c)
        return c

    def delete(self, owner: str, company_id: str) -> None:
        self._col(owner).document(company_id).delete()


def _backend() -> str:
    return os.environ.get("VALORADOR_STORE") or ("firestore" if os.environ.get("K_SERVICE") else "local")


def get_store(collection: str = "companies") -> CompanyStore:
    """Firestore en Cloud Run (o si VALORADOR_STORE=firestore); si no, archivo local."""
    if _backend() == "firestore":
        return FirestoreStore(os.environ.get("GOOGLE_CLOUD_PROJECT"), collection)
    return LocalStore(collection=collection)
