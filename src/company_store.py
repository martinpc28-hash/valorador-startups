"""Biblioteca de empresas del usuario (plantillas para cargar en el modelo y comparar).

Cuentas propias de la app (usuario y contraseña, sin correo). La contraseña se guarda como
hash PBKDF2-SHA256 con sal; nunca en claro.

Dos almacenes con la misma interfaz:
- FirestoreStore: en producción (Cloud Run). Cada usuario tiene su colección, identificada por
  un hash de su nombre de usuario; la app solo muestra las empresas del usuario con sesión.
- LocalStore: en local y en pruebas, un JSON por usuario en `.local_companies/` (no se versiona).
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
    def __init__(self, folder: Path | None = None):
        self.folder = folder or LOCAL_DIR

    def _path(self, owner: str) -> Path:
        return self.folder / f"{owner_key(owner)}.json"

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
    def __init__(self, project: str | None = None):
        from google.cloud import firestore

        self.db = firestore.Client(project=project)

    def _col(self, owner: str):
        return self.db.collection("users").document(owner_key(owner)).collection("companies")

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


# ---------------------------------------------------------------- cuentas (usuario y contraseña)

USERNAME_RE = r"^[a-z0-9._-]{3,32}$"
PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    import hmac

    _, digest = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(digest, hash_hex)


class AccountError(ValueError):
    pass


def validate_new_account(username: str, password: str) -> str:
    import re

    u = username.strip().lower()
    if not re.match(USERNAME_RE, u):
        raise AccountError("El usuario debe tener de 3 a 32 caracteres: letras, números, punto, guion o guion bajo.")
    if len(password) < 6:
        raise AccountError("La contraseña debe tener al menos 6 caracteres.")
    return u


class LocalAccounts:
    def __init__(self, folder: Path | None = None):
        self.path = (folder or LOCAL_DIR) / "_accounts.json"

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get(self, username: str) -> dict | None:
        return self._read().get(username)

    def create(self, username: str, record: dict) -> None:
        data = self._read()
        if username in data:
            raise AccountError("Ese usuario ya existe.")
        data[username] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")


class FirestoreAccounts:
    def __init__(self, project: str | None = None):
        from google.cloud import firestore

        self.col = firestore.Client(project=project).collection("accounts")

    def get(self, username: str) -> dict | None:
        doc = self.col.document(username).get()
        return doc.to_dict() if doc.exists else None

    def create(self, username: str, record: dict) -> None:
        from google.api_core.exceptions import AlreadyExists

        try:
            self.col.document(username).create(record)  # falla si ya existe: no se pisa una cuenta
        except AlreadyExists as e:
            raise AccountError("Ese usuario ya existe.") from e


def register(accounts, username: str, password: str) -> str:
    u = validate_new_account(username, password)
    salt, digest = hash_password(password)
    accounts.create(u, {"salt": salt, "hash": digest, "iterations": PBKDF2_ITERATIONS,
                        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return u


def authenticate(accounts, username: str, password: str) -> str | None:
    u = username.strip().lower()
    rec = accounts.get(u)
    if rec and verify_password(password, rec["salt"], rec["hash"]):
        return u
    return None


def _backend() -> str:
    return os.environ.get("VALORADOR_STORE") or ("firestore" if os.environ.get("K_SERVICE") else "local")


def get_accounts():
    return FirestoreAccounts(os.environ.get("GOOGLE_CLOUD_PROJECT")) if _backend() == "firestore" else LocalAccounts()


def get_store() -> CompanyStore:
    """Firestore en Cloud Run (o si VALORADOR_STORE=firestore); si no, archivo local."""
    if _backend() == "firestore":
        return FirestoreStore(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    return LocalStore()
