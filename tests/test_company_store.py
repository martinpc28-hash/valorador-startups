"""Biblioteca de empresas y cuentas de usuario (almacén local; Firestore comparte la interfaz)."""

import json
import math

import pytest

from src import company_store as cs


@pytest.fixture
def folder(tmp_path):
    return tmp_path


def test_save_list_update_delete(folder):
    store = cs.LocalStore(folder)
    c = cs.new_company("Acme")
    c["inputs"] = {"revenue": 1e6, "growth": math.nan}
    c["financials"] = {2025: {"revenue": 1e6}}
    saved = store.save("ana", c)
    assert saved["inputs"]["growth"] is None  # NaN no se guarda
    assert "2025" in saved["financials"]  # claves de año como texto
    first_created = saved["created_at"]

    saved["name"] = "Acme Corp"
    again = store.save("ana", saved)
    assert again["created_at"] == first_created
    assert [x["name"] for x in store.list("ana")] == ["Acme Corp"]

    store.delete("ana", saved["id"])
    assert store.list("ana") == []


def test_users_do_not_see_each_other(folder):
    store = cs.LocalStore(folder)
    store.save("ana", cs.new_company("De Ana"))
    store.save("luis", cs.new_company("De Luis"))
    assert [c["name"] for c in store.list("ana")] == ["De Ana"]
    assert [c["name"] for c in store.list("luis")] == ["De Luis"]


def test_register_and_authenticate(folder):
    acc = cs.LocalAccounts(folder)
    u = cs.register(acc, "  Martin.PC ", "secreta1")
    assert u == "martin.pc"
    assert cs.authenticate(acc, "MARTIN.pc", "secreta1") == "martin.pc"
    assert cs.authenticate(acc, "martin.pc", "otra") is None
    assert cs.authenticate(acc, "nadie", "secreta1") is None
    with pytest.raises(cs.AccountError):
        cs.register(acc, "martin.pc", "secreta2")  # no se puede pisar una cuenta


def test_password_never_stored_in_clear(folder):
    acc = cs.LocalAccounts(folder)
    cs.register(acc, "ana", "mi-clave-123")
    raw = (folder / "_accounts.json").read_text(encoding="utf-8")
    assert "mi-clave-123" not in raw
    rec = json.loads(raw)["ana"]
    assert len(rec["salt"]) == 32 and len(rec["hash"]) == 64


@pytest.mark.parametrize("user, pwd", [("ab", "secreta1"), ("con espacio", "secreta1"), ("valido", "12345")])
def test_invalid_accounts_rejected(folder, user, pwd):
    with pytest.raises(cs.AccountError):
        cs.register(cs.LocalAccounts(folder), user, pwd)
