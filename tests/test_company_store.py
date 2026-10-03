"""Biblioteca de empresas (almacén local; Firestore comparte la interfaz)."""

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
