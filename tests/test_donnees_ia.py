"""Tests de donnees_ia.py avec une API Gemini simulée (aucun appel réseau)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import donnees_ia

RACINE = Path(__file__).resolve().parent.parent
FLUX_IA = json.loads((RACINE / "fixtures" / "donnees_test.json").read_text(encoding="utf-8"))


class FausseReponse:
    def __init__(self, status: int, texte: str = ""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = texte

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}


@pytest.fixture
def env_propre(monkeypatch, tmp_path):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_ACCESS_TOKEN",
                "GOOGLE_CLOUD_LOCATION", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # pas de .env du dépôt
    return tmp_path


def test_extraire_json_tolere_les_blocs_markdown():
    assert donnees_ia.extraire_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert donnees_ia.extraire_json('Voici :\n[{"a": 1}]') == [{"a": 1}]
    with pytest.raises(donnees_ia.ErreurIA):
        donnees_ia.extraire_json("pas de json")


def test_sans_authentification(env_propre):
    with pytest.raises(donnees_ia.ErreurIA, match="GEMINI_API_KEY"):
        donnees_ia.appeler_ia("x")


def test_cle_dans_l_en_tete_et_bascule_vertex_express(env_propre, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "cle-de-test")
    appels = []

    def faux_post(url, json, headers, timeout):
        appels.append((url, headers))
        return FausseReponse(403) if "generativelanguage" in url else FausseReponse(200, "[]")

    monkeypatch.setattr(donnees_ia.requests, "post", faux_post)
    assert donnees_ia.appeler_ia("x") == "[]"
    assert "aiplatform.googleapis.com" in appels[1][0]
    assert all("cle-de-test" not in url and h["x-goog-api-key"] == "cle-de-test" for url, h in appels)


def test_vertex_avec_projet(env_propre, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "mon-projet")
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "jeton")
    vus = {}

    def faux_post(url, json, headers, timeout):
        vus.update(url=url, auth=headers["Authorization"])
        return FausseReponse(200, "[]")

    monkeypatch.setattr(donnees_ia.requests, "post", faux_post)
    donnees_ia.appeler_ia("x")
    assert vus["url"].startswith("https://us-central1-aiplatform.googleapis.com/v1/projects/mon-projet/")
    assert vus["auth"] == "Bearer jeton"


def test_modele_introuvable(env_propre, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(donnees_ia.requests, "post", lambda *a, **k: FausseReponse(404))
    with pytest.raises(donnees_ia.ErreurIA, match="introuvable"):
        donnees_ia.appeler_ia("x")


def test_bout_en_bout_vers_predictor(env_propre, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(donnees_ia.requests, "post",
                        lambda *a, **k: FausseReponse(200, "```json\n" + json.dumps(FLUX_IA) + "\n```"))
    sortie = env_propre / "flux_agreges.json"
    code = donnees_ia.main([str(RACINE / "exemple flux.json"), "--out", str(sortie)])
    assert code == 0
    flux = json.loads(sortie.read_text(encoding="utf-8"))
    assert len(flux) == len(FLUX_IA)

    import predictor_standalone as ps
    t = ps.predict(flux, 500, "2026-09-30")
    assert t["p_decouvert"].max() < 0.2


def test_synthese_une_commande(env_propre, monkeypatch, capsys):
    import synthese

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(donnees_ia.requests, "post",
                        lambda *a, **k: FausseReponse(200, json.dumps(FLUX_IA)))
    flux = env_propre / "flux.json"
    npy = env_propre / "t.npy"
    code = synthese.main([str(RACINE / "exemple flux.json"), "--flux", str(flux), "--npy", str(npy)])
    assert code == 0
    import numpy as np
    t = np.load(npy)
    assert str(t["date"][0]) == "2026-09-30"          # dernière transaction de l'exemple
    assert t["esperance"][0] == 1000.0                 # solde par défaut
    assert json.loads(flux.read_text(encoding="utf-8")) == FLUX_IA

    # --reutiliser : plus aucun appel à l'IA
    monkeypatch.setattr(donnees_ia.requests, "post", lambda *a, **k: pytest.fail("appel IA inattendu"))
    assert synthese.main([str(RACINE / "exemple flux.json"), "--flux", str(flux), "--reutiliser"]) == 0
    assert "2026-09-30" in capsys.readouterr().out
