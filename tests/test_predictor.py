"""Tests d'acceptation (section 16 de la spec) sur la fixture de Sarah."""

from __future__ import annotations

import copy
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from predictor import Params, predict
from predictor.__main__ import main as cli_main

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sarah.json"
AS_OF = date(2026, 9, 30)


@pytest.fixture(scope="module")
def donnees() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sortie(donnees) -> dict:
    return predict(donnees["flux"], donnees["solde_actuel"], donnees["as_of"])


def _flux(sortie, fid):
    return next(f for f in sortie["flux"] if f["flux_id"] == fid)


def _jour(sortie, d):
    return next(t for t in sortie["timeline"] if t["date"] == d)


# ---------------------------------------------------------------- 1–6 : classification


def test_1_salaire(sortie):
    f = _flux(sortie, "F001")
    assert f["type"] == "periodique"
    assert f["statut"]["actif"] > 0.9
    assert f["changement"]["principal"] == "stable"
    assert f["prochaine_date"] == "2026-10-28"


def test_2_ancien_loyer_eteint(sortie):
    f = _flux(sortie, "F002")
    assert f["changement"]["principal"] == "habitude_eteinte"
    assert f["changement"]["distribution"]["habitude_eteinte"] > 0.95


def test_3_nouveau_loyer(sortie):
    f = _flux(sortie, "F003")
    assert f["changement"]["principal"] == "nouvelle_habitude"
    assert f["changement"]["distribution"]["nouvelle_habitude"] > 0.8
    assert f["prochaine_date"] == "2026-10-05"


def test_4_demenagement_ponctuel(sortie):
    f = _flux(sortie, "F004")
    assert f["type"] == "ponctuel"
    assert f["changement"]["distribution"]["evenement_ponctuel"] == pytest.approx(0.80, abs=0.01)


def test_5_courses_variables(sortie):
    f = _flux(sortie, "F005")
    assert f["type"] == "variable"
    assert 0.2 <= f["taux_journalier"] <= 0.3


def test_6_netflix_stable(sortie):
    assert _flux(sortie, "F006")["changement"]["principal"] == "stable"


# ---------------------------------------------------------------- 7–9 : résumé et simulation


def test_7_resume(sortie):
    assert sortie["resume"]["remuneration"]["date"] == "2026-10-28"
    assert sortie["resume"]["date_alerte"] == "2026-10-05"


def test_8_p_decouvert(sortie):
    assert 0.80 <= _jour(sortie, "2026-10-10")["p_decouvert"] <= 0.95
    assert _jour(sortie, "2026-10-29")["p_decouvert"] < 0.08


def test_9_controle_monte_carlo(donnees):
    """Moyenne simulée vs espérance analytique, jour par jour."""
    from predictor.flux import analyser_tous
    from predictor.projection import SEUIL_ACTIF, renseigner_prochaines_dates
    from predictor.schemas import valider_entree
    from predictor.simulation import simuler_soldes

    params = Params()
    entree = valider_entree(donnees["flux"], donnees["solde_actuel"], donnees["as_of"], 45)
    analyses = analyser_tous(entree.flux, AS_OF, params, [])
    renseigner_prochaines_dates(analyses, AS_OF)
    sortie = predict(donnees["flux"], donnees["solde_actuel"], donnees["as_of"], params=params)
    D = sortie["horizon_effectif_jours"]
    B = simuler_soldes(analyses, entree.solde_actuel, AS_OF, D, params)

    discrets = [e for e in sortie["evenements_prevus"] if e["nature"] == "discret"]
    variables = [fa for fa in analyses if fa.type == "variable" and fa.actif > SEUIL_ACTIF]
    N = params.n_sim
    for j in range(D + 1):
        d = AS_OF + timedelta(days=j)
        analytique = entree.solde_actuel
        analytique += sum(e["esperance"] for e in discrets if date.fromisoformat(e["date"]) <= d)
        analytique += sum(fa.actif * fa.a * fa.lam * j for fa in variables)
        col = B[:, j]
        assert sortie["timeline"][j]["esperance"] == pytest.approx(col.mean(), abs=0.01)
        assert abs(col.mean() - analytique) < 4 * col.std() / math.sqrt(N) + 1, d


# ---------------------------------------------------------------- 10 : cohérence des distributions


def test_10_sommes_a_un(sortie):
    for ligne in sortie["grille_densite"]["probabilites"]:
        assert sum(ligne) == pytest.approx(1.0, abs=1e-9)
    assert len(sortie["grille_densite"]["probabilites"]) == len(sortie["grille_densite"]["dates"])
    assert len(sortie["grille_densite"]["montants"]) == Params().n_bins
    for f in sortie["flux"]:
        assert sum(f["statut"].values()) == pytest.approx(1.0, abs=1e-5)
        assert sum(f["changement"]["distribution"].values()) == pytest.approx(1.0, abs=1e-5)


def test_timeline_jour0_certain(sortie):
    t0 = sortie["timeline"][0]
    assert t0["date"] == "2026-09-30"
    assert t0["esperance"] == t0["q05"] == t0["q95"] == 800.0
    assert t0["p_decouvert"] == 0.0


def test_evenements_tries_et_proba(sortie):
    evs = sortie["evenements_prevus"]
    cles = [e.get("date") or e["date_debut"] for e in evs]
    assert cles == sorted(cles)
    loyers = [e for e in evs if e["flux_id"] == "F003"]
    assert [e["date"] for e in loyers] == ["2026-10-05", "2026-11-05"]
    assert loyers[1]["probabilite"] == pytest.approx(loyers[0]["probabilite"] * 0.95, rel=1e-5)
    assert not any(e["flux_id"] in ("F002", "F004") for e in evs)


def test_evidences(sortie):
    codes = {e["code"] for e in _flux(sortie, "F002")["evidences"]}
    assert {"OCCURRENCES", "PERIODE", "REGULARITE", "PRIOR", "P_RECURRENT", "RETARD", "P_ETEINT", "RECENT"} <= codes
    assert "TAUX" in {e["code"] for e in _flux(sortie, "F005")["evidences"]}


# ---------------------------------------------------------------- 11 : validation


@pytest.mark.parametrize("mutation", [
    lambda f: f.__setitem__("dates_apparition", []),
    lambda f: f.__setitem__("montant", float("nan")),
    lambda f: f.__setitem__("montant", float("inf")),
    lambda f: f.__setitem__("montant", 2e7),
    lambda f: f.__setitem__("dates_apparition", ["30/09/2026"]),
    lambda f: f.__setitem__("dates_apparition", ["2026-02-30"]),
    lambda f: f.__setitem__("dates_apparition", [1790000000]),
    lambda f: f.__setitem__("dates_apparition", ["2026-01-01"] * 1001),
])
def test_11_validation(donnees, mutation):
    flux = copy.deepcopy(donnees["flux"])
    mutation(flux[0])
    with pytest.raises(ValueError):
        predict(flux, 800, "2026-09-30")


def test_11_validation_globale(donnees):
    with pytest.raises(ValueError):
        predict(donnees["flux"] * 400, 800, "2026-09-30")  # 2400 flux
    with pytest.raises(ValueError):
        predict(donnees["flux"], 800, "2026-09-30", horizon_jours=0)
    with pytest.raises(ValueError):
        predict(donnees["flux"], 800, "2026-09-30", horizon_jours=366)
    with pytest.raises(ValueError):
        predict(donnees["flux"], 800, "pas une date")


def test_warnings_dates_futures_et_doublons(donnees):
    flux = copy.deepcopy(donnees["flux"])
    flux[5]["dates_apparition"] += ["2026-09-15", "2026-10-15"]
    s = predict(flux, 800, "2026-09-30")
    assert any("double" in w for w in s["warnings"])
    assert any("postérieure" in w for w in s["warnings"])
    assert _flux(s, "F006")["prochaine_date"] == "2026-10-15"


# ---------------------------------------------------------------- 12 : déterminisme


def test_12_determinisme(donnees, sortie):
    s2 = predict(donnees["flux"], donnees["solde_actuel"], donnees["as_of"])
    assert json.dumps(s2, sort_keys=True) == json.dumps(sortie, sort_keys=True)


def test_cli(tmp_path, capsys):
    out = tmp_path / "output.json"
    code = cli_main([str(FIXTURE), "--solde", "800", "--as-of", "2026-09-30", "--out", str(out), "-q"])
    assert code == 0
    s = json.loads(out.read_text(encoding="utf-8"))
    assert s["resume"]["date_alerte"] == "2026-10-05"


def test_cli_erreur_sans_trace(tmp_path, capsys):
    mauvais = tmp_path / "bad.json"
    mauvais.write_text('[{"montant": "NaN", "dates_apparition": []}]', encoding="utf-8")
    code = cli_main([str(mauvais), "--solde", "800", "--as-of", "2026-09-30"])
    assert code == 1
    err = capsys.readouterr().err
    assert "erreur" in err and "Traceback" not in err


def test_sans_flux():
    s = predict([], 100.0, "2026-09-30")
    assert s["resume"]["remuneration"] is None
    assert s["resume"]["date_alerte"] is None
    assert all(t["esperance"] == 100.0 for t in s["timeline"])
    assert np.allclose([sum(r) for r in s["grille_densite"]["probabilites"]], 1.0)
