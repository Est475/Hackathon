"""Tests d'acceptation de predictor_standalone.py sur la fixture de Sarah (déménagement Liège → Namur)."""

from __future__ import annotations

import copy
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import predictor_standalone as ps

RACINE = Path(__file__).resolve().parent.parent
FIXTURE = RACINE / "fixtures" / "sarah.json"
AS_OF = date(2026, 9, 30)


@pytest.fixture(scope="module")
def donnees() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def flux(donnees) -> dict[str, ps.Flux]:
    e = ps.valider(donnees["flux"], 800, "2026-09-30", 45)
    return {f"F{i:03d}": ps.analyser(f, AS_OF, ps.Params()) for i, f in enumerate(e.flux, start=1)}


@pytest.fixture(scope="module")
def resultat(donnees):
    return ps.predict_scenarios(donnees["flux"], donnees["solde_actuel"], donnees["as_of"])


def _jour(t: np.ndarray, d: str) -> np.void:
    return t[int(np.argmax(t["date"] == np.datetime64(d)))]


def _prochaine(fl: ps.Flux) -> str:
    return ps.dates_futures(fl, AS_OF, AS_OF + timedelta(days=60))[0].isoformat()


# ---------------------------------------------------------------- classification des flux


def test_salaire_actif(flux):
    fl = flux["F001"]
    assert fl.type == "periodique" and fl.actif > 0.9
    assert _prochaine(fl) == "2026-10-28"


def test_ancien_loyer_eteint(flux):
    assert flux["F002"].type == "periodique" and flux["F002"].actif < 0.05


def test_nouveau_loyer(flux):
    assert flux["F003"].actif > 0.8
    assert _prochaine(flux["F003"]) == "2026-10-05"


def test_demenagement_ponctuel(flux):
    assert flux["F004"].type == "ponctuel"
    assert flux["F004"].actif == pytest.approx(0.20)


def test_courses_variables(flux):
    assert flux["F005"].type == "variable" and 0.2 <= flux["F005"].lam <= 0.3


def test_netflix_stable(flux):
    assert flux["F006"].type == "periodique" and flux["F006"].actif > 0.9


# ---------------------------------------------------------------- timeline


def test_forme_et_jour0(resultat):
    t, B = resultat
    assert t.dtype == ps.DTYPE_TIMELINE
    assert B.shape == (ps.Params().n_sim, len(t))
    assert str(t["date"][0]) == "2026-09-30" and str(t["date"][-1]) == "2026-11-14"
    assert t["esperance"][0] == t["q05"][0] == t["q95"][0] == 800.0 and t["p_decouvert"][0] == 0.0
    for a, b in [("q05", "q25"), ("q25", "q50"), ("q50", "q75"), ("q75", "q95")]:
        assert np.all(t[a] <= t[b])


def test_alerte_et_p_decouvert(resultat):
    t, _ = resultat
    alerte = next(str(r["date"]) for r in t if r["p_decouvert"] >= ps.SEUIL_ALERTE)
    assert alerte == "2026-10-05"
    assert 0.80 <= _jour(t, "2026-10-10")["p_decouvert"] <= 0.95
    assert _jour(t, "2026-10-29")["p_decouvert"] < 0.08


def test_controle_monte_carlo(resultat, flux):
    """Moyenne simulée vs espérance analytique, jour par jour."""
    t, B = resultat
    h, N = ps.Params().h, ps.Params().n_sim
    fin = AS_OF + timedelta(days=len(t) - 1)
    for j in range(len(t)):
        d = AS_OF + timedelta(days=j)
        attendu = 800.0
        for fl in flux.values():
            if fl.actif <= 0.01:
                continue
            if fl.type == "periodique":
                attendu += sum(fl.actif * (1 - h) ** k * fl.a
                               for k, dk in enumerate(ps.dates_futures(fl, AS_OF, fin)) if dk <= d)
            elif fl.type == "variable":
                attendu += fl.actif * fl.a * fl.lam * j
        assert abs(B[:, j].mean() - attendu) < 4 * B[:, j].std() / math.sqrt(N) + 1, d


def test_determinisme(donnees, resultat):
    t2 = ps.predict(donnees["flux"], donnees["solde_actuel"], donnees["as_of"])
    assert np.array_equal(t2, resultat[0])


def test_sans_flux():
    t = ps.predict([], 100.0, "2026-09-30")
    assert np.all(t["esperance"] == 100.0) and np.all(t["p_decouvert"] == 0)


# ---------------------------------------------------------------- entrées


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
def test_validation(donnees, mutation):
    fl = copy.deepcopy(donnees["flux"])
    mutation(fl[0])
    with pytest.raises(ValueError):
        ps.predict(fl, 800, "2026-09-30")


def test_validation_globale(donnees):
    for kwargs in ({"horizon_jours": 0}, {"horizon_jours": 366}):
        with pytest.raises(ValueError):
            ps.predict(donnees["flux"], 800, "2026-09-30", **kwargs)
    with pytest.raises(ValueError):
        ps.predict(donnees["flux"] * 400, 800, "2026-09-30")  # 2400 flux
    with pytest.raises(ValueError):
        ps.predict(donnees["flux"], 800, "pas une date")


def test_dates_avec_heure_acceptees():
    """Format produit par l'IA amont (fixtures/donnees_test.json) : l'heure est ignorée."""
    fl = json.loads((RACINE / "fixtures" / "donnees_test.json").read_text(encoding="utf-8"))
    t = ps.predict(fl, 500, "2026-09-30")
    assert t["p_decouvert"].max() < ps.SEUIL_ALERTE


# ---------------------------------------------------------------- CLI et curseur


def test_cli_npy(tmp_path, resultat):
    npy = tmp_path / "t.npy"
    assert ps.main([str(FIXTURE), "--out", str(npy)]) == 0
    assert np.array_equal(np.load(npy, allow_pickle=False), resultat[0])


def test_cli_erreur_sans_trace(tmp_path, capsys):
    mauvais = tmp_path / "bad.json"
    mauvais.write_text('[{"montant": "NaN", "dates_apparition": []}]', encoding="utf-8")
    assert ps.main([str(mauvais), "--solde", "800", "--as-of", "2026-09-30"]) == 1
    err = capsys.readouterr().err
    assert "erreur" in err and "Traceback" not in err


def test_curseur_temporel(resultat):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, B = resultat
    fig, slider = ps.figure_curseur(t, B)
    slider.set_val(10)
    assert slider.valtext.get_text() == "2026-10-10"
    tableau = next(ax.tables[0] for ax in fig.axes if ax.tables)
    assert "10/10" in [c.get_text().get_text() for c in tableau.get_celld().values()]
    textes = [txt.get_text() for ax in fig.axes for txt in ax.texts]
    assert any("Le 10/10 : 86 % de risque" in x for x in textes)
    plt.close(fig)
