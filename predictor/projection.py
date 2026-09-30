"""Étape E : table des événements prévus (date × montant × probabilité)."""

from __future__ import annotations

import math
from datetime import date, timedelta

from .flux import AnalyseFlux
from .params import Params

SEUIL_ACTIF = 0.01
TAILLE_FENETRE = 7


def dates_projetees(fa: AnalyseFlux, as_of: date, fin: date | None = None) -> list[date]:
    """Occurrences futures d'un flux périodique dans ]as_of, fin] (fin=None → la première seulement).

    Les occurrences attendues mais pas encore observées (d* ≤ as_of) sont ramenées à as_of + 1.
    """
    if fa.type != "periodique" or fa.pas is None:
        return []
    dn = fa.dates[-1]
    k = 1
    d = fa.pas.ajouter(dn, k)
    out: list[date] = []
    if d <= as_of:
        while d <= as_of:
            k += 1
            d = fa.pas.ajouter(dn, k)
        out.append(as_of + timedelta(days=1))
        if d == out[0]:
            k += 1
            d = fa.pas.ajouter(dn, k)
    if fin is None:
        return out[:1] or [d]
    out = [x for x in out if x <= fin]
    while d <= fin:
        out.append(d)
        k += 1
        d = fa.pas.ajouter(dn, k)
    return out


def renseigner_prochaines_dates(analyses: list[AnalyseFlux], as_of: date) -> None:
    for fa in analyses:
        if fa.type == "periodique":
            fa.prochaine_date = dates_projetees(fa, as_of)[0]


def projeter(analyses: list[AnalyseFlux], as_of: date, D: int, params: Params) -> list[dict]:
    fin = as_of + timedelta(days=D)
    lignes: list[tuple[date, str, dict]] = []
    for fa in analyses:
        if fa.actif <= SEUIL_ACTIF:
            continue
        base = {"flux_id": fa.flux_id, "marchand": fa.marchand, "categorie": fa.categorie}
        if fa.type == "periodique":
            for k, d in enumerate(dates_projetees(fa, as_of, fin)):
                p = fa.actif * (1 - params.h) ** k
                lignes.append((d, fa.flux_id, {
                    **base, "nature": "discret", "date": d.isoformat(),
                    "montant": round(fa.a, 2), "sigma_montant": round(fa.sigma_a, 2),
                    "probabilite": round(p, 6), "esperance": round(p * fa.a, 2),
                }))
        elif fa.type == "variable" and fa.lam is not None:
            debut = 1
            while debut <= D:
                fin_f = min(debut + TAILLE_FENETRE - 1, D)
                jours = fin_f - debut + 1
                m = fa.a * fa.lam * jours
                # Poisson composé : Var = λ·t·E[X²]
                sigma = math.sqrt(fa.lam * jours * (fa.a ** 2 + fa.sigma_a ** 2))
                d0 = as_of + timedelta(days=debut)
                lignes.append((d0, fa.flux_id, {
                    **base, "nature": "continu",
                    "date_debut": d0.isoformat(),
                    "date_fin": (as_of + timedelta(days=fin_f)).isoformat(),
                    "montant": round(m, 2), "sigma_montant": round(sigma, 2),
                    "probabilite": round(fa.actif, 6), "esperance": round(fa.actif * m, 2),
                }))
                debut = fin_f + 1
    lignes.sort(key=lambda t: (t[0], t[1]))
    return [{"id": f"E{i:04d}", **ligne} for i, (_, _, ligne) in enumerate(lignes, start=1)]
