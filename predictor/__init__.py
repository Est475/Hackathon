"""Module de prédiction Tectonic (défi KBC) : flux → statut probabiliste → distribution du solde."""

from __future__ import annotations

from datetime import date, timedelta

from .flux import analyser_tous
from .params import Params
from .projection import projeter, renseigner_prochaines_dates
from .resume import construire_resume, fin_de_mois, trouver_remuneration
from .schemas import SortiePrediction, valider_entree
from .simulation import construire_grille, construire_timeline, simuler_soldes

__all__ = ["predict", "Params"]


def predict(flux: list[dict], solde_actuel: float, as_of: date | str,
            horizon_jours: int = 45, params: Params | None = None) -> dict:
    """Prédit la distribution du solde et qualifie chaque flux. Lève ValueError si l'entrée est invalide."""
    params = params or Params()
    entree = valider_entree(flux, solde_actuel, as_of, horizon_jours)
    as_of = entree.as_of
    warnings: list[str] = []

    analyses = analyser_tous(entree.flux, as_of, params, warnings)
    renseigner_prochaines_dates(analyses, as_of)

    # Horizon effectif : couvre la fin de mois et la veille de rémunération
    D = max(entree.horizon_jours, (fin_de_mois(as_of) - as_of).days)
    rem = trouver_remuneration(analyses)
    if rem is not None:
        D = max(D, (rem.prochaine_date - timedelta(days=1) - as_of).days)

    evenements = projeter(analyses, as_of, D, params)
    B = simuler_soldes(analyses, entree.solde_actuel, as_of, D, params)
    timeline = construire_timeline(B, as_of, evenements)
    grille = construire_grille(B, as_of, params.n_bins)
    resume = construire_resume(analyses, timeline, as_of, params)

    sortie = {
        "as_of": as_of.isoformat(),
        "solde_actuel": entree.solde_actuel,
        "horizon_jours": entree.horizon_jours,
        "horizon_effectif_jours": D,
        "parametres": params.to_dict(),
        "flux": [fa.to_output() for fa in analyses],
        "evenements_prevus": evenements,
        "timeline": timeline,
        "grille_densite": grille,
        "resume": resume,
        "warnings": warnings,
    }
    return SortiePrediction.model_validate(sortie).model_dump(mode="json")
