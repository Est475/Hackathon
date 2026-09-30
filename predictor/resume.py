"""Section 12 : résumé (rémunération, points clés, alerte, perturbateurs)."""

from __future__ import annotations

from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

from .flux import AnalyseFlux
from .params import Params


def trouver_remuneration(analyses: list[AnalyseFlux]) -> AnalyseFlux | None:
    candidats = [fa for fa in analyses
                 if fa.type == "periodique" and fa.recette and fa.actif >= 0.5 and fa.prochaine_date]
    return max(candidats, key=lambda fa: abs(fa.a), default=None)


def fin_de_mois(as_of: date) -> date:
    fin = as_of + relativedelta(day=31)
    if fin == as_of:
        fin = as_of + relativedelta(months=1, day=31)
    return fin


def _point(timeline: list[dict], as_of: date, d: date) -> dict | None:
    j = (d - as_of).days
    if not 0 <= j < len(timeline):
        return None
    t = timeline[j]
    return {"date": t["date"], "esperance": t["esperance"], "q05": t["q05"],
            "q95": t["q95"], "p_decouvert": t["p_decouvert"]}


def construire_resume(analyses: list[AnalyseFlux], timeline: list[dict], as_of: date,
                      params: Params) -> dict:
    rem = trouver_remuneration(analyses)
    remuneration = None
    veille = None
    if rem is not None:
        remuneration = {"flux_id": rem.flux_id, "marchand": rem.marchand,
                        "date": rem.prochaine_date.isoformat(), "montant": round(rem.a, 2)}
        veille = _point(timeline, as_of, max(rem.prochaine_date - timedelta(days=1), as_of))

    j_max = max(range(len(timeline)), key=lambda j: (timeline[j]["p_decouvert"], -j))
    date_alerte = next((t["date"] for t in timeline if t["p_decouvert"] >= params.seuil_alerte), None)

    perturbateurs = []
    for fa in analyses:
        if not fa.est_perturbateur:
            continue
        perturbateurs.append({
            "flux_id": fa.flux_id, "marchand": fa.marchand, "categorie": fa.categorie,
            "changement": fa.principal,
            "probabilite": round(fa.changement[fa.principal], 6),
            **fa.impact(),
        })
    perturbateurs.sort(key=lambda p: -p["probabilite"])

    return {
        "remuneration": remuneration,
        "fin_de_mois": _point(timeline, as_of, fin_de_mois(as_of)),
        "veille_remuneration": veille,
        "p_decouvert_max": {"valeur": timeline[j_max]["p_decouvert"], "date": timeline[j_max]["date"]},
        "date_alerte": date_alerte,
        "perturbateurs": perturbateurs,
    }
