"""Étapes A–D : caractérisation temporelle, P(récurrent), extinction, type de changement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from dateutil.relativedelta import relativedelta

from .params import Params
from .schemas import FluxEntree

MOIS_MOYEN = 30.44

# (borne basse, borne haute, libellé du pas, adjectif, unité, quantité, P_jours)
PERIODES_CALENDAIRES = [
    (6, 8, "7 jours", "hebdomadaire", "jours", 7, 7.0),
    (12, 16, "14 jours", "bimensuel", "jours", 14, 14.0),
    (26, 35, "1 mois", "mensuel", "mois", 1, 30.44),
    (85, 97, "3 mois", "trimestriel", "mois", 3, 91.31),
    (350, 380, "1 an", "annuel", "mois", 12, 365.25),
]


@dataclass(frozen=True)
class Pas:
    libelle: str
    adjectif: str
    unite: str   # "jours" ou "mois"
    quantite: int
    jours: float

    def ajouter(self, d: date, k: int) -> date:
        """d + k pas ; les pas mensuels sont ancrés sur le jour du mois de d (clampé)."""
        if self.unite == "mois":
            return d + relativedelta(months=self.quantite * k)
        return d + timedelta(days=self.quantite * k)


def arrondir_periode(p_brute: float) -> Pas:
    for bas, haut, libelle, adj, unite, qte, jours in PERIODES_CALENDAIRES:
        if bas <= p_brute <= haut:
            return Pas(libelle, adj, unite, qte, jours)
    n = max(int(round(p_brute)), 1)
    return Pas(f"{n} jours", f"tous les {n} j", "jours", n, float(n))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def _survie_normale(x: float) -> float:
    """1 - Φ(x), numériquement stable pour x grand."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def _f(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


@dataclass
class AnalyseFlux:
    flux_id: str
    marchand: str
    categorie: str
    recette: bool
    charge_fixe: bool
    a: float                      # montant signé moyen
    sigma_a: float
    dates: list[date]
    type: str = "ponctuel"
    pas: Pas | None = None
    p_recurrent: float = 0.0
    lam: float | None = None      # taux par jour (variable)
    q: float = 0.0                # P(éteint | récurrent, silence)
    date_attendue: date | None = None
    prochaine_date: date | None = None
    statut: dict[str, float] = field(default_factory=dict)
    changement: dict[str, float] = field(default_factory=dict)
    principal: str = ""
    est_perturbateur: bool = False
    evidences: list[dict] = field(default_factory=list)

    @property
    def actif(self) -> float:
        return self.statut.get("actif", 0.0)

    def ev(self, code: str, message: str, valeur=None) -> None:
        self.evidences.append({"code": code, "message": message, "valeur": valeur})

    def impact(self) -> dict:
        """Impact pour le résumé (section 8)."""
        if self.principal == "evenement_ponctuel":
            return {"impact_ponctuel": round(self.a, 2)}
        if self.principal in ("nouvelle_habitude", "habitude_eteinte"):
            signe = 1 if self.principal == "nouvelle_habitude" else -1
            if self.type == "variable" and self.lam is not None:
                return {"impact_mensuel": round(signe * self.a * self.lam * MOIS_MOYEN, 2)}
            if self.pas is not None:
                return {"impact_mensuel": round(signe * self.a * MOIS_MOYEN / self.pas.jours, 2)}
        return {}

    def to_output(self) -> dict:
        return {
            "flux_id": self.flux_id,
            "marchand": self.marchand,
            "categorie": self.categorie,
            "recette": self.recette,
            "montant": round(self.a, 2),
            "sigma_montant": round(self.sigma_a, 2),
            "type": self.type,
            "periode": {"pas": self.pas.libelle, "jours": self.pas.jours} if self.pas else None,
            "taux_journalier": round(self.lam, 6) if self.lam is not None else None,
            "p_recurrent": round(self.p_recurrent, 6),
            "statut": {k: round(v, 6) for k, v in self.statut.items()},
            "changement": {
                "distribution": {k: round(v, 6) for k, v in self.changement.items()},
                "principal": self.principal,
                "est_perturbateur": self.est_perturbateur,
            },
            "date_attendue": self.date_attendue.isoformat() if self.date_attendue else None,
            "prochaine_date": self.prochaine_date.isoformat() if self.prochaine_date else None,
            "evidences": self.evidences,
        }


def _preparer_dates(f: FluxEntree, flux_id: str, as_of: date, warnings: list[str]) -> list[date]:
    brutes = sorted(f.dates_apparition)
    uniques = sorted(set(brutes))
    if len(uniques) < len(brutes):
        warnings.append(f"{flux_id}: {len(brutes) - len(uniques)} date(s) en double ignorée(s)")
    futures = [d for d in uniques if d > as_of]
    if futures:
        warnings.append(f"{flux_id}: {len(futures)} date(s) postérieure(s) à as_of ignorée(s)")
    dates = [d for d in uniques if d <= as_of]
    if dates:
        if f.premiere_apparition is not None and f.premiere_apparition != dates[0]:
            warnings.append(f"{flux_id}: premiere_apparition incohérente, recalculée à {dates[0].isoformat()}")
        if f.derniere_apparition is not None and f.derniere_apparition != dates[-1]:
            warnings.append(f"{flux_id}: derniere_apparition incohérente, recalculée à {dates[-1].isoformat()}")
    return dates


def analyser_flux(f: FluxEntree, flux_id: str, as_of: date, params: Params,
                  warnings: list[str]) -> AnalyseFlux | None:
    dates = _preparer_dates(f, flux_id, as_of, warnings)
    if not dates:
        warnings.append(f"{flux_id}: aucune date antérieure ou égale à as_of, flux ignoré")
        return None

    signe = 1.0 if f.recette else -1.0
    a = signe * abs(f.montant)
    fa = AnalyseFlux(flux_id=flux_id, marchand=f.marchand, categorie=f.categorie, recette=f.recette,
                     charge_fixe=f.charge_fixe, a=a, sigma_a=0.0, dates=dates)
    n = len(dates)
    d1, dn = dates[0], dates[-1]

    # ---- Étape A : caractérisation temporelle
    fa.ev("OCCURRENCES",
          f"{n} occurrence{'s' if n > 1 else ''}"
          + (f" du {d1.isoformat()} au {dn.isoformat()}" if n > 1 else f" le {d1.isoformat()}"), n)
    deltas = [(dates[j + 1] - dates[j]).days for j in range(n - 1)]
    if n >= 2:
        p_brute: float | None = float(np.median(deltas))
    elif f.charge_fixe:
        p_brute = 30.0
    else:
        p_brute = None

    k_reg = k_irr = 0
    if p_brute is not None:
        fa.pas = arrondir_periode(p_brute)
        if n >= 2:
            fa.ev("PERIODE", f"{n} occurrences, intervalle médian {p_brute:g} j → {fa.pas.adjectif}", p_brute)
        else:
            fa.ev("PERIODE", f"une seule occurrence, charge fixe → {fa.pas.adjectif} supposé", p_brute)
        tau = max(params.tol_rel * fa.pas.jours, 1.0)
        k_reg = sum(1 for dl in deltas if abs(dl - fa.pas.jours) <= tau)
        k_irr = (n - 1) - k_reg
        if n >= 2:
            fa.ev("REGULARITE",
                  f"{k_reg}/{n - 1} intervalles réguliers (à ± {tau:.1f} j de {fa.pas.jours:g} j)", k_reg)
    else:
        fa.ev("PERIODE", "une seule occurrence, aucune période identifiable", None)

    # ---- Étape B : P(récurrent)
    pi = params.prior_fixe if f.charge_fixe else params.prior_libre
    fa.ev("PRIOR", f"{'charge fixe' if f.charge_fixe else 'dépense libre'} → P(récurrent) a priori = {_f(pi)}", pi)
    if n == 1:
        p_rec = pi
    else:
        lo = (_logit(pi) + k_reg * math.log(params.rho1 / params.rho0)
              + k_irr * math.log((1 - params.rho1) / (1 - params.rho0)))
        p_rec = min(max(_sigmoid(lo), 0.01), 0.99)
    fa.p_recurrent = p_rec
    if n >= 2:
        fa.ev("P_RECURRENT", f"{k_reg} intervalle(s) régulier(s), {k_irr} irrégulier(s) → P(récurrent) = {_f(p_rec)}", p_rec)
    else:
        fa.ev("P_RECURRENT", f"une seule observation → P(récurrent) = a priori = {_f(p_rec)}", p_rec)

    # ---- Type de flux
    if p_rec >= 0.5 and fa.pas is not None:
        fa.type = "periodique"
    elif p_rec < 0.5 and n >= 3:
        fa.type = "variable"
    else:
        fa.type = "ponctuel"

    # ---- Dispersion du montant
    if f.montants is not None and len(f.montants) >= 3:
        fa.sigma_a = float(np.std(np.abs(np.asarray(f.montants, dtype=float)), ddof=1))
    else:
        if fa.type == "variable":
            cv = params.cv_variable
        elif f.charge_fixe:
            cv = params.cv_fixe
        else:
            cv = params.cv_periodique
        fa.sigma_a = cv * abs(a)

    # ---- Étape C : extinction
    h = params.h
    if fa.type == "periodique":
        assert fa.pas is not None
        d_star = fa.pas.ajouter(dn, 1)
        fa.date_attendue = d_star
        retard = (as_of - d_star).days
        sigma_d = max(0.15 * fa.pas.jours, 2.0)
        z = retard / sigma_d
        s = _survie_normale(z)
        fa.q = h / (h + (1 - h) * s)
        if retard > 0:
            msg = f"attendu le {d_star.isoformat()}, en retard de {retard} j ({z:.1f} σ) → P(éteint) = {_f(fa.q)}"
        else:
            msg = f"attendu le {d_star.isoformat()} (dans {-retard} j), pas encore dû → P(éteint) = {_f(fa.q)}"
        fa.ev("RETARD", msg, retard)
        fa.statut = {"ponctuel": 1 - p_rec, "actif": p_rec * (1 - fa.q), "eteint": p_rec * fa.q}
    elif fa.type == "variable":
        fa.lam = (n - 1) / (dn - d1).days
        g = (as_of - dn).days
        s = math.exp(-fa.lam * g)
        fa.q = h / (h + (1 - h) * s)
        fa.ev("TAUX",
              f"{fa.lam:.2f} occurrence/jour (≈ {fa.lam * MOIS_MOYEN:.1f} par mois), "
              f"dernière il y a {g} j → P(silence | actif) = {_f(s)}", round(fa.lam, 4))
        fa.statut = {"ponctuel": 0.0, "actif": 1 - fa.q, "eteint": fa.q}
    else:
        fa.q = 0.0
        fa.statut = {"ponctuel": 1 - p_rec, "actif": p_rec, "eteint": 0.0}
    if fa.type != "ponctuel":
        fa.ev("P_ETEINT", f"P(habitude éteinte | récurrent) = {_f(fa.q)} → P(active) = {_f(fa.actif)}", round(fa.q, 6))

    # ---- Étape D : type de changement
    limite = as_of - timedelta(days=params.fenetre_recente)
    recent_first = d1 >= limite
    recent_last = dn >= limite
    age_first = (as_of - d1).days
    age_last = (as_of - dn).days
    fw = params.fenetre_recente
    if fa.type == "ponctuel":
        comp = "<" if recent_last else "≥"
        suite = "événement récent" if recent_last else "événement ancien"
        fa.ev("RECENT", f"dernière apparition il y a {age_last} j ({comp} {fw} j) → {suite}", age_last)
    else:
        comp = "<" if recent_first else "≥"
        suite = "changement récent" if recent_first else "habitude installée"
        fa.ev("RECENT", f"première apparition il y a {age_first} j ({comp} {fw} j) → {suite}", age_first)

    fa.changement = {
        ("nouvelle_habitude" if recent_first else "stable"): fa.statut["actif"],
        "habitude_eteinte": fa.statut["eteint"],
        ("evenement_ponctuel" if recent_last else "evenement_ancien"): fa.statut["ponctuel"],
    }
    fa.principal = max(fa.changement, key=fa.changement.__getitem__)
    fa.est_perturbateur = fa.principal not in ("stable", "evenement_ancien")
    return fa


def analyser_tous(flux: list[FluxEntree], as_of: date, params: Params,
                  warnings: list[str]) -> list[AnalyseFlux]:
    analyses = []
    for i, f in enumerate(flux, start=1):
        fa = analyser_flux(f, f"F{i:03d}", as_of, params, warnings)
        if fa is not None:
            analyses.append(fa)
    return analyses
