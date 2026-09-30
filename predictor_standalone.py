"""Prédiction du solde — version autonome en un seul fichier.

Même modèle que le package `predictor/` (étapes A–C, projection, Monte Carlo), mais la seule
sortie est la timeline du solde sous forme de tableau numpy structuré, une ligne par jour :

    date (datetime64[D]), esperance, q05, q25, q50, q75, q95, p_decouvert

Usage :
    python predictor_standalone.py fixtures/sarah.json                       # affiche le tableau
    python predictor_standalone.py fixtures/sarah.json --out timeline.npy    # l'enregistre
    python predictor_standalone.py flux.json --solde 800 --as-of 2026-09-30

En Python :
    from predictor_standalone import predict
    t = predict(flux, 800, "2026-09-30")
    t["date"], t["esperance"], t["p_decouvert"]

Dépendances : numpy, python-dateutil, pydantic v2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Annotated

import numpy as np
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, BeforeValidator, Field, ValidationError, field_validator

# =========================================================================== paramètres


@dataclass(frozen=True)
class Params:
    prior_fixe: float = 0.8          # P(récurrent) a priori si charge_fixe
    prior_libre: float = 0.2         # P(récurrent) a priori sinon
    rho1: float = 0.9                # P(intervalle régulier | périodique)
    rho0: float = 0.3                # P(intervalle régulier | non périodique)
    tol_rel: float = 0.15            # tolérance de régularité, relative à la période
    h: float = 0.05                  # P(qu'une habitude s'arrête) par période
    cv_fixe: float = 0.02            # coef. de variation du montant, charge fixe
    cv_periodique: float = 0.15      # idem, périodique non fixe
    cv_variable: float = 0.30        # idem, flux variable
    n_sim: int = 5000                # scénarios Monte Carlo
    seed: int = 42                   # graine (reproductibilité)

    def __post_init__(self) -> None:
        for nom in ("prior_fixe", "prior_libre", "rho1", "rho0", "h"):
            if not 0.0 < getattr(self, nom) < 1.0:
                raise ValueError(f"Paramètre {nom} doit être dans ]0, 1[")
        for nom in ("tol_rel", "cv_fixe", "cv_periodique", "cv_variable"):
            if getattr(self, nom) < 0:
                raise ValueError(f"Paramètre {nom} doit être positif")
        if not 1 <= self.n_sim <= 100_000:
            raise ValueError("Paramètre n_sim doit être dans [1, 100000]")


# =========================================================================== validation des entrées

MAX_FLUX, MAX_DATES, MAX_MONTANT, MAX_SOLDE = 2000, 1000, 1e7, 1e9


def _parse_date_iso(v: object) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, str) and len(v) <= 10:
        try:
            return date.fromisoformat(v)
        except ValueError:
            pass
    raise ValueError("date ISO attendue (AAAA-MM-JJ)")


DateISO = Annotated[date, BeforeValidator(_parse_date_iso)]


def _verifier_montant(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("montant non fini")
    if abs(v) > MAX_MONTANT:
        raise ValueError(f"montant hors limites (|montant| > {MAX_MONTANT:g})")
    return v


class FluxEntree(BaseModel):
    categorie: str = Field(default="", max_length=200)
    marchand: str = Field(default="", max_length=200)
    montant: float
    charge_fixe: bool = False
    recette: bool = False
    dates_apparition: list[DateISO] = Field(min_length=1, max_length=MAX_DATES)
    montants: list[float] | None = Field(default=None, max_length=MAX_DATES)

    @field_validator("montant")
    @classmethod
    def _montant(cls, v: float) -> float:
        return _verifier_montant(v)

    @field_validator("montants")
    @classmethod
    def _montants(cls, v: list[float] | None) -> list[float] | None:
        for x in v or []:
            _verifier_montant(x)
        return v


class Entree(BaseModel):
    flux: list[FluxEntree] = Field(max_length=MAX_FLUX)
    solde_actuel: float
    as_of: DateISO
    horizon_jours: int = Field(default=45, ge=1, le=365)

    @field_validator("solde_actuel")
    @classmethod
    def _solde(cls, v: float) -> float:
        if not math.isfinite(v) or abs(v) > MAX_SOLDE:
            raise ValueError("solde non fini ou hors limites")
        return v


def valider(flux, solde_actuel, as_of, horizon_jours) -> Entree:
    if isinstance(flux, list) and len(flux) > MAX_FLUX:
        raise ValueError(f"Entrée invalide : plus de {MAX_FLUX} flux")
    try:
        return Entree(flux=flux, solde_actuel=solde_actuel, as_of=as_of, horizon_jours=horizon_jours)
    except ValidationError as e:
        details = [".".join(map(str, err["loc"])) + ": " + err["msg"].removeprefix("Value error, ")
                   for err in e.errors()[:10]]
        raise ValueError("Entrée invalide : " + " ; ".join(details)) from None


# =========================================================================== étapes A–C : analyse d'un flux

# (borne basse, borne haute, unité, quantité, P_jours)
PERIODES = [(6, 8, "jours", 7, 7.0), (12, 16, "jours", 14, 14.0), (26, 35, "mois", 1, 30.44),
            (85, 97, "mois", 3, 91.31), (350, 380, "mois", 12, 365.25)]


@dataclass
class Flux:
    recette: bool
    a: float                  # montant signé moyen
    sigma_a: float
    dates: list[date]
    type: str                 # periodique | variable | ponctuel
    actif: float              # P(l'habitude est active)
    pas: tuple[str, int, float] | None = None   # (unité, quantité, P_jours)
    lam: float | None = None  # taux par jour (variable)

    def date_k(self, k: int) -> date:
        """Dernière date observée + k pas (pas mensuels ancrés sur le jour du mois, clampés)."""
        unite, qte, _ = self.pas
        dn = self.dates[-1]
        return dn + (relativedelta(months=qte * k) if unite == "mois" else timedelta(days=qte * k))


def _arrondir_periode(p: float) -> tuple[str, int, float]:
    for bas, haut, unite, qte, jours in PERIODES:
        if bas <= p <= haut:
            return unite, qte, jours
    n = max(int(round(p)), 1)
    return "jours", n, float(n)


def analyser(f: FluxEntree, as_of: date, params: Params) -> Flux | None:
    dates = sorted({d for d in f.dates_apparition if d <= as_of})
    if not dates:
        return None
    n, d1, dn = len(dates), dates[0], dates[-1]
    a = (1.0 if f.recette else -1.0) * abs(f.montant)

    # A — période et régularité
    deltas = [(dates[j + 1] - dates[j]).days for j in range(n - 1)]
    p_brute = float(np.median(deltas)) if n >= 2 else (30.0 if f.charge_fixe else None)
    pas = _arrondir_periode(p_brute) if p_brute is not None else None
    k_reg = k_irr = 0
    if pas is not None:
        tau = max(params.tol_rel * pas[2], 1.0)
        k_reg = sum(1 for dl in deltas if abs(dl - pas[2]) <= tau)
        k_irr = (n - 1) - k_reg

    # B — P(récurrent), mise à jour bayésienne
    pi = params.prior_fixe if f.charge_fixe else params.prior_libre
    if n == 1:
        p_rec = pi
    else:
        lo = (math.log(pi / (1 - pi)) + k_reg * math.log(params.rho1 / params.rho0)
              + k_irr * math.log((1 - params.rho1) / (1 - params.rho0)))
        p_rec = min(max(1 / (1 + math.exp(-lo)), 0.01), 0.99)

    if p_rec >= 0.5 and pas is not None:
        typ = "periodique"
    elif p_rec < 0.5 and n >= 3:
        typ = "variable"
    else:
        typ = "ponctuel"

    if f.montants is not None and len(f.montants) >= 3:
        sigma_a = float(np.std(np.abs(np.asarray(f.montants, dtype=float)), ddof=1))
    else:
        cv = params.cv_variable if typ == "variable" else (params.cv_fixe if f.charge_fixe else params.cv_periodique)
        sigma_a = cv * abs(a)

    # C — extinction : q = P(éteint | récurrent, silence observé)
    fl = Flux(recette=f.recette, a=a, sigma_a=sigma_a, dates=dates, type=typ, actif=p_rec, pas=pas)
    h = params.h
    if typ == "periodique":
        retard = (as_of - fl.date_k(1)).days
        sigma_d = max(0.15 * pas[2], 2.0)
        s = 0.5 * math.erfc(retard / sigma_d / math.sqrt(2))   # 1 - Φ(retard / σ)
        fl.actif = p_rec * (1 - h / (h + (1 - h) * s))
    elif typ == "variable":
        fl.lam = (n - 1) / (dn - d1).days
        s = math.exp(-fl.lam * (as_of - dn).days)
        fl.actif = 1 - h / (h + (1 - h) * s)
    return fl


# =========================================================================== projection et simulation


def dates_futures(fl: Flux, as_of: date, fin: date) -> list[date]:
    """Occurrences dans ]as_of, fin] ; une occurrence en retard est ramenée à as_of + 1."""
    k, d, out = 1, fl.date_k(1), []
    if d <= as_of:
        while d <= as_of:
            k += 1
            d = fl.date_k(k)
        out.append(as_of + timedelta(days=1))
        if d == out[0]:
            k += 1
            d = fl.date_k(k)
    out = [x for x in out if x <= fin]
    while d <= fin:
        out.append(d)
        k += 1
        d = fl.date_k(k)
    return out


def _fin_de_mois(as_of: date) -> date:
    fin = as_of + relativedelta(day=31)
    return as_of + relativedelta(months=1, day=31) if fin == as_of else fin


def _horizon(flux: list[Flux], as_of: date, horizon_jours: int) -> int:
    """Horizon étendu pour couvrir la fin de mois et la veille de la prochaine rémunération."""
    D = max(horizon_jours, (_fin_de_mois(as_of) - as_of).days)
    salaires = [fl for fl in flux if fl.type == "periodique" and fl.recette and fl.actif >= 0.5]
    if salaires:
        rem = max(salaires, key=lambda fl: abs(fl.a))
        prochaine = dates_futures(rem, as_of, as_of + timedelta(days=400))[0]
        D = max(D, (prochaine - timedelta(days=1) - as_of).days)
    return D


def simuler(flux: list[Flux], solde: float, as_of: date, D: int, params: Params) -> np.ndarray:
    """Soldes simulés, forme (n_sim, D+1) ; colonne 0 = as_of (solde certain)."""
    rng = np.random.default_rng(params.seed)
    N = params.n_sim
    cash = np.zeros((N, D))
    fin = as_of + timedelta(days=D)
    for fl in flux:
        if fl.actif <= 0.01:
            continue
        if fl.type == "periodique":
            dates = dates_futures(fl, as_of, fin)
            if not dates:
                continue
            active = rng.random(N) < fl.actif
            K = rng.geometric(params.h, N)       # P(K > k) = (1-h)^k : l'habitude s'arrête après K périodes
            for k, d in enumerate(dates):
                present = active & (k < K)
                cash[:, (d - as_of).days - 1] += present * (fl.a + fl.sigma_a * rng.standard_normal(N))
        elif fl.type == "variable":
            active = rng.random(N) < fl.actif
            c = rng.poisson(fl.lam, (N, D))
            Z = rng.standard_normal((N, D))
            cash += active[:, None] * (c * fl.a + np.sqrt(c) * fl.sigma_a * Z)
    B = np.empty((N, D + 1))
    B[:, 0] = solde
    B[:, 1:] = solde + np.cumsum(cash, axis=1)
    return B


# =========================================================================== API

DTYPE_TIMELINE = np.dtype([("date", "datetime64[D]"), ("esperance", "f8"), ("q05", "f8"), ("q25", "f8"),
                           ("q50", "f8"), ("q75", "f8"), ("q95", "f8"), ("p_decouvert", "f8")])


def predict(flux: list[dict], solde_actuel: float, as_of: date | str,
            horizon_jours: int = 45, params: Params | None = None) -> np.ndarray:
    """Timeline du solde : tableau numpy structuré (DTYPE_TIMELINE), une ligne par jour depuis as_of."""
    params = params or Params()
    e = valider(flux, solde_actuel, as_of, horizon_jours)
    analyses = [fl for f in e.flux if (fl := analyser(f, e.as_of, params)) is not None]
    D = _horizon(analyses, e.as_of, e.horizon_jours)
    B = simuler(analyses, e.solde_actuel, e.as_of, D, params)

    t = np.empty(D + 1, dtype=DTYPE_TIMELINE)
    t["date"] = np.datetime64(e.as_of, "D") + np.arange(D + 1)
    t["esperance"] = B.mean(axis=0)
    for nom, q in zip(("q05", "q25", "q50", "q75", "q95"),
                      np.quantile(B, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)):
        t[nom] = q
    t["p_decouvert"] = (B < 0).mean(axis=0)
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Timeline du solde prévu (tableau numpy).")
    ap.add_argument("entree", help="JSON : liste de flux, ou objet {flux, solde_actuel?, as_of?}")
    ap.add_argument("--solde", type=float, help="solde actuel (sinon lu dans le fichier)")
    ap.add_argument("--as-of", dest="as_of", help="date de référence AAAA-MM-JJ (sinon lue dans le fichier)")
    ap.add_argument("--horizon", type=int, help="horizon en jours (défaut 45)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="enregistre le tableau en .npy (sinon l'affiche)")
    args = ap.parse_args(argv)
    try:
        if os.path.getsize(args.entree) > 20 * 1024 * 1024:
            raise ValueError("fichier d'entrée trop volumineux")
        with open(args.entree, encoding="utf-8") as fh:
            try:
                donnees = json.load(fh)
            except json.JSONDecodeError as err:
                raise ValueError(f"JSON invalide (ligne {err.lineno}, colonne {err.colno})") from None
        meta = donnees if isinstance(donnees, dict) else {}
        flux = meta.get("flux") if meta else donnees
        solde = args.solde if args.solde is not None else meta.get("solde_actuel")
        as_of = args.as_of or meta.get("as_of")
        if solde is None or as_of is None:
            raise ValueError("--solde et --as-of sont requis (ou solde_actuel / as_of dans le fichier)")
        params = Params() if args.seed is None else replace(Params(), seed=args.seed)
        t = predict(flux, solde, as_of, args.horizon or meta.get("horizon_jours", 45), params)
    except (ValueError, OSError) as err:
        print(f"erreur : {err}", file=sys.stderr)
        return 1

    if args.out:
        np.save(args.out, t, allow_pickle=False)
    else:
        print(f"{'date':10} " + " ".join(f"{c:>10}" for c in t.dtype.names[1:]))
        for ligne in t:
            print(f"{str(ligne['date']):10} " + " ".join(f"{ligne[c]:10.2f}" for c in t.dtype.names[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
