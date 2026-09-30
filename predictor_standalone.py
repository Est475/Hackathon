"""Prédiction du solde en un seul fichier.

Qualifie chaque flux (périodique / variable / ponctuel, habitude active ou éteinte), projette les
occurrences futures et simule le solde par Monte Carlo. La sortie est la timeline du solde sous forme
de tableau numpy structuré, une ligne par jour :

    date (datetime64[D]), esperance, q05, q25, q50, q75, q95, p_decouvert

Usage :
    python predictor_standalone.py fixtures/sarah.json                       # affiche le tableau
    python predictor_standalone.py fixtures/sarah.json --out timeline.npy    # l'enregistre
    python predictor_standalone.py flux.json --solde 800 --as-of 2026-09-30
    python predictor_standalone.py fixtures/sarah.json --curseur             # tableau + curseur temporel

En Python :
    from predictor_standalone import predict
    t = predict(flux, 800, "2026-09-30")
    t["date"], t["esperance"], t["p_decouvert"]
    t, B = predict_scenarios(flux, 800, "2026-09-30")   # + les 5000 soldes simulés

Dépendances : numpy, python-dateutil, pydantic v2 (+ matplotlib pour --curseur).
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
    """Date ISO « AAAA-MM-JJ » ou date-heure ISO « AAAA-MM-JJTHH:MM[:SS] » (l'heure est ignorée)."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and len(v) <= 32:
        try:
            return date.fromisoformat(v) if len(v) <= 10 else datetime.fromisoformat(v).date()
        except ValueError:
            pass
    raise ValueError("date ISO attendue (AAAA-MM-JJ ou AAAA-MM-JJTHH:MM:SS)")


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


def predict_scenarios(flux: list[dict], solde_actuel: float, as_of: date | str,
                      horizon_jours: int = 45, params: Params | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Comme predict(), mais renvoie aussi les soldes simulés B, forme (n_sim, nb_jours)."""
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
    return t, B


def predict(flux: list[dict], solde_actuel: float, as_of: date | str,
            horizon_jours: int = 45, params: Params | None = None) -> np.ndarray:
    """Timeline du solde : tableau numpy structuré (DTYPE_TIMELINE), une ligne par jour depuis as_of."""
    return predict_scenarios(flux, solde_actuel, as_of, horizon_jours, params)[0]


# =========================================================================== visualisation (--curseur)

COULEUR_SERIE = "#2a78d6"
COULEUR_RISQUE = "#d03b3b"
COULEUR_TEXTE = "#0b0b0b"
COULEUR_TEXTE_2 = "#52514e"
COULEUR_GRILLE = "#e4e3df"
COULEUR_FOND = "#fcfcfb"
COULEUR_SELECTION = "#dcebfb"
LIGNES_TABLEAU = 5
SEUIL_ALERTE = 0.20

ENTETES_TABLEAU = ["date", "solde moyen", "pessimiste\n(5 %)", "bas\n(25 %)", "le plus probable\n(médiane)",
                   "haut\n(75 %)", "optimiste\n(95 %)", "risque de\ndécouvert"]


def _euros(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def _styler_axe(ax) -> None:
    ax.set_facecolor(COULEUR_FOND)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(COULEUR_GRILLE)
    ax.tick_params(colors=COULEUR_TEXTE_2, labelsize=8)


def figure_curseur(t: np.ndarray, B: np.ndarray | None = None, titre: str = "Solde prévu"):
    """Figure matplotlib pour lire les probabilités jour par jour, avec un curseur temporel.

    - courbe du solde : valeur la plus probable + zones « 1 chance sur 2 » et « 9 chances sur 10 » ;
    - si B (scénarios simulés) est fourni : répartition des scénarios le jour choisi, partie à découvert en rouge ;
    - risque de découvert jour par jour (%), avec le seuil d'alerte ;
    - phrase de synthèse + tableau numpy autour du jour choisi ;
    - curseur en dessous (ou ← / →). Renvoie (fig, slider).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n = len(t)
    x = np.arange(n)
    dates = [str(d) for d in t["date"]]
    jj_mm = [f"{d[8:10]}/{d[5:7]}" for d in dates]
    risque = 100 * t["p_decouvert"]

    fig = plt.figure(figsize=(12, 8.8), facecolor=COULEUR_FOND)
    fig.suptitle(titre, x=0.06, ha="left", y=0.975, color=COULEUR_TEXTE, fontsize=13, fontweight="bold")
    large = 0.60 if B is not None else 0.89
    ax = fig.add_axes([0.07, 0.61, large, 0.32])
    ax_r = fig.add_axes([0.07, 0.43, large, 0.13], sharex=ax)
    ax_h = fig.add_axes([0.72, 0.61, 0.25, 0.32], sharey=ax) if B is not None else None
    ax_txt = fig.add_axes([0.03, 0.315, 0.94, 0.06])
    ax_tab = fig.add_axes([0.03, 0.10, 0.94, 0.21])
    ax_cur = fig.add_axes([0.14, 0.045, 0.70, 0.03], facecolor=COULEUR_GRILLE)
    for a in (ax_txt, ax_tab):
        a.axis("off")

    # --- 1. courbe du solde
    _styler_axe(ax)
    ax.fill_between(x, t["q05"], t["q95"], color=COULEUR_SERIE, alpha=0.13, lw=0,
                    label="9 chances sur 10 que le solde soit dans cette zone")
    ax.fill_between(x, t["q25"], t["q75"], color=COULEUR_SERIE, alpha=0.30, lw=0,
                    label="1 chance sur 2 que le solde soit dans cette zone")
    ax.plot(x, t["q50"], color=COULEUR_SERIE, lw=2, label="solde le plus probable")
    ax.axhline(0, color=COULEUR_RISQUE, lw=1, ls="--", label="0 € : en dessous, découvert")
    ax.set_title("Solde du compte (€)", loc="left", color=COULEUR_TEXTE, fontsize=10)
    ax.grid(axis="y", color=COULEUR_GRILLE, lw=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=COULEUR_TEXTE_2)
    ax.tick_params(labelbottom=False)
    repere = ax.axvline(0, color=COULEUR_TEXTE, lw=1)
    point, = ax.plot([0], [t["q50"][0]], "o", ms=8, color=COULEUR_SERIE, mec=COULEUR_FOND, mew=2, zorder=5)

    # --- 2. risque de découvert jour par jour
    _styler_axe(ax_r)
    couleurs_r = [COULEUR_RISQUE if r >= 100 * SEUIL_ALERTE else "#b9b8b2" for r in risque]
    ax_r.bar(x, risque, width=0.8, color=couleurs_r)
    ax_r.axhline(100 * SEUIL_ALERTE, color=COULEUR_TEXTE_2, lw=1, ls=":")
    ax_r.text(n - 0.5, 100 * SEUIL_ALERTE, f"seuil d'alerte {100 * SEUIL_ALERTE:.0f} %", ha="right",
              va="bottom", fontsize=7, color=COULEUR_TEXTE_2)
    ax_r.set_ylim(0, 100)
    ax_r.set_yticks([0, 50, 100], ["0 %", "50 %", "100 %"])
    ax_r.set_title("Risque d'être à découvert ce jour-là", loc="left", color=COULEUR_TEXTE, fontsize=10)
    ax_r.grid(axis="y", color=COULEUR_GRILLE, lw=0.8)
    pas_x = max(1, n // 9)
    ax_r.set_xticks(x[::pas_x], jj_mm[::pas_x])
    ax_r.set_xlim(-0.5, n - 0.5)
    repere_r = ax_r.axvline(0, color=COULEUR_TEXTE, lw=1)

    # --- 3. répartition des scénarios le jour choisi
    barres = centres = None
    if ax_h is not None:
        _styler_axe(ax_h)
        lo = float(np.quantile(B, 0.01, axis=0).min())
        hi = float(np.quantile(B, 0.99, axis=0).max())
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        bornes = np.linspace(lo, hi, 41)
        centres = (bornes[:-1] + bornes[1:]) / 2
        couleurs_h = [COULEUR_RISQUE if c < 0 else COULEUR_SERIE for c in centres]
        barres = ax_h.barh(centres, np.zeros_like(centres), height=(bornes[1] - bornes[0]) * 0.85,
                           color=couleurs_h)
        ax_h.axhline(0, color=COULEUR_RISQUE, lw=1, ls="--")
        ax_h.set_xlabel("% des scénarios", color=COULEUR_TEXTE_2, fontsize=8)
        ax_h.tick_params(labelleft=False)
        ax_h.grid(axis="x", color=COULEUR_GRILLE, lw=0.8)
        titre_h = ax_h.set_title("", loc="left", color=COULEUR_TEXTE, fontsize=10)
        txt_pos = ax_h.text(0.97, 0.97, "", transform=ax_h.transAxes, ha="right", va="top",
                            color=COULEUR_SERIE, fontsize=9, fontweight="bold")
        txt_neg = ax_h.text(0.97, 0.03, "", transform=ax_h.transAxes, ha="right", va="bottom",
                            color=COULEUR_RISQUE, fontsize=9, fontweight="bold")

    # --- 4. synthèse en clair
    txt_risque = ax_txt.text(0.0, 0.62, "", fontsize=13, fontweight="bold", va="center")
    txt_detail = ax_txt.text(0.0, 0.05, "", fontsize=10, color=COULEUR_TEXTE_2, va="center")

    def dessiner_tableau(j: int) -> None:
        ax_tab.clear()
        ax_tab.axis("off")
        debut = min(max(j - LIGNES_TABLEAU // 2, 0), max(n - LIGNES_TABLEAU, 0))
        rows = [[jj_mm[debut + i]] + [_euros(r[c]) for c in ("esperance", "q05", "q25", "q50", "q75", "q95")]
                + [f"{100 * r['p_decouvert']:.0f} %"] for i, r in enumerate(t[debut:debut + LIGNES_TABLEAU])]
        tab = ax_tab.table(cellText=rows, colLabels=ENTETES_TABLEAU, loc="upper center", cellLoc="right")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.6)
        for (i, c), cell in tab.get_celld().items():
            cell.set_edgecolor(COULEUR_GRILLE)
            cell.set_facecolor(COULEUR_FOND)
            txt = cell.get_text()
            txt.set_color(COULEUR_TEXTE)
            if i == 0:
                cell.set_height(cell.get_height() * 1.6)
                txt.set_color(COULEUR_TEXTE_2)
                txt.set_fontweight("bold")
                txt.set_fontsize(8)
            else:
                if i - 1 == j - debut:
                    cell.set_facecolor(COULEUR_SELECTION)
                    txt.set_fontweight("bold")
                if c == len(ENTETES_TABLEAU) - 1 and rows[i - 1][c] != "0 %" \
                        and t["p_decouvert"][debut + i - 1] >= SEUIL_ALERTE:
                    txt.set_color(COULEUR_RISQUE)

    slider = Slider(ax_cur, "jour", 0, n - 1, valinit=0, valstep=1, color=COULEUR_SERIE)
    slider.label.set_color(COULEUR_TEXTE_2)
    slider.valtext.set_color(COULEUR_TEXTE)

    def maj(_val=None) -> None:
        j = int(slider.val)
        r = t[j]
        for rep in (repere, repere_r):
            rep.set_xdata([j, j])
        point.set_data([j], [r["q50"]])
        slider.valtext.set_text(dates[j])

        p = r["p_decouvert"]
        sur_100 = int(round(100 * p))
        alerte = p >= SEUIL_ALERTE
        txt_risque.set_text(f"Le {jj_mm[j]} : {sur_100} % de risque d'être à découvert"
                            f"  ({sur_100} scénarios sur 100){'  ⚠ alerte' if alerte else ''}")
        txt_risque.set_color(COULEUR_RISQUE if alerte else COULEUR_TEXTE)
        txt_detail.set_text(f"Solde le plus probable : {_euros(r['q50'])}  ·  "
                            f"1 chance sur 2 entre {_euros(r['q25'])} et {_euros(r['q75'])}  ·  "
                            f"9 chances sur 10 entre {_euros(r['q05'])} et {_euros(r['q95'])}")

        if barres is not None:
            counts, _ = np.histogram(np.clip(B[:, j], bornes[0], bornes[-1]), bins=bornes)
            pct = 100 * counts / B.shape[0]
            for b, w in zip(barres, pct):
                b.set_width(w)
            ax_h.set_xlim(0, max(pct.max() * 1.15, 1))
            titre_h.set_text(f"Les {B.shape[0]} scénarios le {jj_mm[j]}")
            txt_pos.set_text(f"{100 * (1 - p):.0f} % au-dessus de 0 €")
            txt_neg.set_text(f"{100 * p:.0f} % à découvert")

        dessiner_tableau(j)
        fig.canvas.draw_idle()

    def clavier(ev) -> None:
        if ev.key in ("right", "left"):
            slider.set_val(min(max(int(slider.val) + (1 if ev.key == "right" else -1), 0), n - 1))

    slider.on_changed(maj)
    fig.canvas.mpl_connect("key_press_event", clavier)
    maj()
    return fig, slider


def afficher_curseur(t: np.ndarray, B: np.ndarray | None = None, titre: str = "Solde prévu") -> None:
    import matplotlib.pyplot as plt

    fig, slider = figure_curseur(t, B, titre)
    fig._slider = slider  # garde une référence, sinon le curseur ne répond plus
    plt.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Timeline du solde prévu (tableau numpy).")
    ap.add_argument("entree", help="JSON : liste de flux, ou objet {flux, solde_actuel?, as_of?}")
    ap.add_argument("--solde", type=float, help="solde actuel (sinon lu dans le fichier)")
    ap.add_argument("--as-of", dest="as_of", help="date de référence AAAA-MM-JJ (sinon lue dans le fichier)")
    ap.add_argument("--horizon", type=int, help="horizon en jours (défaut 45)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="enregistre le tableau en .npy (sinon l'affiche)")
    ap.add_argument("--curseur", action="store_true",
                    help="ouvre une fenêtre : courbe + tableau + curseur temporel (matplotlib)")
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
        t, B = predict_scenarios(flux, solde, as_of, args.horizon or meta.get("horizon_jours", 45), params)
    except (ValueError, OSError) as err:
        print(f"erreur : {err}", file=sys.stderr)
        return 1

    if args.out:
        np.save(args.out, t, allow_pickle=False)
    if args.curseur:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("erreur : --curseur nécessite matplotlib (pip install matplotlib)", file=sys.stderr)
            return 1
        afficher_curseur(t, B, f"Solde prévu — {os.path.basename(args.entree)}")
    elif not args.out:
        print(f"{'date':10} " + " ".join(f"{c:>10}" for c in t.dtype.names[1:]))
        for ligne in t:
            print(f"{str(ligne['date']):10} " + " ".join(f"{ligne[c]:10.2f}" for c in t.dtype.names[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
