"""Étape F : simulation Monte Carlo du solde → timeline et grille de densité."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .flux import AnalyseFlux
from .params import Params
from .projection import SEUIL_ACTIF, dates_projetees


def simuler_soldes(analyses: list[AnalyseFlux], solde: float, as_of: date, D: int,
                   params: Params) -> np.ndarray:
    """Renvoie B de forme (N, D+1) ; colonne 0 = as_of (solde certain)."""
    rng = np.random.default_rng(params.seed)
    N = params.n_sim
    cash = np.zeros((N, D))
    fin = as_of + timedelta(days=D)
    for fa in analyses:
        if fa.actif <= SEUIL_ACTIF:
            continue
        if fa.type == "periodique":
            dates = dates_projetees(fa, as_of, fin)
            if not dates:
                continue
            active = rng.random(N) < fa.actif
            # K = nombre de périodes avant l'arrêt de l'habitude : P(K > k) = (1-h)^k
            K = rng.geometric(params.h, N)
            for k, d in enumerate(dates):
                i = (d - as_of).days - 1
                present = active & (k < K)
                cash[:, i] += present * (fa.a + fa.sigma_a * rng.standard_normal(N))
        elif fa.type == "variable" and fa.lam is not None:
            active = rng.random(N) < fa.actif
            c = rng.poisson(fa.lam, (N, D))
            Z = rng.standard_normal((N, D))
            cash += active[:, None] * (c * fa.a + np.sqrt(c) * fa.sigma_a * Z)
    B = np.empty((N, D + 1))
    B[:, 0] = solde
    B[:, 1:] = solde + np.cumsum(cash, axis=1)
    return B


def construire_timeline(B: np.ndarray, as_of: date, evenements: list[dict]) -> list[dict]:
    par_date: dict[str, list[str]] = {}
    for e in evenements:
        if e["nature"] == "discret":
            par_date.setdefault(e["date"], []).append(e["id"])
    q = np.quantile(B, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    moy = B.mean(axis=0)
    p_dec = (B < 0).mean(axis=0)
    timeline = []
    for j in range(B.shape[1]):
        d = (as_of + timedelta(days=j)).isoformat()
        timeline.append({
            "date": d,
            "esperance": round(float(moy[j]), 2),
            "q05": round(float(q[0, j]), 2),
            "q25": round(float(q[1, j]), 2),
            "q50": round(float(q[2, j]), 2),
            "q75": round(float(q[3, j]), 2),
            "q95": round(float(q[4, j]), 2),
            "p_decouvert": round(float(p_dec[j]), 4),
            "evenements": par_date.get(d, []),
        })
    return timeline


def construire_grille(B: np.ndarray, as_of: date, n_bins: int) -> dict:
    q01 = np.quantile(B, 0.01, axis=0)
    q99 = np.quantile(B, 0.99, axis=0)
    lo, hi = float(q01.min()), float(q99.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    bornes = np.linspace(lo, hi, n_bins + 1)
    Bc = np.clip(B, lo, hi)
    N = B.shape[0]
    probas = []
    for j in range(B.shape[1]):
        counts, _ = np.histogram(Bc[:, j], bins=bornes)
        probas.append((counts / N).tolist())
    centres = (bornes[:-1] + bornes[1:]) / 2
    return {
        "dates": [(as_of + timedelta(days=j)).isoformat() for j in range(B.shape[1])],
        "montants": [round(float(c), 2) for c in centres],
        "probabilites": probas,
    }
