"""Paramètres du modèle de prédiction (section 3 de la spec)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Params:
    prior_fixe: float = 0.8          # P(récurrent) a priori si charge_fixe
    prior_libre: float = 0.2         # P(récurrent) a priori sinon
    rho1: float = 0.9                # P(intervalle régulier | périodique)
    rho0: float = 0.3                # P(intervalle régulier | non périodique)
    tol_rel: float = 0.15            # tolérance de régularité, relative à la période
    h: float = 0.05                  # P(qu'une habitude s'arrête) par période
    fenetre_recente: int = 60        # jours pour qualifier un changement de récent
    cv_fixe: float = 0.02            # coef. de variation du montant, charge fixe
    cv_periodique: float = 0.15      # idem, périodique non fixe
    cv_variable: float = 0.30        # idem, flux variable
    n_sim: int = 5000                # scénarios Monte Carlo
    seed: int = 42                   # graine (reproductibilité)
    n_bins: int = 40                 # tranches de la grille de densité
    seuil_alerte: float = 0.20       # P(découvert) déclenchant date_alerte

    def __post_init__(self) -> None:
        for nom in ("prior_fixe", "prior_libre", "rho1", "rho0", "h"):
            v = getattr(self, nom)
            if not 0.0 < v < 1.0:
                raise ValueError(f"Paramètre {nom} doit être dans ]0, 1[ (reçu {v})")
        for nom in ("tol_rel", "cv_fixe", "cv_periodique", "cv_variable"):
            if getattr(self, nom) < 0:
                raise ValueError(f"Paramètre {nom} doit être positif")
        if not 0.0 <= self.seuil_alerte <= 1.0:
            raise ValueError("Paramètre seuil_alerte doit être dans [0, 1]")
        if not 1 <= self.n_sim <= 100_000:
            raise ValueError("Paramètre n_sim doit être dans [1, 100000]")
        if not 1 <= self.n_bins <= 500:
            raise ValueError("Paramètre n_bins doit être dans [1, 500]")
        if not 0 <= self.fenetre_recente <= 3650:
            raise ValueError("Paramètre fenetre_recente doit être dans [0, 3650]")

    def to_dict(self) -> dict:
        return asdict(self)
