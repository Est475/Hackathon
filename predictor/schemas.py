"""Modèles pydantic d'entrée (validation stricte) et de sortie."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, field_validator

MAX_FLUX = 2000
MAX_DATES = 1000
MAX_MONTANT = 1e7
MAX_SOLDE = 1e9


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


# --------------------------------------------------------------------------- entrée


class FluxEntree(BaseModel):
    model_config = ConfigDict(extra="ignore", str_max_length=200)

    categorie: str = ""
    marchand: str = ""
    montant: float
    charge_fixe: bool = False
    recette: bool = False
    premiere_apparition: DateISO | None = None
    derniere_apparition: DateISO | None = None
    dates_apparition: list[DateISO] = Field(min_length=1, max_length=MAX_DATES)
    montants: list[float] | None = Field(default=None, max_length=MAX_DATES)

    @field_validator("montant")
    @classmethod
    def _montant(cls, v: float) -> float:
        return _verifier_montant(v)

    @field_validator("montants")
    @classmethod
    def _montants(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            for x in v:
                _verifier_montant(x)
        return v


class EntreePrediction(BaseModel):
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


def valider_entree(flux: object, solde_actuel: object, as_of: object, horizon_jours: object) -> EntreePrediction:
    """Valide l'entrée ; lève ValueError avec un message lisible (sans valeurs brutes)."""
    if isinstance(flux, list) and len(flux) > MAX_FLUX:
        raise ValueError(f"Entrée invalide : plus de {MAX_FLUX} flux")
    try:
        return EntreePrediction(flux=flux, solde_actuel=solde_actuel, as_of=as_of, horizon_jours=horizon_jours)
    except ValidationError as e:
        details = []
        for err in e.errors()[:10]:
            loc = ".".join(str(p) for p in err["loc"])
            msg = err["msg"].removeprefix("Value error, ")
            details.append(f"{loc}: {msg}")
        raise ValueError("Entrée invalide : " + " ; ".join(details)) from None


# --------------------------------------------------------------------------- sortie


class Evidence(BaseModel):
    code: Literal["OCCURRENCES", "PERIODE", "REGULARITE", "PRIOR", "P_RECURRENT",
                  "RETARD", "TAUX", "P_ETEINT", "RECENT"]
    message: str
    valeur: float | int | str | None = None


class Periode(BaseModel):
    pas: str
    jours: float


class Changement(BaseModel):
    distribution: dict[str, float]
    principal: str
    est_perturbateur: bool


class FluxSortie(BaseModel):
    flux_id: str
    marchand: str
    categorie: str
    recette: bool
    montant: float
    sigma_montant: float
    type: Literal["periodique", "variable", "ponctuel"]
    periode: Periode | None
    taux_journalier: float | None
    p_recurrent: float
    statut: dict[str, float]
    changement: Changement
    date_attendue: str | None
    prochaine_date: str | None
    evidences: list[Evidence]


class EvenementDiscret(BaseModel):
    id: str
    flux_id: str
    nature: Literal["discret"]
    date: str
    marchand: str
    categorie: str
    montant: float
    sigma_montant: float
    probabilite: float
    esperance: float


class EvenementContinu(BaseModel):
    id: str
    flux_id: str
    nature: Literal["continu"]
    date_debut: str
    date_fin: str
    marchand: str
    categorie: str
    montant: float
    sigma_montant: float
    probabilite: float
    esperance: float


Evenement = Annotated[Union[EvenementDiscret, EvenementContinu], Field(discriminator="nature")]


class JourTimeline(BaseModel):
    date: str
    esperance: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    p_decouvert: float
    evenements: list[str]


class GrilleDensite(BaseModel):
    dates: list[str]
    montants: list[float]
    probabilites: list[list[float]]


class Remuneration(BaseModel):
    flux_id: str
    marchand: str
    date: str
    montant: float


class PointCle(BaseModel):
    date: str
    esperance: float
    q05: float
    q95: float
    p_decouvert: float


class PDecouvertMax(BaseModel):
    valeur: float
    date: str


class Perturbateur(BaseModel):
    flux_id: str
    marchand: str
    categorie: str
    changement: str
    probabilite: float
    impact_mensuel: float | None = None
    impact_ponctuel: float | None = None


class Resume(BaseModel):
    remuneration: Remuneration | None
    fin_de_mois: PointCle
    veille_remuneration: PointCle | None
    p_decouvert_max: PDecouvertMax
    date_alerte: str | None
    perturbateurs: list[Perturbateur]


class SortiePrediction(BaseModel):
    as_of: str
    solde_actuel: float
    horizon_jours: int
    horizon_effectif_jours: int
    parametres: dict
    flux: list[FluxSortie]
    evenements_prevus: list[Evenement]
    timeline: list[JourTimeline]
    grille_densite: GrilleDensite
    resume: Resume
    warnings: list[str]
