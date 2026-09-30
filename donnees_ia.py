"""Regroupe les transactions brutes par marchand à l'aide de Gemini (module amont du predictor).

Entrée  : JSON de transactions [{"date_heure": ..., "marchand": ..., "montant": ...}, ...]
Sortie  : JSON de flux agrégés, au format attendu par `predictor` / `predictor_standalone.py`.

Usage :
    python donnees_ia.py "exemple flux.json" --out flux_agreges.json
    python predictor_standalone.py flux_agreges.json --solde 1000 --as-of 2026-09-30 --curseur

Authentification (jamais de clé dans le code) — dans cet ordre :
  1. Vertex AI sur un projet Google Cloud (ex. le projet Qwiklabs du hackathon, depuis Cloud Shell) :
         export GOOGLE_CLOUD_PROJECT=<project id>
         export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)
     (région : GOOGLE_CLOUD_LOCATION, défaut us-central1)
  2. Clé API (Google AI Studio ou Vertex AI express) :
         export GEMINI_API_KEY=<clé>
     ou une ligne GEMINI_API_KEY=... dans un fichier .env (ignoré par git).
Modèle : GEMINI_MODEL (défaut gemini-3.5-flash).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

MODELE_DEFAUT = "gemini-3.5-flash"
TIMEOUT = 120
MAX_TAILLE_FICHIER = 5 * 1024 * 1024

PROMPT = """Tu es un algorithme de traitement financier.

Voici une liste de transactions brutes (JSON) :
{transactions}

RÈGLE ABSOLUE DE REGROUPEMENT (DÉDUPLICATION) :
Si plusieurs transactions concernent le MÊME marchand et un montant régulier (ex: un Loyer ou Netflix payé
plusieurs fois), tu NE DOIS GÉNÉRER QU'UN SEUL objet JSON pour ce marchand.
Tu dois fusionner ces dépenses en ajoutant toutes leurs dates dans la liste "dates_apparition".
Les achats du même type chez des enseignes différentes (ex: supermarchés) peuvent être regroupés.

--- DÉBUT DE L'EXEMPLE DE LOGIQUE ---
INTERDIT (Ne jamais faire ça) :
[
  {{"marchand": "Propriétaire", "montant": -850, "dates_apparition": ["2026-08-01"]}},
  {{"marchand": "Propriétaire", "montant": -850, "dates_apparition": ["2026-09-01"]}}
]

OBLIGATOIRE (Ce que tu dois faire) :
[
  {{"marchand": "Propriétaire", "montant": -850, "dates_apparition": ["2026-08-01", "2026-09-01"]}}
]
--- FIN DE L'EXEMPLE ---

CHAMPS :
- "montant" : montant MOYEN par occurrence (négatif pour une dépense, positif pour une recette).
- "montants" : la liste des montants de chaque occurrence, dans le même ordre que "dates_apparition".
- "charge_fixe" : true pour une charge contractuelle (loyer, abonnement, salaire…), false sinon.
- "recette" : true si c'est une rentrée d'argent.
- dates au format ISO (AAAA-MM-JJ ou AAAA-MM-JJTHH:MM:SS).

FORMAT DE SORTIE OBLIGATOIRE :
Tu dois répondre UNIQUEMENT par un tableau JSON valide. Ne génère aucun texte avant ou après.
[
  {{
    "categorie": "...",
    "marchand": "...",
    "montant": ...,
    "charge_fixe": true,
    "recette": false,
    "premiere_apparition": "...",
    "derniere_apparition": "...",
    "dates_apparition": ["...", "..."],
    "montants": [..., ...]
  }}
]
"""


class ErreurIA(Exception):
    pass


def _lire_env_fichier(chemin: str = ".env") -> dict[str, str]:
    """Lecture minimale d'un .env (CLE=valeur), sans dépendance externe."""
    valeurs: dict[str, str] = {}
    if os.path.isfile(chemin):
        with open(chemin, encoding="utf-8") as fh:
            for ligne in fh:
                ligne = ligne.strip()
                if ligne and not ligne.startswith("#") and "=" in ligne:
                    cle, val = ligne.split("=", 1)
                    valeurs[cle.strip()] = val.strip().strip('"').strip("'")
    return valeurs


def _config(nom: str, defaut: str | None = None) -> str | None:
    return os.environ.get(nom) or _lire_env_fichier().get(nom) or defaut


def _points_d_acces(modele: str) -> list[tuple[str, dict[str, str]]]:
    """(url, en-têtes) à essayer, dans l'ordre. La clé passe en en-tête, jamais dans l'URL."""
    projet, jeton = _config("GOOGLE_CLOUD_PROJECT"), _config("GOOGLE_ACCESS_TOKEN")
    if projet and jeton:
        region = _config("GOOGLE_CLOUD_LOCATION", "us-central1")
        hote = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        url = (f"https://{hote}/v1/projects/{projet}/locations/{region}"
               f"/publishers/google/models/{modele}:generateContent")
        return [(url, {"Authorization": f"Bearer {jeton}"})]
    cle = _config("GEMINI_API_KEY") or _config("GOOGLE_API_KEY")
    if not cle:
        raise ErreurIA("aucune authentification : définir GEMINI_API_KEY (ou GOOGLE_CLOUD_PROJECT + "
                       "GOOGLE_ACCESS_TOKEN), voir l'en-tête de donnees_ia.py")
    entetes = {"x-goog-api-key": cle}
    return [
        # Google AI Studio
        (f"https://generativelanguage.googleapis.com/v1beta/models/{modele}:generateContent", entetes),
        # Vertex AI express (clé de type « AQ. » émise par la console Google Cloud)
        (f"https://aiplatform.googleapis.com/v1/publishers/google/models/{modele}:generateContent", entetes),
    ]


def extraire_json(texte: str) -> list:
    """Récupère le tableau JSON de la réponse (tolère un bloc ```json … ```)."""
    t = texte.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    debut, fin = t.find("["), t.rfind("]")
    if debut == -1 or fin == -1:
        raise ErreurIA("la réponse de l'IA ne contient pas de tableau JSON")
    try:
        donnees = json.loads(t[debut:fin + 1])
    except json.JSONDecodeError as e:
        raise ErreurIA(f"JSON invalide dans la réponse de l'IA ({e.msg})") from None
    if not isinstance(donnees, list):
        raise ErreurIA("la réponse de l'IA n'est pas une liste")
    return donnees


def appeler_ia(prompt: str, modele: str | None = None) -> str:
    modele = modele or _config("GEMINI_MODEL", MODELE_DEFAUT)
    corps = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    derniere_erreur = ""
    for url, entetes in _points_d_acces(modele):
        try:
            rep = requests.post(url, json=corps, headers={"Content-Type": "application/json", **entetes},
                                timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise ErreurIA(f"connexion impossible : {type(e).__name__}") from None
        if rep.status_code in (401, 403):     # clé refusée ici : on essaie le point d'accès suivant
            derniere_erreur = f"accès refusé (HTTP {rep.status_code}) — clé ou jeton invalide/expiré ?"
            continue
        if rep.status_code == 404:
            raise ErreurIA(f"modèle « {modele} » introuvable (HTTP 404) — changer GEMINI_MODEL")
        if not rep.ok:
            raise ErreurIA(f"erreur de l'API (HTTP {rep.status_code}) : {rep.text[:300]}")
        try:
            return rep.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError):
            raise ErreurIA("réponse inattendue de l'API (pas de texte généré)") from None
    raise ErreurIA(derniere_erreur)


def agreger(transactions: list[dict], modele: str | None = None) -> list[dict]:
    prompt = PROMPT.format(transactions=json.dumps(transactions, ensure_ascii=False))
    return extraire_json(appeler_ia(prompt, modele))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regroupe les transactions par marchand (Gemini).")
    ap.add_argument("entree", nargs="?", default="exemple flux.json", help="transactions brutes (JSON)")
    ap.add_argument("--out", default="flux_agreges.json", help="fichier de sortie (défaut flux_agreges.json)")
    ap.add_argument("--modele", help=f"modèle Gemini (défaut : GEMINI_MODEL ou {MODELE_DEFAUT})")
    args = ap.parse_args(argv)
    try:
        if os.path.getsize(args.entree) > MAX_TAILLE_FICHIER:
            raise ErreurIA("fichier d'entrée trop volumineux")
        with open(args.entree, encoding="utf-8") as fh:
            transactions = json.load(fh)
        flux = agreger(transactions, args.modele)
        # Vérifie que la sortie est bien exploitable par le module de prédiction
        from predictor_standalone import valider
        valider(flux, 0.0, "2100-01-01", 45)
    except (ErreurIA, ValueError, OSError) as e:
        print(f"erreur : {e}", file=sys.stderr)
        return 1
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(flux, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"{len(transactions)} transactions → {len(flux)} flux, écrits dans {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
