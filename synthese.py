"""Chaîne complète en une commande : transactions brutes → regroupement IA → prédiction du solde.

    python synthese.py                                   # exemple flux.json, solde 1000 €, tableau
    python synthese.py "exemple flux.json" --solde 1200 --curseur
    python synthese.py --reutiliser --curseur            # sans rappeler l'IA (reprend flux_agreges.json)

Étapes :
  1. lit les transactions (date_heure, marchand, montant) ;
  2. donnees_ia.py : Gemini regroupe par marchand → flux_agreges.json (clé : voir donnees_ia.py) ;
  3. predictor_standalone.py : simulation → tableau numpy (affiché, --npy, ou fenêtre --curseur).
Date de référence par défaut : la date de la dernière transaction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

import donnees_ia
import predictor_standalone as ps

SOLDE_DEFAUT = 1000.0


def _derniere_date(transactions: list) -> str:
    dates = [str(t.get("date_heure", ""))[:10] for t in transactions if isinstance(t, dict)]
    dates = [d for d in dates if len(d) == 10]
    if not dates:
        raise ValueError("aucune date_heure exploitable dans les transactions ; préciser --as-of")
    return max(dates)


def _afficher(t: np.ndarray) -> None:
    print(f"{'date':10} " + " ".join(f"{c:>10}" for c in t.dtype.names[1:]))
    for ligne in t:
        print(f"{str(ligne['date']):10} " + " ".join(f"{ligne[c]:10.2f}" for c in t.dtype.names[1:]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transactions → IA (regroupement) → prédiction du solde.")
    ap.add_argument("transactions", nargs="?", default="exemple flux.json", help="transactions brutes (JSON)")
    ap.add_argument("--solde", type=float, default=SOLDE_DEFAUT, help=f"solde actuel (défaut {SOLDE_DEFAUT:g})")
    ap.add_argument("--as-of", dest="as_of", help="date de référence (défaut : dernière transaction)")
    ap.add_argument("--horizon", type=int, default=45, help="horizon en jours (défaut 45)")
    ap.add_argument("--flux", default="flux_agreges.json", help="fichier intermédiaire des flux agrégés")
    ap.add_argument("--reutiliser", action="store_true", help="ne pas rappeler l'IA, reprendre --flux")
    ap.add_argument("--modele", help="modèle Gemini")
    ap.add_argument("--npy", help="enregistre le tableau en .npy")
    ap.add_argument("--curseur", action="store_true", help="ouvre la fenêtre tableau + curseur temporel")
    args = ap.parse_args(argv)

    try:
        with open(args.transactions, encoding="utf-8") as fh:
            transactions = json.load(fh)
        as_of = args.as_of or _derniere_date(transactions)

        if args.reutiliser:
            with open(args.flux, encoding="utf-8") as fh:
                flux = json.load(fh)
            print(f"[1/2] flux repris de {args.flux} ({len(flux)} flux)", file=sys.stderr)
        else:
            print(f"[1/2] regroupement de {len(transactions)} transactions par l'IA…", file=sys.stderr)
            flux = donnees_ia.agreger(transactions, args.modele)
            ps.valider(flux, 0.0, "2100-01-01", args.horizon)
            with open(args.flux, "w", encoding="utf-8") as fh:
                json.dump(flux, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            print(f"      → {len(flux)} flux, enregistrés dans {args.flux}", file=sys.stderr)

        print(f"[2/2] prédiction au {as_of}, solde {args.solde:g} €…", file=sys.stderr)
        t, B = ps.predict_scenarios(flux, args.solde, as_of, args.horizon)
    except (donnees_ia.ErreurIA, ValueError, OSError) as e:
        print(f"erreur : {e}", file=sys.stderr)
        return 1

    if args.npy:
        np.save(args.npy, t, allow_pickle=False)
        print(f"      → tableau enregistré dans {args.npy}", file=sys.stderr)
    alerte = next((str(r["date"]) for r in t if r["p_decouvert"] >= ps.SEUIL_ALERTE), None)
    suite = f"alerte à partir du {alerte}" if alerte else "pas d'alerte"
    print(f"      → risque de découvert max {100 * t['p_decouvert'].max():.0f} %, {suite}", file=sys.stderr)

    if args.curseur:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("erreur : --curseur nécessite matplotlib (pip install matplotlib)", file=sys.stderr)
            return 1
        ps.afficher_curseur(t, B, f"Solde prévu — {os.path.basename(args.transactions)}")
    elif not args.npy:
        _afficher(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
