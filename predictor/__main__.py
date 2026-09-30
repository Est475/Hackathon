"""CLI : python -m predictor fixtures/sarah.json --solde 800 --as-of 2026-09-30 --out output.json"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np

from . import Params, predict, timeline_numpy

MAX_TAILLE_FICHIER = 20 * 1024 * 1024


def _charger(chemin: str) -> object:
    if os.path.getsize(chemin) > MAX_TAILLE_FICHIER:
        raise ValueError("fichier d'entrée trop volumineux")
    with open(chemin, encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON invalide (ligne {e.lineno}, colonne {e.colno})") from None


def _afficher(sortie: dict) -> None:
    err = sys.stderr
    print(f"{'id':5} {'marchand':24} {'type':11} {'P(rec)':>6} {'actif':>6} {'éteint':>6}  changement", file=err)
    for f in sortie["flux"]:
        s = f["statut"]
        print(f"{f['flux_id']:5} {f['marchand'][:24]:24} {f['type']:11} {f['p_recurrent']:6.2f} "
              f"{s['actif']:6.2f} {s['eteint']:6.2f}  {f['changement']['principal']}"
              f"{' ⚠' if f['changement']['est_perturbateur'] else ''}", file=err)
    r = sortie["resume"]
    fm = r["fin_de_mois"]
    print(f"\nFin de mois {fm['date']} : espérance {fm['esperance']:.0f} € "
          f"[q05 {fm['q05']:.0f} ; q95 {fm['q95']:.0f}], P(découvert) = {fm['p_decouvert']:.2f}", file=err)
    if r["remuneration"]:
        print(f"Rémunération : {r['remuneration']['marchand']} le {r['remuneration']['date']}", file=err)
    print(f"Date d'alerte : {r['date_alerte'] or 'aucune'}", file=err)
    for w in sortie["warnings"]:
        print(f"warning: {w}", file=err)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m predictor", description=__doc__)
    ap.add_argument("entree", help="JSON : liste de flux, ou objet {flux, solde_actuel?, as_of?}")
    ap.add_argument("--solde", type=float, help="solde actuel (sinon lu dans le fichier)")
    ap.add_argument("--as-of", dest="as_of", help="date de référence AAAA-MM-JJ (sinon lue dans le fichier)")
    ap.add_argument("--horizon", type=int, default=None, help="horizon en jours (défaut 45)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-sim", dest="n_sim", type=int, default=None)
    ap.add_argument("--out", help="fichier de sortie (défaut : stdout)")
    ap.add_argument("--npy", help="enregistre aussi la timeline en tableau numpy (.npy)")
    ap.add_argument("-q", "--quiet", action="store_true", help="ne pas afficher le résumé sur stderr")
    args = ap.parse_args(argv)

    try:
        donnees = _charger(args.entree)
        if isinstance(donnees, dict):
            flux = donnees.get("flux")
            solde = args.solde if args.solde is not None else donnees.get("solde_actuel")
            as_of = args.as_of or donnees.get("as_of")
            horizon = args.horizon or donnees.get("horizon_jours", 45)
        else:
            flux, solde, as_of, horizon = donnees, args.solde, args.as_of, args.horizon or 45
        if solde is None or as_of is None:
            raise ValueError("--solde et --as-of sont requis (ou solde_actuel / as_of dans le fichier)")
        params = Params()
        if args.seed is not None:
            params = replace(params, seed=args.seed)
        if args.n_sim is not None:
            params = replace(params, n_sim=args.n_sim)
        sortie = predict(flux, solde, as_of, horizon, params)
    except (ValueError, OSError) as e:
        print(f"erreur : {e}", file=sys.stderr)
        return 1

    texte = json.dumps(sortie, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(texte + "\n")
    else:
        print(texte)
    if args.npy:
        np.save(args.npy, timeline_numpy(sortie), allow_pickle=False)
    if not args.quiet:
        _afficher(sortie)
    return 0


if __name__ == "__main__":
    sys.exit(main())
