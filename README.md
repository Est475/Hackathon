# Hackathon — Tectonic / défi KBC

Prévoir le solde d'un compte à partir de ses transactions, en probabilités : quel risque d'être à
découvert, et quand.

```
exemple flux.json ──► donnees_ia.py ──► flux_agreges.json ──► predictor_standalone.py ──► tableau numpy
 (transactions)       (Gemini regroupe      (flux par            (Monte Carlo du solde)     + curseur
                       par marchand)         marchand)
```

## Installation

```bash
pip install -r requirements.txt
```

## Clé Gemini

Créer un fichier `.env` à la racine (il n'est pas envoyé sur GitHub) avec la clé du concours :

```
GEMINI_API_KEY=la_cle_du_concours
```

Autre possibilité, depuis Cloud Shell du projet Google Cloud du concours :
`export GOOGLE_CLOUD_PROJECT=<project id>` et `export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)`.

## En une commande

```bash
python synthese.py                                    # exemple flux.json, solde 1000 €, affiche le tableau
python synthese.py "exemple flux.json" --solde 1200 --curseur
python synthese.py --reutiliser --curseur             # sans rappeler l'IA (reprend flux_agreges.json)
python synthese.py --npy timeline.npy                 # enregistre le tableau numpy
```

Date de référence par défaut : la dernière transaction (`--as-of` pour la changer).

## Prédiction seule (sans IA)

```bash
python predictor_standalone.py fixtures/sarah.json                     # affiche le tableau
python predictor_standalone.py fixtures/sarah.json --curseur           # fenêtre avec curseur temporel
python predictor_standalone.py flux.json --solde 800 --as-of 2026-09-30 --out timeline.npy
```

```python
from predictor_standalone import predict, predict_scenarios
t = predict(flux, 800, "2026-09-30")          # tableau structuré, une ligne par jour
t["date"], t["esperance"], t["q05"], t["q50"], t["q95"], t["p_decouvert"]
t, B = predict_scenarios(flux, 800, "2026-09-30")   # + les 5000 soldes simulés (scénarios × jours)
```

Colonnes : `date`, `esperance` (solde moyen), `q05`…`q95` (5 % des scénarios sont sous `q05`, etc. ;
`q50` = le plus probable), `p_decouvert` (part des scénarios à découvert ce jour-là).

`--curseur` (matplotlib) ouvre une fenêtre avec le solde le plus probable et les zones « 1 chance sur 2 » /
« 9 chances sur 10 », la répartition des scénarios le jour choisi (en rouge : à découvert), le risque de
découvert jour par jour avec le seuil d'alerte de 20 %, et le tableau autour du jour choisi. Le curseur en
dessous (ou ← / →) fait défiler les jours.

### Format d'un flux

```json
{"categorie": "logement", "marchand": "Agence Namur", "montant": 950.0,
 "charge_fixe": true, "recette": false,
 "dates_apparition": ["2026-08-05", "2026-09-05"], "montants": [950.0, 950.0]}
```

`montant` = montant moyen par occurrence (le signe vient de `recette`). `montants` est optionnel.
Dates `AAAA-MM-JJ` ou `AAAA-MM-JJTHH:MM:SS`. Une entrée invalide lève `ValueError`.

### Méthode

1. **Période** : intervalle médian entre occurrences, arrondi (semaine, quinzaine, mois, trimestre, an).
2. **P(récurrent)** : a priori selon `charge_fixe`, mis à jour par chaque intervalle régulier (×3) ou
   irrégulier (÷7). Sinon : flux **variable** (courses…, taux de Poisson) ou **ponctuel**.
3. **P(habitude éteinte)** : selon le retard par rapport à la date attendue (périodique) ou la durée du
   silence (variable).
4. **Monte Carlo** (5000 scénarios, graine fixe) : chaque habitude est active ou non pour tout le
   scénario, peut s'arrêter en cours de route, montants bruités.

Démo : `fixtures/sarah.json` — Sarah a déménagé de Liège à Namur. L'ancien loyer est détecté comme
éteint, le nouveau comme nouvelle habitude, et le risque de découvert passe à 86 % le 05/10 (nouveau
loyer avant le salaire du 28).

## Tests

```bash
python -m pytest
```

## Limites

- Flux supposés indépendants entre eux (un même déménagement explique plusieurs changements).
- Dates futures déterministes (pas de gigue sur le jour de paiement).
- Paramètres fixés à la main ; en production ils s'apprendraient par catégorie sur l'historique réel.
- Les dates en double sont fusionnées : deux achats le même jour chez le même marchand comptent pour un.
