# Hackathon — Tectonic / défi KBC

## Module de prédiction (`predictor/`)

À partir des flux agrégés par marchand (produits par l'amont), du solde actuel et d'une date de
référence, le module :

1. qualifie chaque flux de façon probabiliste (ponctuel / habitude active / habitude éteinte) et
   détecte le type de changement (`nouvelle_habitude`, `habitude_eteinte`, `evenement_ponctuel`…),
   avec des **évidences lisibles** en français ;
2. projette les **événements futurs** (date × montant × probabilité) ;
3. simule le solde par **Monte Carlo** (5000 scénarios) → espérance, quantiles et P(découvert) jour
   par jour, plus une **grille de densité** pour une heatmap ;
4. produit un **résumé** : fin de mois, veille de rémunération, date d'alerte, perturbateurs.

Aucun appel réseau, aucune clé API. Pas d'interprétation sémantique (rôle du module LLM en aval).

### Installation

```bash
pip install -r requirements.txt
```

### Utilisation

```bash
python -m predictor fixtures/sarah.json --solde 800 --as-of 2026-09-30 --out output.json
```

Le fichier d'entrée est soit une liste de flux, soit un objet `{"flux": [...], "solde_actuel": ..., "as_of": ...}`
(les options `--solde` / `--as-of` priment). Options : `--horizon`, `--seed`, `--n-sim`, `-q`,
`--npy timeline.npy` (timeline en tableau numpy, voir plus bas).
Un tableau des flux classés et le résumé sont affichés sur stderr.

```python
from predictor import predict, Params
sortie = predict(flux, solde_actuel=800, as_of="2026-09-30", horizon_jours=45, params=Params(seed=1))
```

Schéma d'un flux en entrée :

```json
{"categorie": "logement", "marchand": "Agence Namur", "montant": 950.0,
 "charge_fixe": true, "recette": false,
 "dates_apparition": ["2026-08-05", "2026-09-05"], "montants": [950.0, 950.0]}
```

`montant` = montant moyen par occurrence (le signe est donné par `recette`). `montants` est optionnel
(≥ 3 valeurs → écart-type empirique). Les dates postérieures à `as_of` et les doublons sont ignorés
(avec warning). Une entrée invalide lève `ValueError` (CLI : message d'erreur, code 1, pas de trace).

### Timeline en tableau numpy

```python
from predictor import predict, timeline_numpy
t = timeline_numpy(predict(flux, 800, "2026-09-30"))   # tableau structuré, une ligne par jour
t["date"]         # datetime64[D]
t["esperance"], t["q05"], t["q25"], t["q50"], t["q75"], t["q95"], t["p_decouvert"]
```

Depuis le CLI : `--npy timeline.npy`, puis `np.load("timeline.npy")`.

### Version autonome en un seul fichier

`predictor_standalone.py` reprend le même modèle dans un seul fichier (à copier dans un notebook,
par exemple) ; sa seule sortie est la timeline en tableau numpy.

```bash
python predictor_standalone.py fixtures/sarah.json                     # affiche le tableau
python predictor_standalone.py fixtures/sarah.json --out timeline.npy  # l'enregistre
```

```python
from predictor_standalone import predict
t = predict(flux, 800, "2026-09-30")   # mêmes colonnes que timeline_numpy()
```

### Sortie

`flux`, `evenements_prevus`, `timeline`, `grille_densite`, `resume`, `warnings`, plus `parametres` et
`horizon_effectif_jours` (l'horizon est étendu pour couvrir la fin de mois et la veille de
rémunération). Les modèles pydantic de sortie sont dans `predictor/schemas.py`.

### Méthode

| Étape | Fichier | Contenu |
|---|---|---|
| A–D | `flux.py` | période médiane arrondie au calendrier, régularité, P(récurrent) bayésien, P(éteint) selon le retard (périodique) ou le silence (variable, Poisson), réétiquetage en type de changement |
| E | `projection.py` | occurrences futures, p_k = actif·(1−h)^k ; flux variables par fenêtres de 7 jours |
| F | `simulation.py` | Monte Carlo corrélé par flux (une habitude éteinte fait disparaître toutes ses occurrences), timeline, grille |
| — | `resume.py` | points clés, alerte, perturbateurs et leur impact |

Démo : `fixtures/sarah.json` — Sarah a déménagé de Liège à Namur : l'ancien loyer est détecté comme
éteint, le nouveau comme nouvelle habitude, le déménageur comme événement ponctuel, et l'alerte de
découvert tombe le 2026-10-05 (paiement du nouveau loyer avant le salaire du 28).

### Tests

```bash
python -m pytest
```

Tests d'acceptation de la spec (classification des 6 flux, résumé, P(découvert), contrôle Monte Carlo
contre l'espérance analytique, sommes à 1, validation, déterminisme) + CLI.

### Sécurité

Validation stricte (≤ 2000 flux, ≤ 1000 dates par flux, montants finis et ≤ 1e7, dates ISO,
horizon ∈ [1, 365], fichier ≤ 20 Mo) ; pas d'`eval`, `pickle`, `yaml`, `subprocess` ; aucun secret.
Si le module est exposé en HTTP : limiter la taille du body et contrôler l'accès aux données client
(pas d'identifiant client dans l'URL sans contrôle — IDOR).

### Limites connues

- Flux supposés indépendants entre eux (un seul déménagement explique pourtant F002, F003, F004 : la
  corrélation sémantique relève du module LLM en aval).
- Dates d'occurrence futures déterministes (pas de gigue sur le jour de paiement).
- `montant` moyen par occurrence ; la variance réelle n'est connue que si l'amont fournit `montants`.
- Paramètres fixés à la main. En production chez KBC, ρ₀, ρ₁, h et les priors s'apprendraient par
  catégorie sur l'historique réel ; le calcul étant indépendant par client, le passage à 2,3 M de
  clients est un problème de parallélisation.
