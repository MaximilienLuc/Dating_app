# Arbitrages et recommandation (Alex)

[Notebook exécuté](../../notebooks/07_tradeoffs.executed.ipynb) · [Source](../../notebooks/07_tradeoffs.ipynb) · Code : `src/tradeoffs.py`

Synthèse des 6 modèles sur les 4 dimensions, avec une seule règle de décision pour tous : l'app montre à chaque
utilisateur ses 3 candidats les mieux notés. Lit `reports/economics/` et `reports/interpretability/` : exécuter
`06_economics` et `02_interpretability` avant. Environ 5 minutes (100 réentraînements pour la stabilité).

- **Stabilité** : protocole de Rémi (`src/stability.py`, mêmes hyperparamètres, tirage de sessions d'entraînement),
  étendu aux modèles mitigés (schéma `mitigated_features.json`) et à une mesure propre à l'app : part des 3 profils
  recommandés qui change.
- **Fairness** : rapport des chances d'être dans un top 3 (groupe / candidats blancs), IC par bootstrap de sessions,
  comparé au même rapport dans les « oui » réels.

**Recommandation** : XGBoost mitigé en mode top k par utilisateur, avec stabilisation (bagging), suivi continu de la
fairness et A/B test du coût d'une mauvaise recommandation.

| Fichier | Contenu |
|---|---|
| `01_tableau_arbitrages.png` | Tableau 6 modèles × 6 critères (slide de conclusion) |
| `02_fairness_exposition_top3.png` | Rapports d'exposition par origine, IC 95 % |
| `03_stabilite.png` | Décisions qui basculent et top 3 qui change |
| `summary.json`, `*.csv` | Chiffres complets |
