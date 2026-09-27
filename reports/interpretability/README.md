# Interprétabilité (Alex)

[Notebook exécuté](../../notebooks/02_interpretability.executed.ipynb) · [Source](../../notebooks/02_interpretability.ipynb) · Code : `src/interpretability.py`

## Reproduire

```bash
pip install -r requirements-interpretability.txt
jupyter nbconvert --to notebook --execute notebooks/02_interpretability.ipynb --output 02_interpretability.executed.ipynb
```

Environ 3 minutes sur un portable (le bootstrap du logit et LIME sur 100 couples sont les étapes les plus longues).
N'utilise ni `xgb.joblib` ni `tabicl*.joblib` : XGBoost mitigé est chargé tel quel, le logit est ré-entraîné
proprement (voir ci-dessous), TabICL est lu via `data/tabicl_mitigated_predictions.parquet`.

## Choix

- **XGBoost mitigé** : SHAP exact (TreeExplainer, additivité vérifiée), PDP/ICE, LIME, permutation importance.
- **Logit** : ré-entraîné sur `features_logit` sans les 4 proxies (152 variables, AUC test 0,578), car les
  coefficients de `logit_mitigated.joblib` sont faussés par la colinéarité de l'ancien schéma. IC à 95 % par
  bootstrap de 200 tirages de sessions d'entraînement.
- **TabICL mitigé** : surrogate sur les prédictions TEST uniquement (sur le train, TabICL est « in-context » :
  AUC 0,92). Arbre de profondeur 3 (lisible, fidélité R² ≈ 0) et XGBoost de profondeur 2 (fidélité R² ≈ 0,34).

## Fichiers pour les slides et l'app

| Fichier | Contenu |
|---|---|
| `01_shap_global_top15.png` | Top 15 variables SHAP, couleur = côté A / B |
| `02_shap_by_side_and_category.png` | 53 % A, 44 % B, 4 % couple ; par famille |
| `03_shap_beeswarm.png` | Sens des effets |
| `04_local_shap_*.png` | Deux couples expliqués (84 % et 9 %) |
| `05_shap_vs_lime_*.png` | Disagreement problem sur un couple |
| `06_pdp_ice.png`, `06b_pdp_ice_top.png` | PDP + ICE centrées |
| `08_logit_odds_ratios.png` | Odds ratios avec IC |
| `09_tabicl_surrogate_tree.png` | Illusion d'interprétabilité |
| `10_agreement_between_methods.png` | Corrélation des classements entre méthodes |
| `summary.json` | Tous les chiffres clés (lisible par l'app) |
| `*.csv` | Tables complètes |
