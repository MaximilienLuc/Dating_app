# Valeur économique (Alex)

[Notebook exécuté](../../notebooks/06_economics.executed.ipynb) · [Source](../../notebooks/06_economics.ipynb) · Code : `src/economics.py`

Complète `05_performance` (Blanquette) : compare les modèles à des stratégies sans modèle, explique le seuil de
rentabilité, teste la sensibilité aux coûts et ajoute une contrainte de capacité (k profils par utilisateur).
Mêmes hypothèses de coûts (`data/economic_assumptions.json`) et même fonction `calculate_pnl`.
Incertitude : bootstrap apparié des 6 sessions de test (2 000 tirages). Moins d'une minute d'exécution.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/06_economics.ipynb --output 06_economics.executed.ipynb
```

| Fichier | Contenu |
|---|---|
| `01_taux_oui_par_decile_vs_seuil.png` | Pourquoi « montrer tout le monde » est si difficile à battre |
| `02_valeur_ajoutee_vs_montrer_tout.png` | Gain des 6 modèles vs « montrer tout le monde », IC 95 % |
| `03_sensibilite_cout_mauvaise_reco.png` | Valeur du modèle selon le coût d'une mauvaise recommandation |
| `04_top3_par_utilisateur.png`, `05_top3_euros.png` | Contrainte de capacité : 3 profils par utilisateur |
| `06_precision_at_k.png` | Précision@k pour k = 1 à 5 |
| `summary.json`, `*.csv` | Chiffres et tables complètes |

⚠️ Le seuil « optimisé » de TabICL dans `05_performance` est choisi sur ses prédictions du train, vues en contexte
(AUC 0,92 contre 0,63 en test) : il est trop sélectif, d'où un P&L artificiellement mauvais. Utiliser plutôt la règle
théorique p > Y/(X+Y+Z), ou le signaler comme limite.
