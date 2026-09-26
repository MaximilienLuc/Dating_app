# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Contexte du projet

Projet de groupe HEC (cours *Interpretability, Stability, and Algorithmic Fairness*, Pr. Christophe Pérignon et Dr Sébastien Saurin, MSc DSAIB, septembre 2026).

**Client fictif : « HEC Match »**, une app de rencontre qui décide quels profils montrer à chaque utilisateur. Objectif : analyse de scoring (cible binaire) comparant trois modèles — un white box (logit), XGBoost, et TabICL (Tabular Foundation Model — remplace TabPFN, initialement prévu) — sur quatre dimensions : **performance** (statistique et économique), **interprétabilité** (globale et locale), **stabilité**, **fairness**. Il faut arbitrer entre ces dimensions et recommander un modèle selon une logique d'IA de confiance, pas seulement la meilleure AUC.

**Dataset** : Speed Dating Experiment (Fisman et Iyengar, Columbia, 2002–2004) — 8 378 lignes, 195 variables brutes, 21 sessions, rendez-vous de 4 minutes. Chaque ligne = un participant sur un rendez-vous (chaque rencontre apparaît deux fois).

**Cible** : `dec` (A veut revoir B, ~40 % de oui). `match` (réciproque, ~16,5 %) sert d'indicateur économique.

### Échéances
- Pré-validation du dataset par mail au prof : **jeudi 24 septembre, 9h40**.
- Envoi final (deck + notebook complet + app de test) : **lundi 28 septembre, 9h40**.
- Soutenance lundi 28 : 15 min présentation + 10 min Q&A (tout membre interrogeable sur tout, code compris).
- Notation /25 : technique /5, présentation /5, Q&A /5, slides+code+app /10.

## Choix méthodologiques clés

1. **Pas de fuite d'information** : n'utiliser que les variables connues *avant* le rendez-vous (questionnaire d'inscription, suffixe `_1`). Exclure toutes les notes données pendant/après la soirée (`attr`, `sinc`, `like`, `prob`, `*_o`, `match_es`, `*_s`, `*_2`, `*_3`, `dec_o`, `match`) et l'ordre des rendez-vous. AUC attendue modeste (~0,65–0,70) : c'est un résultat à présenter, pas un échec.
2. **Split train/test par session (wave)** avec `GroupShuffleSplit`/`GroupKFold`, seed = 42. Un split par participant ne suffit pas (fuite des paires A/B et B/A).
3. **Sessions 6 à 9** : préférences notées sur 10 au lieu de 100 points → à renormaliser en parts de 100. Le Data Key.doc décrit fidèlement le protocole de collecte de waves 6-9 (notation 1-10 indépendante par attribut). Cependant, vérification empirique faite : le CSV tel que distribué sur Kaggle contient déjà la conversion de ces notes en parts de 100, appliquée en amont par les auteurs/curateurs — pas au moment de la collecte, et pas par notre pipeline. Confirmé par deux tests indépendants : (1) empreinte décimale (waves 6-9 : 47 décimales distinctes, valeurs "irrégulières" type 12.77 — incompatible avec une répartition manuelle de 100 points ; autres waves : 6 décimales distinctes, valeurs rondes typiques d'une répartition manuelle), (2) test de reconstruction exacte note_i/S×100 : 101/101 lignes reconstructibles sur waves 6-9, contre 61% de faux positifs structurels sur les autres waves utilisées comme groupe de contrôle. La renormalisation de notre pipeline est donc un no-op sur ces données (idempotente) — son vrai rôle utile est de corriger les ~2% de lignes, toutes waves confondues, où la somme déclarée s'écarte de 100 (erreurs de saisie des répondants), pas d'harmoniser une échelle entre groupes de waves.
4. **Fairness = équité d'exposition** : les attributs protégés (`cand_female`, `cand_race`, `same_race`, `gender_A`, `race_A`) servent **uniquement à l'audit**, jamais en entrée modèle. Débat central : reproduire les préférences des utilisateurs (y compris raciales) ou les corriger au prix de la précision. Méthodes attendues : tests de fairness, FPDP (proxy), mitigation, test d'équivalence TOST.
5. **P&L économique** : oui = +X €, non = −Y €, match manqué = revenu perdu. Seuil optimisé sur le P&L + analyse de sensibilité à X et Y.
6. **Stabilité** : train/test croisé entre sessions + bootstrap. Mesurer distances entre coefficients / importances, et % de décisions qui basculent.
7. **Limites à annoncer** : données anciennes, étudiants Columbia uniquement, beaucoup de valeurs manquantes (87 % des lignes, 1,8 % des cellules), origine utilisée seulement pour l'audit.
8. **Colinéarité des 6 préférences pour le logit** : `attr1_1+sinc1_1+intel1_1+fun1_1+amb1_1+shar1_1` somme à 100 par construction (confirmé structurel sur toutes les waves, pas un artefact — voir point 3), donc parfaitement colinéaires. **Implémenté** : `shar1_1_A`/`shar1_1_B` retirées des features du **logit uniquement** (`feature_dict["features_logit"]`, XGBoost/TabICL gardent les 6) — analogue à `drop_first=True`. Alternative plus rigoureuse (transform log-ratio sur données compositionnelles) écartée pour le calendrier. VIF de contrôle fait sur les 156 features finales du logit : 2 paires de colonnes à VIF infini documentées comme limite connue (petits effectifs sur des catégories rares), pas corrigées faute de temps — voir `docs/soutenance_notes.md`.

## Équipe

| Membre | Rôle |
|---|---|
| Blanquette (moi) | **Rôles échangés avec Alex** : données, EDA, les 3 modèles, performance et P&L (voir tout ce document — c'est le travail fait jusqu'ici) |
| Alex | Rôle échangé avec Blanquette — à confirmer ce qu'il reprend (pas fairness, c'est Max qui l'a pris) |
| Max | Interprétabilité (coefficients, SHAP, LIME, PDP/ICE, permutation importance, XPER) ; mail de pré-validation. **A aussi démarré la fairness** (branche `fairness` : TOST, audit racial, `src/metrics.py`, `src/generate_fairness_reports.py`) — rôle réellement couvert plus large que prévu initialement |
| Remi | Stabilité — **livré et mergé** : bootstrap par session sur logit/xgb (`src/stability.py`, `04_stability.ipynb`, `reports/stability/`). À relancer une fois `features_logit` disponible (retrait shar1_1) ; TabICL à ajouter une fois qu'il aura accès à Colab/CUDA lui aussi |
| Oli | App Streamlit (deux profils → proba des 3 modèles + SHAP + onglet fairness + onglet stabilité) et template du deck (absent le premier jour) |

## Contrat de fichiers (pour travailler en parallèle)

**Ces noms et interfaces sont figés — ne pas les renommer ni changer leur forme sans en avertir toute l'équipe.** La liste des variables est figée samedi matin au plus tard.

- `data/clean.parquet` — dataset nettoyé
- `data/features.json` — cible, variables (`features`, 158, dummies drop_first=True depuis la correction), `features_logit` (158 moins shar1_1_A/B, 156), attributs protégés, variables limites (`borderline` + `borderline_reasons`), règle d'exclusion
- `data/split.json` — découpage train/test par session
- `models/logit.joblib`, `models/xgb.joblib`, `models/tabicl.joblib` — tous avec `predict_proba`
- **`models/xgb_mitigated.joblib`** ⚠️ — **version débiaisée de XGBoost à utiliser pour toutes les analyses (SHAP, P&L, soutenance)**. Les 4 proxies raciaux ont été retirés : `attr3_1_B`, `career_c_B_12`, `go_out_B`, `intel3_1_B`. Performances quasi-identiques (AUC −0.8pp). Ne jamais utiliser `xgb.joblib` pour les analyses finales.
- `models/logit_mitigated.joblib` — version débiaisée du logit (mêmes proxies retirés)
- **`data/tabicl_predictions.parquet`** ⚠️ — prédictions TabICL pré-calculées sur Colab (colonnes : `iid`, `pid`, `wave`, `tabicl_proba`). À utiliser **à la place de** `models/tabicl.joblib` qui est verrouillé CUDA et ne charge pas en local.
- `src/build_dataset.py` — `load_and_clean()` + `engineer_features()`, logique partagée (Alex), importée par `01_data_models_v0.py` et le notebook EDA
- `src/metrics.py` — métriques partagées (Max, livré)
- `src/stability.py` — bootstrap de stabilité par session (Remi, livré)
- Un notebook par bloc :
  - `00_eda_cleaning` (Alex) — EDA et nettoyage, livré
  - `01_data_models` (Alex)
  - `02_interpretability` (Max)
  - `03_fairness` (Blanquette)
  - `04_stability` (Remi, livré : `04_stability.ipynb` + `.executed.ipynb`)
- `docs/data_dictionary.md`, `docs/soutenance_notes.md` — livrés (Alex)
- `reports/stability/` — sorties du bootstrap de Remi (CSV, PNG, `summary.json`)
- `app/` — application Streamlit (Oli)

## État actuel

`01_data_models_v0.py` tourne de bout en bout. Il importe `load_and_clean()`/`engineer_features()` depuis `src/build_dataset.py`, découpe par session (`GroupShuffleSplit`, seed=42), sauvegarde le contrat de fichiers, entraîne logit + XGBoost, et contient une section d'audit. EDA méthodique faite dans `notebooks/00_eda_cleaning.ipynb` (10 sections). `docs/data_dictionary.md` et `docs/soutenance_notes.md` livrés et tenus à jour.

### Fairness — état branche `fairness` (commit `d3076ae`, pushé)

- **TOST (δ=10pp, α=5%)** implémenté dans `src/metrics.py`.
- **Mitigation FPDP** (`src/mitigation.py`) : 4 proxies raciaux identifiés et retirés → `xgb_mitigated.joblib` + `logit_mitigated.joblib`.
- **Décomposition biais sociétal vs algorithmique** (`src/fairness_bias_decomposition.py`) : sépare ce qui vient des données humaines vs ce qu'ajoutent les modèles. Résultats dans `data/fairness_bias_decomposition.json`, graphe dans `reports/fairness/bias_decomposition.png`.
- **Argument soutenance** : nos modèles ne sont pas structurellement racistes. Les échecs TOST (Asiatiques −8.6pp, Latinos −6.5pp) reflètent le biais sociétal des participants humains Speed Dating, pas un biais algorithmique. XGBoost ajoute ≤4pp de biais propre. Le Logit ajoute −12.5pp vs Latinos → renforce la recommandation XGBoost mitigé.
- **TabICL** : toujours GPU-only (Colab). Utiliser `data/tabicl_predictions.parquet`, **ne jamais appeler `joblib.load('models/tabicl.joblib')` en local** (segfault CUDA garanti).

Le repo a été mergé avec la branche de stabilité de Remi (`src/stability.py`, `04_stability.ipynb`, `reports/stability/`) — split identique bit à bit des deux côtés, aucun conflit hors `README.md` (résolu). Son `model_factory()` duplique les pipelines logit/xgb déjà dans `01_data_models_v0.py` — dette technique notée, factorisation prévue après le gel des modèles vendredi soir.

**Fait depuis la dernière mise à jour** : retrait de `shar1_1` pour le logit (`features_logit`), correction de l'encodage des dummies (`drop_first=True`, vocabulaire fixe `CAT_CODES`), VIF de contrôle calculé, TabICL réentraîné sur Colab avec le `features.json` à jour (158 features, comparaison désormais à iso-features) — voir point 8 et `docs/soutenance_notes.md` pour le détail. **Reste à faire** : prévenir Max (interprétabilité du logit à refaire sur le nouveau schéma) et Rémi (stabilité à relancer — déjà noté).

**TabICL débloqué, mais pas portable.** `TabICLClassifier.fit()` (tabicl==2.2.0) segfault de façon reproductible sur le chemin CPU — confirmé sur Mac Apple Silicon (M4) **et** sur Colab en CPU (x86_64) : pas un bug spécifique à Apple Silicon, le chemin CPU de la lib est cassé plus largement. Seul `device="cuda"` sur Colab fonctionne. Le `models/tabicl.joblib` qui en résulte est verrouillé sur l'état CUDA (`torch==2.11.0+cu128`) : ne se charge pas sur une machine sans CUDA (plante dès `joblib.load()`, avant `predict_proba()`) — pas portable, mais accepté (usage ponctuel via Colab, pas de réentraînement fréquent). Script versionné dans `colab/train_tabicl.py`. En-tête standardisé dans `src/tabicl_model.py`. Détail complet du diagnostic dans `docs/soutenance_notes.md`.

**Comparaison à 3 modèles obtenue, à iso-features (158)** (AUC test split unique / GroupKFold 5 folds moyenne±écart-type) : logit 0.589 / 0.603±0.024, XGBoost 0.611 / 0.599±0.024, TabICL 0.632 / 0.591±0.038 (Colab, torch==2.11.0+cu128). Les 3 convergent vers ~0.59-0.60 en GroupKFold — plafond du problème, pas un effet d'algorithme. Voir tableau complet dans `docs/soutenance_notes.md`.

**Fairness (Max) mergée dans `main`** : branche `fairness` (TOST, audit racial, `src/metrics.py`, `src/generate_fairness_reports.py`, `reports/fairness/`) mergée sans conflit avec `blanche`. Tag `avant-merge-blanquette-2026-09-25` posé sur `origin/main` avant ce merge (sécurité, sur GitHub).

**Performance prédictive (statistique + économique) faite sur `blanche-performance`** (pas encore mergée) : `src/performance.py`, notebook `notebooks/05_performance.ipynb`, résultats dans `reports/performance/`. PR-AUC, calibration, matrice de confusion, XPER (income_A/B absent du top 10 des 2 modèles calculables), seuil P&L optimisé par GroupKFold, test de robustesse (2 scénarios — **le classement des 3 modèles n'est pas stable**, TabICL passe premier à coûts égaux X=1/Y=1). XPER-TabICL tenté sur Colab (`colab/xper_tabicl.py`) — **conclu non exploitable** (voir plus bas et `docs/soutenance_notes.md`). **Note opérationnelle** : le calcul XPER-XGBoost local a pris 12h31 à cause de la mise en veille du Mac (process gelé, pas planté) — utiliser `caffeinate -w <PID>` pour les prochains calculs longs sur cette machine.

**Fait** : réalignement des hypothèses économiques sur celles de Max (`X=2, Y=1, Z=1` dans `data/economic_assumptions.json`, miroir de `GAIN_TP=2, COST_FP=1, COST_FN=1` dans `src/mitigation.py`) — ajoute un coût aux occasions manquées (Z/FN), absent de notre première version. Comparaison faite : nos 3 modèles de base (logit/xgb/tabicl) vs les 2 modèles mitigés de Max (logit_mitigated/xgb_mitigated, pas de TabICL mitigé pour l'instant) sur PR-AUC/P&L/seuil optimal — **impact économique de la mitigation faible et pas clairement défavorable** (logit +22 de P&L optimal après mitigation, xgb -5 ; à confirmer par un test de significativité, pas encore fait — voir "Rémi" ci-dessous). **Incompatibilité de schéma découverte et résolue** : les modèles mitigés de Max utilisent l'ancien encodage à 164 features (avant notre `drop_first`) — reconstruction exacte des 6 colonnes de référence manquantes (`field_cd_A_1` etc. = 1 − somme des autres catégories, encodage one-hot exhaustif) plutôt que d'attendre que Max relance sur le nouveau schéma.

XPER étendu à 3 métriques (base + mitigés, logit/xgb) : `AUC` (perf statistique), `MC` (coût de classification brut de Max), et **`compute_xper_pnl` (nouveau)** — décomposition exacte de notre vrai P&L (pas juste le coût sans récompense TP de `MC`), dérivée algébriquement (`P&L = X·P − (X+Z)·FN − Y·FP`, la constante X·P n'affecte aucune feature). **Bug vérifié dans la librairie XPER** au passage : ses paramètres `CFP`/`CFN` sont inversés en interne par rapport à leur docstring (`CFP` pondère en réalité les FN, `CFN` les FP) — sans incidence tant que les deux coûts sont égaux (notre cas, Y=Z=1), documenté et corrigé dans `src/performance.py` pour ne pas piéger un futur usage à coûts asymétriques.

**TabICL mitigé — exécuté et intégré.** Script de Max (`colab/train_tabicl_mitigated.py`/`.ipynb`, branche `fairness`, pas encore mergée dans `main`) lancé par Blanquette sur Colab GPU T4. Sorties : `models/tabicl_mitigated.joblib` (CUDA-only, ne pas charger en local) et **`data/tabicl_mitigated_predictions.parquet`** ⚠️ (colonnes `iid`, `pid`, `wave`, `tabicl_mit_proba` — à utiliser comme `data/tabicl_predictions.parquet`, jointure sur `iid`/`pid`/`wave`, jamais de `joblib.load` du `.joblib`). **Piège pyarrow, corrigé pour toute l'équipe** : ce parquet est écrit par une version de pyarrow plus récente que celle qu'utilisait `requirements.txt` — lecture échoue avec `OSError: Repetition level histogram size mismatch` sinon. **`requirements.txt` a été mis à jour (`pyarrow==19.0.0` → `pyarrow==25.0.1`)** — tout le monde doit relancer `pip install -r requirements.txt --upgrade` (ou `pip install -U pyarrow`) pour ne pas planter en chargeant ce fichier. Compatibilité ascendante : ce pyarrow plus récent lit aussi bien les anciens fichiers (`tabicl_predictions.parquet`, écrit par l'ancien pin) que le nouveau. `src/performance.py` (`load_mitigated_predictions`, `build_model_specs`) intègre déjà ce 3e modèle mitigé ; perf stat/éco (Part 1/3) calculée comme pour les 5 autres modèles. XPER hors de portée en local (verrouillage CUDA) et **conclu non exploitable même sur Colab** pour TabICL de base (voir ci-dessous) — jamais tenté sur la version mitigée en conséquence.

### Prochaines étapes

**Max (interprétabilité)** :
- Relancer SHAP, LIME, PDP/ICE sur **`xgb_mitigated.joblib`** (pas `xgb.joblib`). Les features `attr3_1_B`, `career_c_B_12`, `go_out_B`, `intel3_1_B` ont été supprimées — elles n'apparaîtront plus dans les graphes SHAP.
- Pour le Logit, utiliser `features_logit` (156 features, `drop_first=True`) et `logit_mitigated.joblib` — **⚠️ à signaler à Max, confirmé par vérification directe de `src/mitigation.py`/`mitigated_features.json`** : `logit_mitigated.joblib` cumule DEUX problèmes de colinéarité non corrigés (entraîné sur l'ancien schéma) : (1) `shar1_1_A`/`shar1_1_B` toujours présentes (160 features, pas 156) — colinéarité compositionnelle du point 8 ; (2) `drop_first=True` jamais appliqué — les 6 colonnes de référence (`field_cd_A_1`, `career_c_A_1`, `goal_A_1`, etc.) sont toutes là, piège classique des variables muettes. Effet déjà visible : `career_c_B_1` (une colonne de référence piégée) apparaît dans le top 5 XPER-PNL de `logit_mitigated` — probable artefact d'encodage, pas un vrai signal, voir `docs/soutenance_notes.md` "Performance prédictive". Pas encore vérifié par VIF.

**Rémi (stabilité)** :
- Relancer `src/stability.py` avec `features_logit` pour le Logit (schéma dummies changé, `shar1_1_A`/`shar1_1_B` retirées + `drop_first=True`).
- Idéalement : relancer aussi avec les modèles mitigés pour comparer stabilité avant/après mitigation.
- TabICL : utiliser `data/tabicl_predictions.parquet` pour les prédictions (pas de `joblib.load`).
- **Nouveau** : tester la stabilité du **seuil optimal P&L** maintenant que la matrice de coûts est fixée (X=2/Y=1/Z=1, `data/economic_assumptions.json`) — son propre README de stabilité notait que c'était différé faute de matrice de coûts définie, ce qui est fait maintenant.
- **Nouveau** : un test de significativité formel (bootstrap apparié par session, même logique que `paired_auc_differences`) sur l'écart de P&L base vs mitigé (logit +22, xgb -5 en point estimé, pas encore de CI) — proposé à Rémi plutôt que fait par Blanquette, pour réutiliser directement son infrastructure de bootstrap. Message détaillé envoyé hors-repo.

**Oli (app Streamlit)** :
- Brancher `xgb_mitigated.joblib` (pas `xgb.joblib`) dans l'interface.
- Pour TabICL, charger `data/tabicl_predictions.parquet` et faire une jointure sur `iid`/`pid` plutôt qu'appeler `predict_proba`.
- **Nouveau** : `logit_mitigated.joblib` disponible aussi (mêmes réserves que pour Max plus haut sur sa colinéarité non corrigée — utilisable pour la proba, moins pour des coefficients affichés tels quels). `tabicl_mitigated` disponible en prédictions figées : `data/tabicl_mitigated_predictions.parquet`, colonne `tabicl_mit_proba`, même jointure `iid`/`pid`/`wave` que TabICL de base — jamais de `joblib.load('models/tabicl_mitigated.joblib')` en local (même verrouillage CUDA).

**Blanquette (moi, restant)** :
1. P&L réaligné sur Max (X=2/Y=1/Z=1) et étendu à la comparaison base vs mitigé, 3 paires (logit/xgb/tabicl) — fait (voir "Fait" ci-dessus). Sur les 3 paires, aucune dégradation économique franche de la mitigation, 2 l'améliorent même légèrement (logit +22, tabicl +16 ; xgb -5) — bon signal pour l'argumentaire business, pas encore de test de significativité (proposé à Rémi).
2. Merger `blanche-performance` dans `main` une fois `notebooks/05_performance.ipynb` resynchronisé avec les nouveaux résultats (actuellement narre encore l'ancienne version à 3 modèles).
3. Prévenir Max (logit à réinterpréter sur `features_logit` + vérifier la collinéarité shar1_1 dans son modèle mitigé) et Rémi (stabilité à relancer + significativité base/mitigé + stabilité du seuil P&L) — message Rémi rédigé, à envoyer.
4. Créer `src/logit_model.py`, `src/xgb_model.py` avec docstring standardisé (même format que `src/tabicl_model.py`) — toujours pas fait.
5. ~~Tenter `colab/xper_tabicl.py` pour XPER-TabICL~~ — fait, 3 tentatives, **conclu non exploitable** (2 crashes mémoire OOM, puis résultat statistiquement dégénéré à N_coalition_sampled=3 : 8 valeurs uniques sur 158 features). Acté à ne pas retenter, voir `docs/soutenance_notes.md`.
6. Nettoyer les anciens fichiers de sortie `reports/performance/` sous l'ancienne convention de nommage (`summary_3_models.csv`, `xper_values_logit.csv`, `xper_values_xgb.csv`), remplacés par `summary_all_models.csv` et `xper_{auc,mc,pnl}_*.csv`.

### Planning
- Jeudi/vendredi : analyses par bloc.
- Samedi : intégration, branchement des vrais modèles dans l'app.
- Dimanche : slides, deux répétitions chronométrées, formation croisée Q&A.
- Lundi 9h40 : envoi final.
