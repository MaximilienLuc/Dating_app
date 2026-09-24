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
- `models/logit.joblib`, `models/xgb.joblib`, `models/tabicl.joblib` — tous avec `predict_proba` (`tabicl.joblib` pas encore créé, TabICL bloqué)
- `src/build_dataset.py` — `load_and_clean()` + `engineer_features()`, logique partagée (Alex), importée par `01_data_models_v0.py` et le notebook EDA
- `src/metrics.py` — métriques partagées (Max, pas encore livré)
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

`01_data_models_v0.py` tourne de bout en bout (testé, plus le script v0 non testé d'origine). Il importe `load_and_clean()`/`engineer_features()` depuis `src/build_dataset.py` (logique de nettoyage/feature engineering centralisée, plus dupliquée dans le script), découpe par session (`GroupShuffleSplit`, seed=42), sauvegarde le contrat de fichiers, entraîne logit + XGBoost, et contient une section d'audit (missingness, GroupKFold vs split unique, ANOVA income~race dédupliquée). EDA méthodique faite dans `notebooks/00_eda_cleaning.ipynb` (10 sections, exécuté sans erreur). `docs/data_dictionary.md` et `docs/soutenance_notes.md` livrés et tenus à jour.

Le repo a été mergé avec la branche de stabilité de Remi (`src/stability.py`, `04_stability.ipynb`, `reports/stability/`) — split identique bit à bit des deux côtés, aucun conflit hors `README.md` (résolu). Son `model_factory()` duplique les pipelines logit/xgb déjà dans `01_data_models_v0.py` — dette technique notée, factorisation prévue après le gel des modèles vendredi soir.

**Fait depuis la dernière mise à jour** : retrait de `shar1_1` pour le logit (`features_logit`), correction de l'encodage des dummies (`drop_first=True`, vocabulaire fixe `CAT_CODES`), VIF de contrôle calculé, TabICL réentraîné sur Colab avec le `features.json` à jour (158 features, comparaison désormais à iso-features) — voir point 8 et `docs/soutenance_notes.md` pour le détail. **Reste à faire** : prévenir Max (interprétabilité du logit à refaire sur le nouveau schéma) et Rémi (stabilité à relancer — déjà noté).

**TabICL débloqué, mais pas portable.** `TabICLClassifier.fit()` (tabicl==2.2.0) segfault de façon reproductible sur le chemin CPU — confirmé sur Mac Apple Silicon (M4) **et** sur Colab en CPU (x86_64) : pas un bug spécifique à Apple Silicon, le chemin CPU de la lib est cassé plus largement. Seul `device="cuda"` sur Colab fonctionne. Le `models/tabicl.joblib` qui en résulte est verrouillé sur l'état CUDA (`torch==2.11.0+cu128`) : ne se charge pas sur une machine sans CUDA (plante dès `joblib.load()`, avant `predict_proba()`) — pas portable, mais accepté (usage ponctuel via Colab, pas de réentraînement fréquent). Script versionné dans `colab/train_tabicl.py`. En-tête standardisé dans `src/tabicl_model.py`. Détail complet du diagnostic dans `docs/soutenance_notes.md`.

**Comparaison à 3 modèles obtenue, à iso-features (158)** (AUC test split unique / GroupKFold 5 folds moyenne±écart-type) : logit 0.589 / 0.603±0.024, XGBoost 0.611 / 0.599±0.024, TabICL 0.632 / 0.591±0.038 (Colab, torch==2.11.0+cu128). Les 3 convergent vers ~0.59-0.60 en GroupKFold — plafond du problème, pas un effet d'algorithme. Voir tableau complet dans `docs/soutenance_notes.md`.

### Prochaines étapes (Blanquette)
1. Prévenir Max (logit à réinterpréter sur le nouveau schéma) et Rémi (stabilité à relancer — déjà su, mais schéma dummies aussi changé).
2. Créer `src/logit_model.py`, `src/xgb_model.py` avec docstring standardisé (même format que `src/tabicl_model.py`).
3. Implémenter le P&L (matrice de coûts X=2€/Y=0.5€, seuil optimisé sur GroupKFold train, sensibilité à X/Y — voir `docs/soutenance_notes.md`), y compris pour TabICL (nécessitera de repasser par Colab pour les prédictions).

### Planning
- Jeudi/vendredi : analyses par bloc.
- Samedi : intégration, branchement des vrais modèles dans l'app.
- Dimanche : slides, deux répétitions chronométrées, formation croisée Q&A.
- Lundi 9h40 : envoi final.
