# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Contexte du projet

Projet de groupe HEC (cours *Interpretability, Stability, and Algorithmic Fairness*, Pr. Christophe Pérignon et Dr Sébastien Saurin, MSc DSAIB, septembre 2026).

**Client fictif : « HEC Match »**, une app de rencontre qui décide quels profils montrer à chaque utilisateur. Objectif : analyse de scoring (cible binaire) comparant trois modèles — un white box (logit, idéalement PLTR ou AdaLogit), XGBoost, et TabPFN (Tabular Foundation Model) — sur quatre dimensions : **performance** (statistique et économique), **interprétabilité** (globale et locale), **stabilité**, **fairness**. Il faut arbitrer entre ces dimensions et recommander un modèle selon une logique d'IA de confiance, pas seulement la meilleure AUC.

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

## Équipe

| Membre | Rôle |
|---|---|
| Alex (moi) | Données, variables, découpage, les 3 modèles, performance et P&L |
| Max | Interprétabilité (coefficients, SHAP, LIME, PDP/ICE, permutation importance, XPER) ; `src/metrics.py` ; mail de pré-validation |
| Blanquette | Fairness (tests, FPDP, mitigation, TOST) et récit de la soutenance |
| Remi | Stabilité (absent le premier jour) |
| Oli | App Streamlit (deux profils → proba des 3 modèles + SHAP + onglet fairness + onglet stabilité) et template du deck (absent le premier jour) |

## Contrat de fichiers (pour travailler en parallèle)

**Ces noms et interfaces sont figés — ne pas les renommer ni changer leur forme sans en avertir toute l'équipe.** La liste des variables est figée samedi matin au plus tard.

- `data/clean.parquet` — dataset nettoyé
- `data/features.json` — cible, variables, attributs protégés, variables limites, règle d'exclusion
- `data/split.json` — découpage train/test par session
- `models/logit.joblib`, `models/xgb.joblib`, `models/tabpfn.joblib` — tous avec `predict_proba`
- `src/metrics.py` — métriques partagées (Max)
- Un notebook par bloc :
  - `01_data_models` (Alex)
  - `02_interpretability` (Max)
  - `03_fairness` (Blanquette)
  - `04_stability` (Remi)
- `app/` — application Streamlit (Oli)

## État actuel

`01_data_models_v0.py` est un script v0 **non testé**. Il charge le CSV (encodage ISO-8859-1), construit un profil par participant, renormalise les préférences (sessions 6–9), fusionne les profils de A et B sur `iid`/`pid`, crée les variables de couple et les attributs protégés, encode `field_cd`, `career_c`, `goal` en indicatrices, découpe par session (25 % test), sauvegarde le contrat de fichiers ci-dessus, puis entraîne un logit (imputation + standardisation + régression logistique), XGBoost et TabPFN (avec gestion d'erreur si TabPFN échoue sur CPU).

### Prochaines étapes (Alex)
1. Faire tourner et déboguer la v0, la pousser et prévenir le groupe.
2. Valider le tri des variables avec le dictionnaire (`Speed Dating Data Key.doc`).
3. Remplacer le logit par PLTR ou AdaLogit (package trust-free).
4. Régler XGBoost par validation croisée groupée par session.
5. Faire tourner TabPFN sur GPU (Colab) ou via `tabpfn-client` si besoin.

### Planning
- Jeudi/vendredi : analyses par bloc.
- Samedi : intégration, branchement des vrais modèles dans l'app.
- Dimanche : slides, deux répétitions chronométrées, formation croisée Q&A.
- Lundi 9h40 : envoi final.
