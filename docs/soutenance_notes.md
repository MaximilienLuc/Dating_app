# Notes de soutenance — HEC Match

## Choix méthodologiques à assumer explicitement
- `dec` est un jugement post-rencontre réelle (4 min face à face), pas un like/dislike sur une
  fiche pré-rencontre — le meilleur proxy disponible de compatibilité réelle, à présenter comme
  un choix assumé, pas une faiblesse
- Seules les variables pré-rendez-vous (`_1`) sont utilisées en features — conséquence directe :
  AUC modeste (~0.60) attendue et acceptée
- Split strict par wave (GroupShuffleSplit/GroupKFold) obligatoire : chaque rencontre génère 2
  lignes symétriques (A→B et B→A) qui partagent des variables de couple — un split aléatoire
  fuiterait
- Architecture train / validation (CV interne par wave) / test : le test n'est touché qu'une
  fois à la toute fin ; tout réglage (hyperparamètres, seuil P&L, mitigation fairness) passe par
  le GroupKFold sur les waves restantes

## Dérive de protocole entre waves
- Waves 6-9 : préférences notées 1-10 au lieu de 100 points réparties — renormalisées en amont
  dans le pipeline (feature engineering unique pour toutes les waves, pas de modèle séparé)
- Wave 13 : blackout total de la collecte income (100% manquant) — cas structurel à part, pas
  un manquant aléatoire à imputer comme les autres
- Le Data Key.doc décrit fidèlement le protocole de collecte de waves 6-9 (notation 1-10
  indépendante par attribut). Cependant, vérification empirique faite : le CSV tel que
  distribué sur Kaggle contient déjà la conversion de ces notes en parts de 100, appliquée en
  amont par les auteurs/curateurs — pas au moment de la collecte, et pas par notre pipeline.
  Confirmé par deux tests indépendants : (1) empreinte décimale (waves 6-9 : 47 décimales
  distinctes, valeurs "irrégulières" type 12.77 — incompatible avec une répartition manuelle de
  100 points ; autres waves : 6 décimales distinctes, valeurs rondes typiques d'une répartition
  manuelle), (2) test de reconstruction exacte note_i/S×100 : 101/101 lignes reconstructibles
  sur waves 6-9, contre 61% de faux positifs structurels sur les autres waves utilisées comme
  groupe de contrôle. La renormalisation de notre pipeline est donc un no-op sur ces données
  (idempotente) — son vrai rôle utile est de corriger les ~2% de lignes, toutes waves
  confondues, où la somme déclarée s'écarte de 100 (erreurs de saisie des répondants), pas
  d'harmoniser une échelle entre groupes de waves.

## Validation croisée — comparaison à 3 modèles

| Modèle | AUC test (split unique) | AUC GroupKFold (5 folds, moyenne ± écart-type) |
|---|---|---|
| Logit | 0.589 | 0.603 ± 0.024 |
| XGBoost | 0.611 | 0.599 ± 0.024 |
| TabICL | 0.632 | 0.591 ± 0.038 |

Comparaison **à iso-features** (158 features, schéma post-correction dummies `drop_first`,
mêmes pour les 3 modèles) : TabICL réentraîné sur Colab avec le `features.json` à jour --
résultat quasi identique à la version pré-correction (0.638/0.591±0.039 -> 0.632/0.591±0.038),
confirmant que la correction des dummies ne changeait pas l'information disponible, juste des
colonnes redondantes.

- Les 3 modèles convergent vers le même plafond ~0.59-0.60 en GroupKFold, malgré des
  architectures très différentes (linéaire, arbres boostés, transformer en in-context
  learning) — confirme que le plafond reflète la difficulté intrinsèque du problème (prédire
  une alchimie à partir d'un profil pré-rencontre), pas un choix d'algorithme sous-optimal.
  Cohérent avec le test RF vs XGBoost (voir "Choix XGBoost vs Random Forest").
- TabICL a l'AUC test (split unique) la plus haute (0.632) mais aussi l'écart-type GroupKFold
  le plus large (±0.038, contre ±0.024 pour logit/xgb) — son résultat sur le split unique
  est probablement optimiste (fold 5 tombe à 0.533, fold 3 monte à 0.652) plutôt qu'un vrai
  avantage. **Le chiffre GroupKFold est celui à citer en priorité**, pas le split unique.
- Chiffres TabICL obtenus sur Colab (`device="cuda"`, torch==2.11.0+cu128) — voir "TabICL — diagnostic du crash CPU
  et décision". Logit/XGBoost obtenus en local (`01_data_models_v0.py`).

## income_A / income_B — proxy potentiel
- Revenu médian du zip code déclaré, pas un revenu individuel — à clarifier dans le rapport
- ~46% de valeurs manquantes en moyenne (hors wave 13), réparties au niveau individuel
- ANOVA income_A ~ race_A (individus uniques, n=281) : F=2.66, p=0.033, eta²=0.037 — effet réel
  mais fragile, petits effectifs sur groupes minoritaires (Black n=18, Latino n=16)
- Contrôle croisé (income_B ~ race_A, income_A ~ cand_race) : quasi nul (p=0.61) — confirme que
  le lien est bien income↔origine du propriétaire, pas un artefact d'appariement
- Catégorie "Native Am." totalement absente des données — limite structurelle

## Piège méthodologique à documenter (leçon générale)
- Tester sur les paires (4273 lignes, chaque personne ~15x) au lieu des individus uniques (281)
  gonfle artificiellement la significativité par pseudo-réplication (F=48.68 sur paires vs
  F=2.66 dédoublonné) — toujours dédoublonner avant un test statistique par groupe protégé

## Méthode
- Ne pas se fier à la documentation du dataset (Data Key.doc) sans la confronter aux valeurs
  réelles du CSV : exemple concret avec la renormalisation waves 6-9 (voir "Dérive de protocole
  entre waves") — le Key décrit le protocole de collecte d'origine, mais ne dit rien de ce que
  le CSV publié contient réellement après d'éventuels traitements en amont par les
  auteurs/curateurs. Vérifié empiriquement par empreinte décimale + test de reconstruction
  exacte plutôt que supposé. À mentionner si le prof demande comment on valide nos hypothèses
  de nettoyage.

## Feature engineering
- income_missing_A / income_missing_B : flag binaire créé avant imputation, découle
  directement du taux de missing observé en EDA (~46%)
- Stratégie en deux passes : modèles d'abord AVEC income tel quel + mesure de disparité, PUIS
  variante avec income neutralisé/retiré pour tester si la disparité persiste — miroir direct de
  l'argument central du projet foot ("retirer l'attribut protégé ne suffit pas")
- **Implémenté** : retrait de shar1_1_A et shar1_1_B des features du logit uniquement
  (XGBoost et TabICL gardent les 6 préférences) — `feature_dict["features_logit"]`
  (`features.json`), construit dans `engineer_features()`. Casse la colinéarité parfaite créée
  par la contrainte de somme à 100 (confirmée structurelle sur toutes les waves, pas seulement
  6-9 — voir "Dérive de protocole entre waves"). AUC quasi inchangée après le retrait (logit
  0.589 split unique / 0.603±0.024 GroupKFold, contre 0.587/0.603±0.024 avant).
- **Implémenté** : encodage des catégorielles (field_cd/career_c/goal) en one-hot avec
  `drop_first=True` sur un vocabulaire de catégories FIXE (`CAT_CODES` dans
  `src/build_dataset.py`, pas les valeurs observées) — garantit des colonnes identiques quel
  que soit le sous-échantillon, et un encodage non basé sur une statistique calculée sur les
  données (donc pas de fuite). La catégorie "1" de chaque variable est la référence implicite.
- **VIF de contrôle fait** sur les 156 features finales du logit : 19 variables > 10, dont 8 à
  l'infini. Diagnostiqué avant de conclure : `age_diff`/`age_A`/`age_B` (55/32/32) et
  `fit_score`/`attr1_1_A` (12-15) sont des colinéarités structurelles attendues (age_diff =
  age_B - age_A ; fit_score construit à partir de attr1_1_A). Les VIF infinis sont deux paires
  de colonnes **exactement identiques**, propres au petit effectif de ce split train : (1)
  `field_cd_A_17` ≡ `career_c_A_17` (les 10 personnes étudiant l'architecture en train visent
  toutes aussi une carrière d'architecte — mêmes lignes, mêmes côtés A et B) ; (2)
  `field_cd_A_NA` ≡ `goal_A_NA` (les 58 personnes n'ayant pas renseigné leur domaine d'études
  sont exactement celles n'ayant pas non plus renseigné leur objectif de la soirée — tout un
  bloc du questionnaire sauté ensemble, mêmes côtés A et B). **Décision : documenté tel quel,
  pas corrigé** (fusion des catégories concernées écartée faute de temps) — deux coefficients
  du logit seront indéterminés/redondants dans ces paires, sans remettre en cause le reste du
  modèle.
- Alternative plus rigoureuse envisagée puis écartée : transformation centered log-ratio (CLR)
  sur les 6 préférences — méthodologiquement supérieure pour des données compositionnelles, mais
  écartée par souci de calendrier (modèles gelés vendredi soir) ; le retrait simple reste
  défendable pour ce niveau de projet — phrase à avoir prête en Q&A si le sujet vient.

## Économie / P&L
- Matrice de coûts : profil montré + oui = +X, profil montré + non = −Y, match manqué = revenu
  perdu
- Seuil de décision à optimiser sur ce P&L plutôt que sur l'AUC seule, avec analyse de
  sensibilité à X et Y — répond à l'exigence officielle du brief de performance "économique" en
  plus de "statistique"
- Simplification assumée : le P&L est calculé au niveau de la décision individuelle (dec), pas
  au niveau du vrai match mutuel (match) qui nécessiterait de croiser deux décisions d'une paire
  — limite documentée, pas cachée
- Hypothèses de départ : X=2€ (recommandation qui aboutit à un oui), Y=0.5€ (recommandation
  gâchée) — hypothèses narratives assumées, pas mesurées
- Seuil de décision optimisé par grille sur GroupKFold (train), jamais sur le test final ; test
  final touché une seule fois avec le seuil retenu
- Analyse de sensibilité prévue sur 3 jeux d'hypothèses (X=2/Y=0.5, X=1/Y=1, X=3/Y=0.3) pour
  vérifier la stabilité du modèle gagnant et du seuil optimal

## TabICL — diagnostic du crash CPU et décision
- `TabICLClassifier.fit()` (tabicl==2.2.0) provoque un segfault natif (SIGSEGV) reproductible.
  Diagnostic méthodique fait avant de conclure quoi que ce soit : environnement vérifié
  identique pour toutes les libs (pas de mismatch type pandas/xgboost du début de projet),
  install/import OK, crash uniquement à `.fit()` et immédiat (pas un hang) ; dtypes/NaN/Inf
  vérifiés propres avant et après imputation (aucune anomalie) ; taille (30 à 500 lignes),
  nombre de colonnes (1 à 164) et équilibre des classes testés sans effet sur le crash ;
  données 100% synthétiques (`np.random.randn`, même forme exacte) ne crashent jamais.
  Recherche binaire sur les 164 colonnes : n'importe quelle colonne seule fait déjà crasher
  (pas une colonne coupable en particulier) — la variable qui distingue crash/pas-crash est
  "données réelles" vs "synthétiques", pas la forme ni le contenu d'une colonne précise.
- Confirmé sur DEUX machines : Mac Apple Silicon (M4, torch 2.14.0, CPU) ET Colab (x86_64, CPU)
  crashent tous les deux. **Ce n'est donc pas un bug spécifique à Apple Silicon** : le chemin
  CPU de tabicl est cassé plus largement (au moins sur ces deux architectures).
- Seul `device="cuda"` sur Colab fonctionne (`torch==2.11.0+cu128` observé). Mais le `.joblib`
  qui en résulte est verrouillé sur l'état CUDA : le charger sur une machine sans CUDA plante
  aussi, dès `joblib.load()`, avant même `predict_proba()` — donc pas portable.
- Décision d'équipe : accepté. L'usage de TabICL est ponctuel (pas de réentraînement fréquent,
  dashboard Streamlit pas concerné), pas besoin qu'il tourne ailleurs que sur Colab pour
  l'instant. Script d'entraînement/inférence versionné dans `colab/train_tabicl.py` (à coller
  dans une cellule Colab, runtime GPU requis — Modifier > Paramètres du notebook > GPU).

## Coordination équipe / dette technique
- Merge avec le travail de stabilité de Rémi fait le 24/09 — split.json identique bit à bit des
  deux côtés (mêmes train_waves/test_waves), aucun recalcul nécessaire sur ce point
- **features.json a maintenant une clé features_logit** séparée de features (retrait shar1_1,
  voir "Feature engineering") — **Rémi à prévenir** : son model_factory() (src/stability.py)
  lit un seul contract['features'] identique pour logit ET xgb, donc il devra explicitement
  lire features_logit pour le logit et relancer sa stabilité pour qu'elle reflète le retrait
  de shar1_1. Le schéma de dummies a aussi changé (drop_first=True, 158 features au lieu de
  164) : sa stabilité est à relancer de toute façon, pas seulement pour shar1_1.
- **Max à prévenir aussi** (pas encore fait) : même chose pour toute interprétabilité déjà
  commencée sur le logit avec l'ancien schéma de features (voir "À faire")
- model_factory() de Rémi (src/stability.py) duplique la définition des pipelines logit/xgb déjà
  présente dans 01_data_models_v0.py — dette technique notée, factorisation prévue APRÈS le gel
  des modèles vendredi soir, pas avant
- TabICL absent de la stabilité de Rémi pour l'instant (normal, substitution décidée après son
  travail initial) — pour l'ajouter, il devra aussi passer par Colab/CUDA (même contrainte que
  nous, voir "TabICL — diagnostic du crash CPU et décision") : son bootstrap par session ne
  peut pas tourner en local sur TabICL

## À faire (pas encore réalisé, à ne pas oublier)
- [ ] Construire la matrice de coûts P&L et implémenter le calcul du profit total pour un seuil
      donné, à partir des prédictions des 3 modèles
- [ ] Optimiser le seuil de décision sur ce P&L en utilisant le GroupKFold sur train — jamais sur
      le test final
- [ ] Faire l'analyse de sensibilité à X et Y (le seuil optimal et la conclusion économique
      changent-ils si les hypothèses de coût varient ?)
- [ ] Tester income_A/income_B comme proxy sur d'autres variables démographiques (career_c,
      field_cd) avant de conclure qu'une seule mitigation suffit
- [ ] Entraîner une variante des modèles avec income neutralisé/retiré pour comparer la
      disparité avant/après (2e passe de feature engineering fairness)
- [x] Implémenter le retrait de shar1_1_A/B pour le logit uniquement — fait,
      `feature_dict["features_logit"]` dans `src/build_dataset.py`
- [x] Vérifier/corriger l'encodage des dummies catégorielles (field_cd/career_c/goal),
      drop_first=True — fait, vocabulaire fixe (`CAT_CODES`) dans `src/build_dataset.py`
- [x] VIF de contrôle sur les features finales du logit — fait, voir "Feature engineering" ;
      2 paires de colonnes à VIF infini documentées comme limite connue (petits effectifs),
      pas corrigées faute de temps
- [x] Débloquer TabICL et comparer les 3 modèles — fait via Colab/CUDA, voir "TabICL —
      diagnostic du crash CPU et décision" et le tableau dans "Validation croisée"
- [x] Réentraîner TabICL sur Colab avec le features.json post-correction dummies (158
      features au lieu de 164) — fait, résultat quasi identique (0.632/0.591±0.038)
- [ ] Prévenir Max : le logit a maintenant un features_logit distinct de features -- toute
      interprétation (SHAP/coefficients) du logit déjà commencée sur l'ancien schéma sera à
      refaire
- [ ] Lancer le P&L sur les 3 modèles (dépend de "Construire la matrice de coûts P&L" plus haut)


## Choix XGBoost vs Random Forest
- RF testé en contrôle (hyperparamètres par défaut, même split, mêmes features) : AUC=0.600
  vs XGBoost AUC=0.603 — écart de 0.003, dans le bruit (l'écart-type du GroupKFold est déjà
  de ±0.02-0.03)
- Confirme que le plafond ~0.60 reflète la difficulté intrinsèque du problème (prédire une
  alchimie à partir d'un profil pré-rencontre), pas un choix d'algorithme sous-optimal
- XGBoost retenu comme modèle ML officiel : gère nativement les NaN (pertinent vu le taux de
  missing sur income), permet d'exploiter income_missing_A/B plus finement que RF


## Stabilité (Rémi) : contrat de variables par modèle
- Le logit doit utiliser `contract.get("features_logit", contract["features"])` ; XGBoost
  conserve `features`. Cela s'applique au modèle de référence, à chaque réentraînement,
  aux prédictions et aux coefficients exportés. Le split par wave et seed=42 restent inchangés.
- D'après le message de Blanche, retirer `shar1_1_A` et `shar1_1_B` sert à lever la dépendance
  entre les six parts de préférences. Les coefficients sont donc à réinterpréter avec une
  catégorie de référence implicite ; ce retrait ne garantit pas à lui seul l'absence de toute
  autre colinéarité.
- Les résultats de stabilité déjà publiés sont historiques, antérieurs au nouveau contrat.
  Ils ne doivent pas servir de chiffres pour le logit corrigé. Relance en attente de la
  publication de `features_logit` (absent du main vérifié au commit 4cd5bfe).
- TabICL remplace TabPFN, conformément à la décision transmise par Blanche. Aucun résultat
  TabICL de stabilité n'est revendiqué avant disponibilité de sa configuration d'entraînement.
- Les dépendances de stabilité utilisent désormais les versions épinglées du groupe.
- Le test final reste hors réglage des variables, des hyperparamètres et du seuil. Les
  réévaluations de stabilité sont descriptives ; le test a déjà été consulté lors de la v0,
  il ne serait donc pas exact de le présenter comme totalement inédit à la soutenance.
