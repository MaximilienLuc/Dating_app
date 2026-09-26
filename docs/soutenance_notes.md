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

## Performance prédictive (statistique + économique) — FAIT, sur les 3 modèles

Implémenté dans `src/performance.py` (branche `blanche-performance`), notebook narratif
`notebooks/05_performance.ipynb`, résultats dans `reports/performance/`. Portée : PR-AUC,
calibration, matrice de confusion, XPER (décomposition de l'AUC par feature), optimisation du
seuil P&L, test de robustesse léger. **Pas de Brier score ni de log-loss** (hors cours ISAF,
écarté volontairement).

**Matrice de coûts** : profil montré + oui = +X, profil montré + non = −Y. Hypothèses dans
`data/economic_assumptions.json` (jamais en dur dans le code) : **X=2€, Y=0.5€** (ratio 4:1) —
hypothèses narratives assumées, pas mesurées, à affiner.

**Simplification assumée** : le P&L est calculé au niveau de la décision individuelle (`dec`),
pas du vrai match mutuel (`match`) qui nécessiterait de croiser deux décisions d'une paire —
limite documentée dans `src/performance.py`, pas cachée. "Match manqué" → "occasion de oui
manquée" au niveau individuel, pas de coût d'opportunité modélisé pour l'instant.

**Seuil optimisé par grille (0.05-0.95) sur GroupKFold (train, 5 folds sur wave)**, appliqué une
seule fois sur test :

| Modèle | PR-AUC | Seuil optimal | P&L test @ seuil optimal | P&L test @ 0.5 | Top features XPER |
|---|---|---|---|---|---|
| Logit | 0.504 | 0.05 | 1042 € | 516.5 € | career_c_A_4, fit_score, exphappy_A, amb1_1_B, attr3_1_B |
| XGBoost | 0.506 | 0.10 | 1051 € | 507.5 € | sports_B, career_c_B_16, field_cd_B_NA, goal_A_2, imprelig_A |
| TabICL | 0.545 | 0.30 | 947 € | 498 € | non calculé en local (voir ci-dessous) |

- Seuils optimaux nettement < 0.5 pour les 3 modèles — cohérent avec le ratio de coûts 4:1 qui
  favorise la recommandation (un faux positif ne coûte que 0.5€, un vrai positif rapporte 2€).
  Optimiser le seuil ~double le P&L test par rapport au seuil naïf 0.5.
- **`income_A`/`income_B` n'apparaissent dans le top 10 XPER d'AUCUN des deux modèles**
  (logit, xgb) — pas de lien direct performance/proxy fairness détecté sur cette lecture,
  malgré le statut "borderline" de ces variables (à croiser avec l'audit TOST de Max).

**XPER (Sinclair et al.)** : `pip install XPER`, `ModelPerformance(...).calculate_XPER_values(["AUC"])`.
Nécessite le vrai objet modèle (appelé sur des centaines de coalitions de features masquées) —
pas juste des prédictions figées. Calculé pour logit et XGBoost en local avec des paramètres
réduits (`N_coalition_sampled=500` au lieu du défaut ~2360 à 156-158 features ; défaut testé et
abandonné après >10 min sans terminer). **XGBoost a pris 12h31 au total** (dont l'écrasante
majorité en veille système — la machine s'est mise en veille pendant l'exécution, ce qui gèle le
process ; temps CPU réel ~2h14, soit ~15-20 min si la machine était restée éveillée. Utiliser
`caffeinate -w <PID>` pour les prochains calculs longs, sur cette machine).
**TabICL non calculable en local** (modèle non chargeable, cf. section TabICL) — tenté sur Colab
via `colab/xper_tabicl.py` avec paramètres encore réduits (300 coalitions, échantillon 100) ;
résultat non garanti dans un temps raisonnable, à documenter tel quel si non concluant.

**Test de robustesse** (2 scénarios alternatifs, `data/economic_assumptions.json` réécrit
temporairement puis restauré — jamais de valeur X/Y en dur) :

| Scénario | Logit (seuil / P&L) | XGBoost (seuil / P&L) | TabICL (seuil / P&L) | Classement |
|---|---|---|---|---|
| X=2/Y=0.5 (base) | 0.05 / 1042€ | 0.10 / 1051€ | 0.30 / 947€ | xgb > logit > tabicl |
| X=1/Y=1 (égal) | 0.60 / 28€ | 0.60 / 17€ | 0.45 / 55€ | **tabicl > logit > xgb** |
| X=3/Y=0.3 (10:1) | 0.05 / 2029€ | 0.05 / 2036€ | 0.25 / 1910€ | xgb > logit > tabicl |

- **Le classement des modèles n'est PAS stable** : à coûts égaux (X=1/Y=1), TabICL passe premier
  — inversion complète par rapport au scénario de base et au scénario généreux. À ne pas
  présenter comme "XGBoost est le meilleur modèle" sans préciser sous quelle hypothèse
  économique.
- Le seuil optimal, lui, est globalement stable en ordre de grandeur **sauf** au scénario à
  coûts égaux, où il saute à ~0.5-0.6 pour tous les modèles (cohérent : sans asymétrie de coût,
  optimal ≈ le seuil qui maximise l'accuracy plutôt qu'un seuil bas favorisant le recall).

**Réservé au(x) finaliste(s)** (pas fait maintenant, décision explicite pour tenir le
calendrier) : analyse de sensibilité complète (grille fine sur X/Y, pas 2 scénarios ponctuels),
et XPER appliqué directement au P&L plutôt qu'à l'AUC seule.

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

## Audit de Fairness : test d'équivalence TOST (Schuirmann, 1987) sur les origines
- **Principe d'IA de confiance** : le test de deux t-tests unilatéraux (TOST) renverse la charge
  de la preuve. Contrairement à un t-test classique qui postule l'équité par défaut ($p_1 = p_2$),
  l'approche par équivalence pose le modèle **non équivalent / biaisé par défaut** et exige de prouver
  statistiquement que l'écart d'exposition reste confiné dans une marge de tolérance $\delta = 10\text{ pp}$
  ($\alpha = 0.05$, soit un intervalle de confiance conjoint à $90\%$).
- **Protocole d'audit pairwise** : groupe de référence = Caucasiens ($n=1\,109$ sur le set de test),
  comparé individuellement à chaque minorité présente : Asiatiques ($n=391$), Latinos ($n=135$),
  Noirs ($n=130$), Autres ($n=56$).
- **Résultats au seuil de tolérance $\delta = 10\text{ pp}$** (seuil de décision = 0.5) :
  - **Logit** (taux moyen Caucasiens = 36.8%) :
    - vs Asiatiques : écart = **+10.2 pp** (IC 90% : [+5.8, +14.6]) — **Échec** (sous-exposition nette des Asiatiques)
    - vs Latinos : écart = **−18.8 pp** (IC 90% : [−26.2, −11.3]) — **Échec** (surexposition des candidats Latinos)
    - vs Noirs : écart = **−3.2 pp** (IC 90% : [−10.7, +4.3]) — **Échec** (l'écart estimé est faible mais l'IC déborde des 10 pp par manque de puissance statistique, $n=130$)
    - vs Autres : écart = **−4.3 pp** (IC 90% : [−15.4, +6.8]) — **Échec** ($n=56$, variance trop forte)
    - *Équivalence globale raciale : Rejetée (0 / 4 tests validés).*
  - **XGBoost** (taux moyen Caucasiens = 36.7%) :
    - vs Noirs : écart = **+2.1 pp** (IC 90% : [−5.2, +9.4]) — **Validé (Fair)** (IC strictement inclus dans [−10 pp, +10 pp])
    - vs Asiatiques : écart = **+11.1 pp** (IC 90% : [+6.8, +15.5]) — **Échec** (sous-exposition nette des Asiatiques)
    - vs Latinos : écart = **−10.7 pp** (IC 90% : [−18.2, −3.2]) — **Échec** (surexposition des Latinos)
    - vs Autres : écart = **+9.9 pp** (IC 90% : [−0.1, +19.9]) — **Échec**
    - *Équivalence globale raciale : Rejetée (1 / 4 tests validés).*
- **Points clés pour la soutenance (récit Blanquette / Max)** :
  1. *Démonstration concrète du renversement de la charge de la preuve* : pour Logit vs Noirs (−3.2 pp) et
     Logit vs Autres (−4.3 pp), l'écart brut observé est inférieur à 10 pp, mais le test TOST échoue quand même.
     Ce résultat illustre parfaitement aux jurys qu'un faible écart apparent ne suffit pas : sans effectif suffisant
     pour resserrer l'intervalle de confiance à 90%, on ne peut garantir l'absence de biais.
  2. *Reproduction des biais sociologiques historiques sans variable protégée* : bien que la variable d'origine
     ne soit jamais entrée dans les modèles, les deux modèles sous-exposent systématiquement les candidats asiatiques
     d'environ 10 à 11 pp par rapport aux caucasiens, captant des proxys d'intérêts ou de préférences déclarées.
  3. *Graphiques du rapport générés* : `reports/fairness/tost_intervals.png` (intervalles de confiance TOST vs zone
     d'équivalence [−10 pp, +10 pp]) et `reports/fairness/racial_exposure_rates.png` (taux d'exposition réels vs prédits).

## Mitigation du biais : suppression de proxys via FPDP (Option A)
- **Principe méthodologique (Pérignon & Saurin)** : pré-traitement rigoureux où une variable candidate
  $X_A$ est identifiée si elle est à l'origine du rejet de l'hypothèse nulle d'équité. Par balayage
  FPDP (Fairness Partial Dependence Plot) sur `X_train`, on force chaque variable à une valeur constante
  $c_k$ (déciles pour les variables continues, valeurs uniques pour les discrètes/indicatrices) et on
  évalue la métrique de fairness.
- **4 variables candidates identifiées et supprimées (`PROXY_FEATURES_TO_DROP`)** :
  1. `attr3_1_B` (auto-évaluation d'attractivité déclarée par le candidat B) : neutraliser cette variable
     sur le train fait chuter l'écart Caucasiens vs Asiatiques de $+12.65\%$ à $+7.94\%$ et valide
     simultanément l'ensemble des tests d'équivalence raciale sur le train.
  2. `career_c_B_12` (code carrière spécifique du candidat B) : neutralise la disparité sur le groupe Autres.
  3. `go_out_B` (fréquence de sorties du candidat B) : neutralise la disparité sur le groupe Autres.
  4. `intel3_1_B` (auto-évaluation d'intelligence du candidat B) : neutralise la disparité sur le groupe Autres.
- **Modèles ré-entraînés sur le sous-ensemble épuré (160 variables au lieu de 164)** :
  - `models/xgb_mitigated.joblib` et `models/logit_mitigated.joblib`
  - Métadonnées et contrats sauvegardés : `data/mitigated_features.json` et `data/mitigation_metrics.json`
- **Bilan de l'arbitrage Utilité-Fairness (Trade-off) sur le test set** :
  - **Sur Logit** : l'écart Caucasiens vs Asiatiques passe de **+10.19 pp** à **+8.20 pp** (rentre sous la
    barre des 10 pp de tolérance !), au prix d'un coût d'utilité quasi nul ($\Delta\text{AUC} = −0.007$,
    $0.587 \rightarrow 0.581$).
  - **Sur XGBoost** : l'écart Caucasiens vs Autres est presque totalement effacé (**+9.91 pp $\rightarrow$ −0.66 pp**),
    l'écart Caucasiens vs Noirs passe de $+2.08\text{ pp}$ à $+0.20\text{ pp}$, et l'écart vs Asiatiques est réduit
    de $+11.12\text{ pp}$ à $+10.77\text{ pp}$.
  - **Préservation de la performance économique et statistique** : l'AUC test de XGBoost est intacte
    ($0.604 \rightarrow 0.612$), et le profit total au seuil P&L optimal calibré sur train ($t^*=0.36$) reste
    positif ($+286\text{ €} \rightarrow +290\text{ €}$).
  - *Graphique synthétique exporté* : `reports/fairness/mitigation_tradeoff.png` (évolution des écarts raciaux
    et comparaison d'AUC avant/après).

## Décomposition biais sociétal vs biais algorithmique (3 modèles incl. TabICL)

**Contexte** : les tests TOST échouent pour les groupes Asian et Latino quel que soit le seuil de
décision ou la mitigation. La question posée en soutenance sera inévitablement : "vos modèles
sont-ils racistes ?". La réponse correcte distingue deux couches de biais, qu'on a maintenant
mesurées précisément (`src/fairness_bias_decomposition.py`, graphique `reports/fairness/bias_decomposition.png`).

### Définitions formelles

Soit $p_{\text{cauc}}^{\text{données}}$ le taux de oui réel dans $y_{\text{test}}$ pour les Caucasiens, et
$p_g^{\text{données}}$ le même taux pour le groupe $g$. Soit $\hat{p}_{\text{cauc}}$ et $\hat{p}_g$
les taux d'exposition prédits par le modèle au seuil $t$.

$$\text{biais\_sociétal}(g) = p_{\text{cauc}}^{\text{données}} - p_g^{\text{données}}$$
$$\text{biais\_total}(g) = \hat{p}_{\text{cauc}} - \hat{p}_g$$
$$\text{biais\_algorithmique}(g) = \text{biais\_total}(g) - \text{biais\_sociétal}(g)$$

Le biais algorithmique est ce que le **modèle ajoute** au-delà du biais présent dans les données
brutes. Un modèle parfaitement neutre aurait un biais algorithmique = 0 pour tous les groupes.

### Niveau 1 — Biais sociétal (données brutes Speed Dating, test set)

Taux de "oui" réels dans les données, **avant tout modèle** :

| Groupe | Taux de oui | Écart vs Caucasiens | n |
|---|---|---|---|
| **Caucasiens** | **43.9%** | — | 1109 |
| Latinos | 50.4% | **−6.5 pp** (sur-aimés) | 135 |
| Autres | 48.2% | −4.3 pp | 56 |
| Noirs | 46.2% | −2.2 pp | 130 |
| Asiatiques | 35.3% | **+8.6 pp** (sous-aimés) | 391 |

Ces écarts sont documentés dans la littérature (Fisman & Iyengar, 2006 ; Hitsch, Hortaçsu &
Ariely, 2010) et reflètent des préférences réelles des participants du dataset. **L'algorithme
ne les invente pas.** En particulier, la sous-représentation des Asiatiques (+8.6 pp de gap
dans les données brutes) explique structurellement pourquoi tous les modèles exhibent un gap
positif vs Asiatiques : ils *apprennent* cette préférence humaine via les proxys.

### Niveau 2 — Biais algorithmique pur (à t = 0.50)

Ce que chaque modèle *ajoute* au-delà du biais sociétal :

| Modèle | vs Noirs | vs Latinos | vs Asiatiques | vs Autres |
|---|---|---|---|---|
| **XGBoost** | +4.2 pp | −2.2 pp | +2.1 pp | +0.8 pp |
| **Logit** | −1.2 pp | **−12.5 pp** | +1.4 pp | −0.2 pp |
| **TabICL** | +1.5 pp | −7.9 pp | +1.7 pp | +9.1 pp |

**Lecture** : un biais algorithmique proche de 0 signifie que le modèle ne discrimine pas
au-delà de ce que font les humains dans les données. Un biais négatif vs Latino pour le Logit
(−12.5 pp) signifie que le Logit *amplifie fortement* la discrimination envers les Latinos
au-delà du biais sociétal. XGBoost, lui, est quasi neutre algorithmiquement (biais ≤ 4 pp
sur tous les groupes). TabICL amplifie la discrimination envers les "Autres" (+9.1 pp).

**Conclusion pour la soutenance** :
- **XGBoost** est le modèle le plus equitable algorithmiquement. Ses échecs au TOST (Latino,
  Asian) viennent presque entièrement du biais sociétal et non d'un biais ajouté. Il passe
  même le TOST pour les Noirs (biais sociétal faible de −2.2 pp → facile à contenir).
- **Logit** amplifie massivement la discrimination envers les Latinos (−12.5 pp purement
  algorithmiques) : comportement non défendable pour un déploiement, indépendamment du TOST.
- **TabICL** est intermédiaire mais présente une anomalie sur le groupe "Autres" (+9.1 pp),
  à documenter comme limite de la généralisation du modèle sur les petits effectifs.
- *Script* : `src/fairness_bias_decomposition.py` | *Données* : `data/fairness_bias_decomposition.json`
- *Graphique* : `reports/fairness/bias_decomposition.png` (Panel A = biais total sociétal+algo,
  Panel B = biais algorithmique pur isolé)

### Résultats TOST complets — TabICL (t = 0.50, δ = 10 pp)

| Comparaison | Gap total | Biais algo pur | TOST |
|---|---|---|---|
| vs Noirs | −0.7 pp | +1.5 pp | ✅ Fair |
| vs Latinos | −14.4 pp | −7.9 pp | ❌ Unfair |
| vs Asiatiques | +10.3 pp | +1.7 pp | ❌ Unfair |
| vs Autres | +4.8 pp | +9.1 pp | ❌ Unfair |

TabICL passe le test vs Noirs (biais sociétal faible + biais algo faible = gap total dans
la zone d'équivalence). Il échoue sur Latino (principalement biais algorithmique amplifié)
et sur Autres (biais sociétal faible mais biais algorithmique fort, probablement lié au
très petit effectif $n=56$). Pas de modèle sauvegardé TabICL en local (contrainte CUDA) —
l'évaluation s'appuie sur `data/tabicl_predictions.parquet` (inférence GPU Colab).

## À faire (pas encore réalisé, à ne pas oublier)
- [x] Construire la matrice de coûts P&L et implémenter le calcul du profit total pour un seuil
      donné, à partir des prédictions des 3 modèles — fait, `calculate_pnl()` dans
      `src/performance.py`, voir "Performance prédictive"
- [x] Optimiser le seuil de décision sur ce P&L en utilisant le GroupKFold sur train — jamais sur
      le test final — fait, `optimize_threshold_groupkfold()`
- [x] Faire l'analyse de sensibilité à X et Y (le seuil optimal et la conclusion économique
      changent-ils si les hypothèses de coût varient ?) — fait (2 scénarios légers) : le
      classement des modèles change (TabICL premier à X=1/Y=1), pas stable — voir "Performance
      prédictive". Analyse complète (grille fine) réservée au(x) finaliste(s).
- [ ] Tester income_A/income_B comme proxy sur d'autres variables démographiques (career_c,
      field_cd) avant de conclure qu'une seule mitigation suffit
- [ ] Entraîner une variante des modèles avec income neutralisé/retiré pour comparer la
      disparité avant/après (2e passe de feature engineering fairness)
- [ ] Tenter XPER sur TabICL sur Colab (`colab/xper_tabicl.py`, paramètres réduits, résultat non
      garanti) — XPER logit/xgb fait en local, income_A/B absent du top 10 des deux
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
- [x] Entraîner une variante des modèles avec proxys neutralisés/retirés pour comparer la
      disparité avant/après (2e passe de feature engineering fairness via FPDP) — fait
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

## Stabilité : intégration des prédictions TabICL (Rémi)
- Source d'intégration : branche `blanche`, commit `5a3b1b3`. `features_logit` est bien
  disponible sur cette branche ; le constat d'absence sur main ci-dessus était daté.
- Trois modèles évalués côté test, deux réentraînés côté train. TabICL est joint par
  `(iid, pid, wave)` avec contrôle d'unicité, de couverture et de probabilités valides.
  Aucun chargement de son modèle CUDA et aucun réentraînement TabICL ne sont effectués.
- Bootstrap par sessions avec remplacement, pas GroupKFold. Les tirages test sont les mêmes
  pour les trois modèles. TabICL : incertitude de performance uniquement, aucune conclusion
  sur les changements de décisions ou d'importance après réentraînement.
- Ajout de la distance euclidienne, en complément de la distance cosinus : coefficients
  logit exprimés dans une unité commune (écart-type du train original), gains normalisés
  pour XGBoost. Les distances des deux familles ne se comparent pas directement.
- Le logit existant utilise déjà la pénalité L2 par défaut de scikit-learn (C=1).
  Une comparaison Elastic Net ou un autre niveau de régularisation serait une expérience
  supplémentaire à sélectionner en validation interne par wave, pas sur le test final.
- Aucun arbitrage performance/stabilité de XGBoost n'est démontré par la seule variance
  des importances. Il faudrait comparer des configurations prédéfinies en validation interne.
- Le déplacement du seuil optimal P&L est une extension différée : coûts et seuil métier
  ne sont pas encore fixés. Le taux de bascule actuel utilise le seuil descriptif 0,5.


### Résultats de stabilité actualisés
- Run cloud `36034895577`, code `64b37a5`, données de Blanche `5a3b1b3` :
  200 réentraînements par modèle (logit/XGBoost), 2 000 bootstraps test appariés (les trois).
- AUC : logit 0,589 ; XGBoost 0,614 ; TabICL 0,632. Les trois intervalles de différence
  appariée incluent zéro. Ne pas annoncer de supériorité statistique démontrée.
- Décisions qui basculent après réentraînement : 16,7% logit, 21,4% XGBoost en moyenne
  au seuil fixe 0,5. Ce ne sont pas des taux d'erreur. Non mesuré pour TabICL.
- Ces exports remplacent les anciens résultats gelés ; les remarques d'attente ci-dessus
  décrivent l'état antérieur de main. L'intégration utilise bien `features_logit` (156 variables)
  et `features` (158), avec le split original et seed=42.
