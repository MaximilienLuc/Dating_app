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

## Validation croisée
- GroupKFold 5 folds sur waves : AUC stable (logit 0.600 ± 0.023, xgb 0.596 ± 0.026), cohérent
  avec le split unique initial — confirme que l'AUC modeste est réelle, pas un artefact du split

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

## Économie / P&L
- Matrice de coûts : profil montré + oui = +X, profil montré + non = −Y, match manqué = revenu
  perdu
- Seuil de décision à optimiser sur ce P&L plutôt que sur l'AUC seule, avec analyse de
  sensibilité à X et Y — répond à l'exigence officielle du brief de performance "économique" en
  plus de "statistique"

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


## Choix XGBoost vs Random Forest
- RF testé en contrôle (hyperparamètres par défaut, même split, mêmes features) : AUC=0.600
  vs XGBoost AUC=0.603 — écart de 0.003, dans le bruit (l'écart-type du GroupKFold est déjà
  de ±0.02-0.03)
- Confirme que le plafond ~0.60 reflète la difficulté intrinsèque du problème (prédire une
  alchimie à partir d'un profil pré-rencontre), pas un choix d'algorithme sous-optimal
- XGBoost retenu comme modèle ML officiel : gère nativement les NaN (pertinent vu le taux de
  missing sur income), permet d'exploiter income_missing_A/B plus finement que RF
