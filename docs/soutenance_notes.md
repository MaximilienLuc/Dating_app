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
- Décision actée pour le logit : retrait de shar1_1_A et shar1_1_B des features du logit
  uniquement (XGBoost et TabICL gardent les 6 préférences), pour casser la colinéarité parfaite
  créée par la contrainte de somme à 100 (confirmée structurelle sur toutes les waves, pas
  seulement 6-9 — voir "Dérive de protocole entre waves"). **Décision actée, pas encore
  implémentée dans le code** : features.json n'a pas encore de clé features_logit séparée,
  build_dataset.py n'a pas encore été modifié (voir "À faire").
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

## Coordination équipe / dette technique
- Merge avec le travail de stabilité de Rémi fait le 24/09 — split.json identique bit à bit des
  deux côtés (mêmes train_waves/test_waves), aucun recalcul nécessaire sur ce point
- features.json aura une clé features_logit séparée de features (retrait shar1_1) une fois
  l'implémentation faite côté nous (pas encore fait, voir "Feature engineering" et "À faire") —
  Rémi à prévenir à ce moment-là : son model_factory() (src/stability.py) lit un seul
  contract['features'] identique pour logit ET xgb, donc il devra explicitement lire
  features_logit pour le logit et relancer sa stabilité pour qu'elle reflète le retrait de
  shar1_1
- model_factory() de Rémi (src/stability.py) duplique la définition des pipelines logit/xgb déjà
  présente dans 01_data_models_v0.py — dette technique notée, factorisation prévue APRÈS le gel
  des modèles vendredi soir, pas avant
- TabICL absent de la stabilité de Rémi pour l'instant (normal, substitution décidée après son
  travail initial) — à ajouter une fois débloqué de notre côté (voir "À faire")

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
- [ ] Implémenter le retrait de shar1_1_A/B pour le logit uniquement dans build_dataset.py +
      features.json (décision actée, code pas encore fait)
- [ ] Vérifier/corriger l'encodage des dummies catégorielles (field_cd/career_c/goal) pour
      confirmer drop_first=True — pas encore vérifié dans le code actuel
- [ ] VIF de contrôle sur les features finales du logit (résultat en attente, bloqué derrière
      les deux points ci-dessus)
- [ ] Débloquer TabICL : crash reproductible (segfault) sur nos données réelles même réduites à
      quelques colonnes/lignes, cause pas encore identifiée — nécessaire avant de pouvoir
      comparer les 3 modèles et lancer le P&L sur les 3


## Choix XGBoost vs Random Forest
- RF testé en contrôle (hyperparamètres par défaut, même split, mêmes features) : AUC=0.600
  vs XGBoost AUC=0.603 — écart de 0.003, dans le bruit (l'écart-type du GroupKFold est déjà
  de ±0.02-0.03)
- Confirme que le plafond ~0.60 reflète la difficulté intrinsèque du problème (prédire une
  alchimie à partir d'un profil pré-rencontre), pas un choix d'algorithme sous-optimal
- XGBoost retenu comme modèle ML officiel : gère nativement les NaN (pertinent vu le taux de
  missing sur income), permet d'exploiter income_missing_A/B plus finement que RF
