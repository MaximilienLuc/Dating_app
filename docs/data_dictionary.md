# Dictionnaire de données — HEC Match

Source : `data/Speed Dating Data Key.doc` (converti en texte via `textutil`). Ne documente que les
variables utilisées dans le pipeline (`data/features.json`, généré par `src/build_dataset.py`) —
pas l'ensemble des 195 colonnes brutes du CSV. Toutes les variables ci-dessous, sauf mention
contraire, viennent du questionnaire rempli **avant** le rendez-vous ("signup/Time1").

Convention : chaque variable existe en deux versions dans le dataset de paires, `<nom>_A` (le
décideur) et `<nom>_B` (le candidat évalué), sauf les variables de couple (déjà au niveau paire)
et les attributs protégés (voir plus bas).

## Cible

| Variable | Description | Échelle | Catégorie |
|---|---|---|---|
| `dec` | Décision de A à propos de B : "je voudrais revoir cette personne" — jugement porté **après** la rencontre réelle de 4 minutes, pas un like/dislike sur profil | binaire (0/1) | cible |

## Variables d'entrée (FEATURES)

### Démographie / profil

| Variable | Description | Échelle | Catégorie |
|---|---|---|---|
| `age` | Âge du participant | années | démographie |
| `date` | Fréquence de sorties amoureuses ("how often do you go on dates?") | 1 (plusieurs fois/semaine) à 7 (presque jamais) | démographie |
| `go_out` | Fréquence de sorties en général (pas forcément amoureuses) | 1 à 7, même échelle que `date` | démographie |
| `imprace` | Importance déclarée que le partenaire soit de la même origine | 1 (pas important) à 10 (très important) | démographie — **borderline** (proxy direct de `race`) |
| `imprelig` | Importance déclarée que le partenaire soit de la même religion | 1 à 10 | démographie |
| `exphappy` | Bonheur attendu vis-à-vis des rencontres de la soirée | 1 à 10 | démographie |
| `income` | Revenu médian du zip code déclaré (Census Bureau), **pas un revenu individuel**. Absent si le participant vient de l'étranger ou n'a pas renseigné son zip code | $ (continu) | démographie — **borderline** (proxy géographique de `race`, cf. `notebooks/00_eda_cleaning.ipynb` §7) |

### Catégorielles (one-hot encodées `<nom>_<A/B>_<code>`)

| Variable | Description | Codes | Catégorie |
|---|---|---|---|
| `field_cd` | Domaine d'études, codé | 1=Law, 2=Math, 3=Social Science/Psych, 4=Medical/Pharma/Bio Tech, 5=Engineering, 6=English/Creative Writing/Journalism, 7=History/Religion/Philosophy, 8=Business/Econ/Finance, 9=Education/Academia, 10=Biological Sciences/Chemistry/Physics, 11=Social Work, 12=Undergrad/undecided, 13=Political Science/Int'l Affairs, 14=Film, 15=Fine Arts, 16=Languages, 17=Architecture, 18=Other | démographie |
| `career_c` | Carrière visée, codée | 1=Lawyer, 2=Academic/Research, 3=Psychologist, 4=Doctor/Medicine, 5=Engineer, 6=Creative Arts/Entertainment, 7=Banking/Consulting/Finance/Business, 8=Real Estate, 9=Int'l/Humanitarian Affairs, 10=Undecided, 11=Social Work, 12=Speech Pathology, 13=Politics, 14=Pro sports/Athletics, 15=Other, 16=Journalism, 17=Architecture | démographie |
| `goal` | Objectif principal en participant à l'événement | 1=Soirée amusante, 2=Rencontrer des gens, 3=Obtenir un rendez-vous, 4=Relation sérieuse, 5=Pour dire que je l'ai fait, 6=Autre | démographie |

### Intérêts (17 variables, `<nom>_A`/`<nom>_B`)

`sports`, `tvsports`, `exercise`, `dining`, `museums`, `art`, `hiking`, `gaming`, `clubbing`,
`reading`, `tv`, `theater`, `movies`, `concerts`, `music`, `shopping`, `yoga` — intérêt déclaré
pour chaque activité, échelle 1 (pas du tout) à 10 (beaucoup). Catégorie : intérêts.

### Préférences déclarées — ce que je recherche (`attr1_1`...`shar1_1`)

| Variable | Description | Échelle | Catégorie |
|---|---|---|---|
| `attr1_1` | Importance de l'attractivité chez un partenaire | Waves 1-5,10-21 : 100 pts à répartir entre les 6 attributs (somme=100). Waves 6-9 : 1-10 indépendant par attribut (renormalisé en parts de 100 dans `load_and_clean()`, cf. `notebooks/00_eda_cleaning.ipynb` §4) | préférences déclarées |
| `sinc1_1` | Importance de la sincérité | idem | préférences déclarées |
| `intel1_1` | Importance de l'intelligence | idem | préférences déclarées |
| `fun1_1` | Importance du côté "fun" | idem | préférences déclarées |
| `amb1_1` | Importance de l'ambition | idem | préférences déclarées |
| `shar1_1` | Importance des centres d'intérêt partagés | idem | préférences déclarées |

### Auto-évaluation — comment je me note (`attr3_1`...`amb3_1`)

| Variable | Description | Échelle | Catégorie |
|---|---|---|---|
| `attr3_1` | Auto-évaluation de sa propre attractivité | 1 (awful) à 10 (great) | auto-évaluation |
| `sinc3_1` | Auto-évaluation de sa propre sincérité | 1 à 10 | auto-évaluation |
| `intel3_1` | Auto-évaluation de sa propre intelligence | 1 à 10 | auto-évaluation |
| `fun3_1` | Auto-évaluation de son propre côté "fun" | 1 à 10 | auto-évaluation |
| `amb3_1` | Auto-évaluation de sa propre ambition | 1 à 10 | auto-évaluation |

## Variables de couple / dérivées (feature engineering)

| Variable | Description | Échelle | Catégorie | Origine |
|---|---|---|---|---|
| `int_corr` | Corrélation entre les intérêts de A et B (calculée dans le CSV source) | -1 à 1 | dérivée | CSV brut |
| `age_diff` | `age_B - age_A` | années (signé) | dérivée | `load_and_clean()` |
| `abs_age_diff` | Valeur absolue de `age_diff` | années | dérivée | `load_and_clean()` |
| `fit_score` | Adéquation : Σ (poids recherché par A pour l'attribut / 100 × auto-note de B sur cet attribut), sur attr/sinc/intel/fun/amb | continu | dérivée | `load_and_clean()` |
| `income_missing_A` / `income_missing_B` | Indicateur binaire : `income` manquant pour ce participant (1) ou non (0). Créé suite à l'EDA — le manquant d'income (~47%, hors wave 13) est lié à des caractéristiques individuelles documentées (étranger / zip code non renseigné), potentiellement informatif | binaire (0/1) | dérivée — issue de l'EDA | `engineer_features()` |

## Attributs protégés (audit uniquement — jamais en entrée du modèle)

| Variable | Description | Codes |
|---|---|---|
| `gender_A` | Genre du décideur | 0=Femme, 1=Homme |
| `race_A` | Origine du décideur | 1=Black/African American, 2=European/Caucasian-American, 3=Latino/Hispanic American, 4=Asian/Pacific Islander/Asian-American, 5=Native American, 6=Other — **catégorie 5 (Native American) absente de ce dataset**, cf. `notebooks/00_eda_cleaning.ipynb` §8 |
| `cand_race` | Origine du candidat B (= `race_B`) | mêmes codes que `race_A` |
| `cand_female` | Le candidat B est une femme | binaire (0/1), dérivé de `gender_B` |
| `same_race` | A et B sont de la même origine | binaire (0/1), recalculé à partir de `race_A`/`race_B` (indépendamment de la colonne `samerace` du CSV brut) |

## Écarts CSV / Key constatés

En convertissant `Speed Dating Data Key.doc` et en le confrontant aux colonnes réelles du CSV :

1. **`int3_1` (Key) vs `intel3_1` (CSV réel)** : le Key documente la variable d'auto-évaluation de l'intelligence sous le nom `int3_1`, mais la colonne réelle du CSV est `intel3_1`. Coquille dans le document, pas dans les données.
2. **`excersice` (Key, coquille) vs `exercise` (CSV)** : le Key liste l'intérêt "Body building/exercising" sous le nom mal orthographié `excersice` ; la colonne réelle est `exercise`.
3. **`go out` (Key, avec espace) vs `go_out` (CSV)** : simple différence de formatage (espace vs underscore), pas une vraie divergence.
4. **`samerace` (CSV brut) vs `same_race` (notre variable)** : le CSV source contient déjà une colonne `samerace` précalculée pour le couple (A, partenaire du soir). Notre `same_race` est recalculée indépendamment à partir de `race_A`/`race_B` fusionnés — les deux devraient coïncider mais n'ont pas été vérifiées comme identiques (non testé dans l'EDA actuelle).
5. **Mécanisme du manquant d'`income`, documenté uniquement dans le Key** (absent du CSV lui-même) : *"When there is no income it means that they are either from abroad or did not enter their zip code."* — information clé pour justifier `income_missing_A`/`income_missing_B` (voir `notebooks/00_eda_cleaning.ipynb` §6), à ne pas perdre si quelqu'un retravaille sur ce point sans avoir consulté le `.doc` original.
