"""
HEC Match — construction du dataset.

Distingue volontairement deux étapes :
  - load_and_clean()   : nettoyage brut du CSV (dtypes, dédoublonnage, correction de
                          dérive de protocole entre waves) -> dataframe de paires (A, B)
                          + dict de métadonnées features (target/features/protected/...).
  - engineer_features() : ajoute les variables dérivées identifiées pendant l'EDA
                          (notebooks/00_eda_cleaning.ipynb), au-dessus du dataframe nettoyé.

Utilisé à la fois par 01_data_models_v0.py et par notebooks/00_eda_cleaning.ipynb, pour
ne jamais dupliquer cette logique entre les deux.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming",
             "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
PREFS = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]  # ce que je recherche
SELF = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]              # comment je me note
OTHER = ["age", "field_cd", "career_c", "goal", "date", "go_out", "imprace", "imprelig",
         "exphappy", "income"]
PROTECTED_RAW = ["gender", "race"]  # gender : 0 = femme, 1 = homme
CAT = ["field_cd", "career_c", "goal"]

# Codes fixes (cf. docs/data_dictionary.md / Speed Dating Data Key.doc), PAS déduits des valeurs
# observées dans les données : garantit des colonnes dummies identiques quel que soit le
# sous-échantillon (train/test), et un encodage sans fuite (le vocabulaire est une connaissance
# du domaine, pas une statistique calculée sur les données).
CAT_CODES = {
    "field_cd": [str(i) for i in range(1, 19)],   # 1=Law ... 18=Other
    "career_c": [str(i) for i in range(1, 18)],   # 1=Lawyer ... 17=Architecture
    "goal": [str(i) for i in range(1, 7)],         # 1=Fun night out ... 6=Other
}

EXCLUDED_RULE = ("Toute variable notée pendant ou après la soirée (attr, sinc, intel, fun, amb, "
                  "shar, like, prob, met, *_o, match_es, *_s, *_2, *_3, dec_o, match) + ordre du "
                  "rendez-vous (round, position, order)")


def load_and_clean(csv_path=None):
    """Charge "Speed Dating Data.csv" et retourne le dataframe de paires NETTOYÉ
    (une ligne = un participant A qui décide à propos d'un candidat B) ainsi qu'un
    dict de métadonnées features (target, features de base, attributs protégés,
    variables limites, règle d'exclusion).

    "Nettoyage" ici = corriger des problèmes de la donnée brute elle-même, pas créer
    de nouvelles variables prédictives :
      - conversion income en numérique
      - renormalisation des préférences waves 6-9 (échelle 1-10 -> 100 points, pour
        que toutes les waves soient sur la même échelle avant modélisation)
      - fusion des profils A/B, encodage one-hot des catégorielles

    Les variables réellement "engineered" à partir des constats de l'EDA (ex.
    income_missing_A/B) sont ajoutées séparément par engineer_features().

    Encodage des catégorielles (field_cd/career_c/goal) : one-hot avec drop_first=True sur un
    vocabulaire FIXE (CAT_CODES, pas les valeurs observées) -- la catégorie "1" de chaque
    variable est la référence implicite (son effet est absorbé dans l'intercept), "NA" reste une
    catégorie explicite (non droppée) pour garder le signal "non renseigné" visible.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "Speed Dating Data.csv"
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    # ------------------------------------------------------------------
    # Profil PRÉ-rendez-vous de chaque participant (questionnaire d'inscription,
    # suffixe _1). Tout ce qui est noté pendant/après la soirée est EXCLU (fuite).
    # ------------------------------------------------------------------
    prof = df[["iid"] + OTHER + INTERESTS + PREFS + SELF + PROTECTED_RAW].drop_duplicates("iid").copy()
    prof["income"] = pd.to_numeric(prof["income"].astype(str).str.replace(",", ""), errors="coerce")

    # Vagues 6-9 : préférences notées de 1 à 10 au lieu de 100 points à répartir
    # -> renormalisation en parts de 100 (nettoyage : dérive de protocole connue)
    s = prof[PREFS].sum(axis=1).replace(0, np.nan)
    prof[PREFS] = prof[PREFS].div(s, axis=0) * 100

    # ------------------------------------------------------------------
    # Paires (A décide, B est le candidat)
    # ------------------------------------------------------------------
    pairs = df[["iid", "pid", "wave", "dec", "match", "int_corr"]].dropna(subset=["pid"]).copy()
    pairs["pid"] = pairs["pid"].astype(int)

    A = prof.add_suffix("_A").rename(columns={"iid_A": "iid"})
    B = prof.add_suffix("_B").rename(columns={"iid_B": "pid"})
    data = pairs.merge(A, on="iid", how="left").merge(B, on="pid", how="left")

    # Variables de couple
    data["age_diff"] = data["age_B"] - data["age_A"]
    data["abs_age_diff"] = data["age_diff"].abs()
    # Adéquation : ce que A recherche (poids) x comment B se décrit
    data["fit_score"] = sum(data[f"{a}1_1_A"] / 100 * data[f"{a}3_1_B"]
                            for a in ["attr", "sinc", "intel", "fun", "amb"])

    # Attributs protégés : gardés pour l'AUDIT, jamais en entrée des modèles
    data["cand_female"] = (data["gender_B"] == 0).astype(int)
    data["cand_race"] = data["race_B"]  # 1 Black, 2 White, 3 Latino, 4 Asian, 5 Native Am., 6 Other
    data["same_race"] = (data["race_A"] == data["race_B"]).astype(int)
    protected = ["cand_female", "cand_race", "same_race", "gender_A", "race_A"]

    # ------------------------------------------------------------------
    # Liste des variables d'entrée
    # ------------------------------------------------------------------
    num_base = [c for c in OTHER if c not in CAT] + INTERESTS + PREFS + SELF
    cat_cols = [f"{c}_{side}" for c in CAT for side in "AB"]

    dummy_frames = []
    for col in cat_cols:
        base = col[:-2]  # "field_cd_A" -> "field_cd"
        categories = CAT_CODES[base] + ["NA"]
        values = data[col].astype("Int64").astype(str).replace("<NA>", "NA")
        cat_series = pd.Series(pd.Categorical(values, categories=categories), index=data.index)
        dummy_frames.append(pd.get_dummies(cat_series, prefix=col, drop_first=True).astype(int))
    dummies = pd.concat(dummy_frames, axis=1)
    data = pd.concat([data, dummies], axis=1)

    features = ([f"{c}_A" for c in num_base] + [f"{c}_B" for c in num_base]
                + ["int_corr", "age_diff", "abs_age_diff", "fit_score"] + list(dummies.columns))

    borderline = ["imprace_A", "imprace_B"]
    borderline_reasons = {
        "imprace_A": "importance déclarée de l'origine (auto-évaluée) : proxy direct de race_A",
        "imprace_B": "importance déclarée de l'origine (auto-évaluée) : proxy direct de race_B",
    }

    feature_dict = {
        "target": "dec",
        "features": features,
        "protected": protected,
        "borderline": borderline,
        "borderline_reasons": borderline_reasons,
        "excluded_rule": EXCLUDED_RULE,
    }
    return data, feature_dict


def engineer_features(df, feature_dict):
    """Ajoute les variables dérivées identifiées pendant l'EDA
    (notebooks/00_eda_cleaning.ipynb, section 6) au-dessus du dataframe nettoyé.

    income_missing_A / income_missing_B : indicateur binaire de non-réponse au
    revenu. income_A/B manque pour ~47% des lignes (hors wave 13, qui est un
    blackout total de la question pour cette session -- non renormalisable, à
    la différence de la dérive d'échelle des waves 6-9). Le dictionnaire de
    données précise que income est absent quand le participant vient de
    l'étranger ou n'a pas renseigné son zip code : ce n'est donc pas un pur
    hasard (ni un problème de collecte par wave, cf. EDA section 6), et le signal
    "a répondu ou non" peut être informatif en plus de la valeur imputée.

    features_logit : retrait de shar1_1_A/shar1_1_B des features du LOGIT UNIQUEMENT (les 6
    autres modèles -- XGBoost, TabICL -- gardent les 6 préférences complètes, cf.
    feature_dict["features"]). Les 6 préférences (attr1_1+sinc1_1+intel1_1+fun1_1+amb1_1+
    shar1_1) somment à 100 par construction (contrainte structurelle confirmée empiriquement sur
    toutes les waves, cf. docs/soutenance_notes.md "Dérive de protocole entre waves") : elles
    sont donc parfaitement colinéaires, ce qui est problématique pour un modèle linéaire (logit)
    mais pas pour XGBoost/TabICL. Retirer shar1_1 est analogue à drop_first=True sur un
    encodage catégoriel : shar1_1 devient la référence implicite, son effet se lisant en négatif
    des 5 coefficients restants. Alternative plus rigoureuse envisagée (transform log-ratio sur
    données compositionnelles) puis écartée pour rester dans le calendrier du projet.

    Retourne (df, feature_dict) mis à jour -- ne mute pas les objets passés en argument.
    """
    df = df.copy()
    df["income_missing_A"] = df["income_A"].isna().astype(int)
    df["income_missing_B"] = df["income_B"].isna().astype(int)

    feature_dict = dict(feature_dict)
    feature_dict["features"] = list(feature_dict["features"]) + ["income_missing_A", "income_missing_B"]
    feature_dict["borderline"] = list(feature_dict["borderline"]) + ["income_A", "income_B"]
    feature_dict["borderline_reasons"] = {
        **feature_dict["borderline_reasons"],
        "income_A": "revenu médian du zip code, corrélé à race_A (ANOVA sur individus uniques : "
                    "eta²=0.037, p=0.033 -- proxy géographique potentiel, cf. notebooks/00_eda_cleaning.ipynb §7)",
        "income_B": "revenu médian du zip code, corrélé à cand_race (même test, symétrique)",
    }

    logit_excluded = ["shar1_1_A", "shar1_1_B"]
    feature_dict["features_logit"] = [f for f in feature_dict["features"] if f not in logit_excluded]
    feature_dict["preprocessing_logit_specific"] = {
        "removed_features": logit_excluded,
        "reason": "Colinéarité parfaite par construction : attr1_1+sinc1_1+intel1_1+fun1_1+amb1_1"
                  "+shar1_1 somme à 100 pour chaque participant (contrainte structurelle confirmée "
                  "empiriquement, cf. docs/soutenance_notes.md) -- problématique pour un modèle "
                  "linéaire (logit) uniquement, pas pour XGBoost/TabICL. shar1_1_A/B retirées "
                  "uniquement des features du logit (feature_dict['features_logit']) -- analogue à "
                  "drop_first=True sur un encodage catégoriel : la 6e préférence devient une "
                  "référence implicite. XGBoost et TabICL gardent les 6 préférences brutes "
                  "(feature_dict['features']).",
        "alternative_considered": "Une transformation log-ratio (ex. isometric/additive log-ratio) "
                  "sur données compositionnelles serait plus rigoureuse pour traiter la contrainte "
                  "de somme à 100, mais a été écartée pour rester dans le calendrier du projet -- "
                  "simplification volontaire à assumer si le sujet vient en Q&A.",
    }
    return df, feature_dict
