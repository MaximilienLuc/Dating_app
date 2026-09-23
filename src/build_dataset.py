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

    dummies = pd.get_dummies(data[cat_cols].astype("Int64").astype(str), prefix=cat_cols).astype(int)
    dummies.columns = dummies.columns.str.replace("<NA>", "NA", regex=False)  # XGBoost refuse '<'
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
    return df, feature_dict
