"""
HEC Match — 01_data_models (v0)
Socle technique : chargement, profils pré-rendez-vous, paires (A, B), split par session, modèles de base.

Prérequis :
    pip install pandas numpy scikit-learn xgboost pyarrow joblib tabpfn
    Télécharger "Speed Dating Data.csv" (Kaggle : Speed Dating Experiment) dans data/
    Garder aussi "Speed Dating Data Key.doc" (dictionnaire des variables).
"""
import json
import warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42
DATA, MODELS = Path("data"), Path("models")
MODELS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA / "Speed Dating Data.csv", encoding="ISO-8859-1")
print("Brut :", df.shape)  # attendu : (8378, 195)

# ---------------------------------------------------------------------------
# 2. Profil PRÉ-rendez-vous de chaque participant (questionnaire d'inscription, suffixe _1)
#    Tout ce qui est noté pendant/après la soirée est EXCLU (fuite d'information).
# ---------------------------------------------------------------------------
INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming",
             "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
PREFS = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]  # ce que je recherche
SELF = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"]              # comment je me note
OTHER = ["age", "field_cd", "career_c", "goal", "date", "go_out", "imprace", "imprelig",
         "exphappy", "income"]
PROTECTED_RAW = ["gender", "race"]  # gender : 0 = femme, 1 = homme

prof = df[["iid"] + OTHER + INTERESTS + PREFS + SELF + PROTECTED_RAW].drop_duplicates("iid").copy()
prof["income"] = pd.to_numeric(prof["income"].astype(str).str.replace(",", ""), errors="coerce")

# Vagues 6-9 : préférences notées de 1 à 10 au lieu de 100 points à répartir -> renormalisation en parts de 100
s = prof[PREFS].sum(axis=1).replace(0, np.nan)
prof[PREFS] = prof[PREFS].div(s, axis=0) * 100

# ---------------------------------------------------------------------------
# 3. Paires (A décide, B est le candidat)
# ---------------------------------------------------------------------------
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
PROTECTED = ["cand_female", "cand_race", "same_race", "gender_A", "race_A"]

# ---------------------------------------------------------------------------
# 4. Liste des variables d'entrée
# ---------------------------------------------------------------------------
CAT = ["field_cd", "career_c", "goal"]
num_base = [c for c in OTHER if c not in CAT] + INTERESTS + PREFS + SELF
cat_cols = [f"{c}_{side}" for c in CAT for side in "AB"]

dummies = pd.get_dummies(data[cat_cols].astype("Int64").astype(str), prefix=cat_cols).astype(int)
dummies.columns = dummies.columns.str.replace("<NA>", "NA", regex=False)  # XGBoost refuse '<'
data = pd.concat([data, dummies], axis=1)

FEATURES = ([f"{c}_A" for c in num_base] + [f"{c}_B" for c in num_base]
            + ["int_corr", "age_diff", "abs_age_diff", "fit_score"] + list(dummies.columns))
print("Paires :", data.shape, "| variables d'entrée :", len(FEATURES),
      "| taux de oui :", round(data["dec"].mean(), 3))

# ---------------------------------------------------------------------------
# 5. Split PAR SESSION (wave) : tous les participants d'une soirée dans le même échantillon
#    (sinon (A, B) en train et (B, A) en test = fuite)
# ---------------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_idx, te_idx = next(gss.split(data, groups=data["wave"]))
train_waves = sorted(data.iloc[tr_idx]["wave"].unique().tolist())
test_waves = sorted(data.iloc[te_idx]["wave"].unique().tolist())
print("Sessions test :", test_waves, "| lignes train/test :", len(tr_idx), len(te_idx))

X_tr, y_tr = data.iloc[tr_idx][FEATURES], data.iloc[tr_idx]["dec"]
X_te, y_te = data.iloc[te_idx][FEATURES], data.iloc[te_idx]["dec"]

# ---------------------------------------------------------------------------
# 6. Sauvegarde du contrat de fichiers
# ---------------------------------------------------------------------------
keep = ["iid", "pid", "wave", "dec", "match"] + PROTECTED + FEATURES
data[keep].to_parquet(DATA / "clean.parquet", index=False)

with open(DATA / "features.json", "w") as f:
    json.dump({
        "target": "dec",
        "features": FEATURES,
        "protected": PROTECTED,
        "borderline": ["imprace_A", "imprace_B"],  # importance déclarée de la même origine : à discuter
        "excluded_rule": "Toute variable notée pendant ou après la soirée (attr, sinc, intel, fun, amb, "
                         "shar, like, prob, met, *_o, match_es, *_s, *_2, *_3, dec_o, match) + ordre du "
                         "rendez-vous (round, position, order)",
    }, f, indent=2)

with open(DATA / "split.json", "w") as f:
    json.dump({"seed": SEED, "group": "wave", "train_waves": train_waves, "test_waves": test_waves}, f)

# ---------------------------------------------------------------------------
# 7. Modèles de base
# ---------------------------------------------------------------------------
results = {}

logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=5000))
logit.fit(X_tr, y_tr)
results["logit"] = roc_auc_score(y_te, logit.predict_proba(X_te)[:, 1])
joblib.dump(logit, MODELS / "logit.joblib")

xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)
xgb.fit(X_tr, y_tr)  # gère les valeurs manquantes nativement
results["xgb"] = roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1])
joblib.dump(xgb, MODELS / "xgb.joblib")

# TabPFN : passé pour la v0 CPU (à faire sur GPU/Colab cet après-midi)
RUN_TABPFN = False
if RUN_TABPFN:
    try:
        from tabpfn import TabPFNClassifier
        tab = TabPFNClassifier(random_state=SEED)
        tab.fit(X_tr, y_tr)
        results["tabpfn"] = roc_auc_score(y_te, tab.predict_proba(X_te)[:, 1])
        joblib.dump(tab, MODELS / "tabpfn.joblib")
    except Exception as e:
        print("TabPFN non lancé :", e)
else:
    print("TabPFN : passé pour la v0 (à entraîner sur GPU/Colab cet après-midi)")

print("AUC test :", {k: round(v, 3) for k, v in results.items()})
