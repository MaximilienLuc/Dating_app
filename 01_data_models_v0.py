"""
HEC Match — 01_data_models (v0)
Socle technique : chargement, profils pré-rendez-vous, paires (A, B), split par session, modèles de base.

Prérequis :
    pip install pandas numpy scikit-learn xgboost pyarrow joblib scipy tabpfn
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
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
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

# Variables limites : pas des attributs protégés, mais suspectées d'être des proxys
# (voir section 8, audit ANOVA income ~ race)
BORDERLINE = ["imprace_A", "imprace_B", "income_A", "income_B"]
BORDERLINE_REASONS = {
    "imprace_A": "importance déclarée de l'origine (auto-évaluée) : proxy direct de race_A",
    "imprace_B": "importance déclarée de l'origine (auto-évaluée) : proxy direct de race_B",
    "income_A": "revenu médian du zip code, corrélé à race_A "
                "(ANOVA sur individus uniques : eta²=0.037, p=0.033 -- voir section 8)",
    "income_B": "revenu médian du zip code, corrélé à cand_race "
                "(même test, symétrique -- voir section 8)",
}

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
        "borderline": BORDERLINE,
        "borderline_reasons": BORDERLINE_REASONS,
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

# ---------------------------------------------------------------------------
# 8. Audit qualité des données & proxys potentiels
#    Chaque fonction est autonome et réexécutable isolément sur `data`/`FEATURES`
#    (ex. dans un notebook, en rappelant juste audit_xxx(data) après un nouveau run).
# ---------------------------------------------------------------------------

def audit_missingness(df, cols=("fit_score", "income_A", "income_B"), wave_col="wave"):
    """Pourquoi : avant d'imputer, il faut savoir si le manquant vient d'un pépin de
    collecte propre à une session entière (auquel cas l'imputation classique est
    trompeuse) ou d'un manquant individuel diffus (auquel cas médiane/modèle est OK).
    """
    print("\n--- Audit missingness (variables clés) ---")
    report = {}
    for col in cols:
        pct = df[col].isna().mean() * 100
        report[col] = round(pct, 2)
        print(f"% manquant {col} : {pct:.2f}%")

    by_wave = df.groupby(wave_col)["income_A"].apply(lambda s: s.isna().mean() * 100)
    full_waves = by_wave[(by_wave == 0) | (by_wave == 100)]
    other_waves = by_wave.drop(full_waves.index)
    print(f"Waves income à 0%/100% de manquant : {full_waves.round(1).to_dict()} "
          f"-- exception attendue : wave 13 = 100% (blackout complet de la question "
          f"revenu pour cette session, à isoler/discuter à part).")
    print(f"Les {len(other_waves)} autres waves varient en continu entre "
          f"{other_waves.min():.1f}% et {other_waves.max():.1f}% -- pas de pattern "
          f"tout-ou-rien, cohérent avec du manquant individuel plutôt qu'un problème "
          f"de collecte par session.")
    report["full_missing_waves"] = full_waves.round(1).to_dict()
    return report


def audit_groupkfold_vs_single_split(df, features, target="dec", group_col="wave", seed=SEED):
    """Pourquoi : le split unique (GroupShuffleSplit, section 5) tire au hasard 25%
    des sessions en test -- un seul tirage peut tomber sur des waves atypiques par
    chance. La validation croisée GroupKFold (5 folds sur wave) donne une AUC
    moyenne + un écart-type, pour juger si l'AUC du split unique est représentative
    ou juste un coup de chance/malchance.
    """
    X, y, groups = df[features], df[target], df[group_col]
    gkf = GroupKFold(n_splits=5)
    aucs = {"logit": [], "xgb": []}
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        X_tr_f, X_te_f = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr_f, y_te_f = y.iloc[tr_idx], y.iloc[te_idx]

        m_logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(max_iter=5000))
        m_logit.fit(X_tr_f, y_tr_f)
        aucs["logit"].append(roc_auc_score(y_te_f, m_logit.predict_proba(X_te_f)[:, 1]))

        m_xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
        m_xgb.fit(X_tr_f, y_tr_f)
        aucs["xgb"].append(roc_auc_score(y_te_f, m_xgb.predict_proba(X_te_f)[:, 1]))

    print("\n--- Audit GroupKFold (5 folds sur wave) vs split unique ---")
    summary = {}
    for name, scores in aucs.items():
        scores = np.array(scores)
        mean, std = scores.mean(), scores.std()
        summary[name] = {"mean": round(mean, 3), "std": round(std, 3)}
        single = results.get(name)
        single_txt = f" (split unique section 7 : {single:.3f})" if single is not None else ""
        print(f"{name} : AUC 5-fold = {mean:.3f} ± {std:.3f}{single_txt}")
    return summary


def audit_income_race_anova(df):
    """Pourquoi ce test : income_A/income_B (revenu médian du zip code) sont-ils des
    proxys de race_A/cand_race ? Si oui, un modèle qui utilise income peut reproduire
    indirectement une discrimination par origine même si race_A/cand_race ne sont
    jamais en entrée -- à vérifier avant de figer la liste de features (samedi).

    Pourquoi dédupliquer : chaque individu apparaît dans `df` une fois par candidat
    rencontré (~15 lignes par personne dans une wave), avec la MEME valeur
    income_A/race_A répétée à chaque ligne. Faire l'ANOVA sur les paires viole donc
    l'indépendance des observations (pseudo-réplication) : ça gonfle artificiellement
    F et fait chuter la p-value. Constaté sur income_A ~ race_A : sur les paires
    (n=4273) F=48.68 p<0.0001 ; sur individus uniques (n=281, dédup sur iid) F=2.66
    p=0.033. L'eta² (taille d'effet, ~0.04 dans les deux cas) ne bouge presque pas :
    l'effet est réel, c'est la CONFIANCE statistique qui était surestimée par la
    pseudo-réplication, pas l'ampleur de l'effet.

    Les deux tests "own" (income_A~race_A, income_B~cand_race) sont donc faits sur
    individus uniques (dédup sur iid / pid respectivement). Les deux tests de
    CONTRÔLE (income_A~cand_race, income_B~race_A) testent le lien entre le revenu
    d'un individu et l'origine de son PARTENAIRE, qui change à chaque ligne pour un
    même individu (il rencontre plusieurs candidats différents dans sa wave) : il
    n'existe donc pas de valeur "cand_race" unique par individu à dédupliquer -- ces
    deux tests restent au niveau des paires, ce qui est le bon niveau pour vérifier
    l'absence de biais d'appariement (pas un problème de pseudo-réplication de la
    même nature que les tests "own").
    """
    RACE_LABELS = {1: "Black", 2: "White", 3: "Latino", 4: "Asian", 5: "Native Am.", 6: "Other"}

    def anova_eta(sub, group_col, value_col):
        groups = [g[value_col].values for _, g in sub.groupby(group_col)]
        f_stat, p_val = stats.f_oneway(*groups)
        grand_mean = sub[value_col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((sub[value_col] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
        means = sub.groupby(group_col)[value_col].agg(["mean", "count"]).round(0)
        means.index = means.index.map(lambda x: RACE_LABELS.get(int(x), x))
        return f_stat, p_val, eta_sq, means

    uniq_A = df.drop_duplicates("iid")[["iid", "race_A", "income_A"]].dropna()
    uniq_B = df.drop_duplicates("pid")[["pid", "cand_race", "income_B"]].dropna()
    pairs_ctrl_A = df[["cand_race", "income_A"]].dropna()
    pairs_ctrl_B = df[["race_A", "income_B"]].dropna()

    tests = [
        ("income_A ~ race_A (individus uniques, dédup iid)", uniq_A, "race_A", "income_A"),
        ("income_A ~ cand_race (contrôle, paires)", pairs_ctrl_A, "cand_race", "income_A"),
        ("income_B ~ cand_race (individus uniques, dédup pid)", uniq_B, "cand_race", "income_B"),
        ("income_B ~ race_A (contrôle, paires)", pairs_ctrl_B, "race_A", "income_B"),
    ]
    print("\n--- Audit ANOVA income ~ race (proxys potentiels, dédupliqué) ---")
    report = {}
    for label, sub, group_col, value_col in tests:
        f_stat, p_val, eta_sq, means = anova_eta(sub, group_col, value_col)
        print(f"{label} [n={sub.shape[0]}] : F={f_stat:.2f} p={p_val:.4f} eta²={eta_sq:.4f}")
        report[label] = {"F": round(f_stat, 2), "p": round(p_val, 4), "eta_sq": round(eta_sq, 4)}
    return report


missingness_report = audit_missingness(data)
groupkfold_report = audit_groupkfold_vs_single_split(data, FEATURES)
anova_report = audit_income_race_anova(data)
