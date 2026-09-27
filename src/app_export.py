"""
HEC Match — export des données de l'application web (app/index.html).

L'app est une page autonome (aucun serveur) :
  - les scores des 3 modèles sur les couples TEST sont précalculés (TabICL ne tourne que sur GPU) ;
  - XGBoost mitigé et le logit sont aussi exportés sous forme recalculable dans le navigateur
    (arbres et coefficients), pour le simulateur « et si ? ». La parité navigateur / Python est
    vérifiée par `verify_parity()` ;
  - chaque score est accompagné de son explication : SHAP exact pour XGBoost, contributions
    exactes (coefficient × valeur standardisée) pour le logit.

Logit de l'app = logit réentraîné proprement sur `features_logit` sans les 4 proxies (cf.
02_interpretability) : ses coefficients sont interprétables, contrairement à ceux de
`logit_mitigated.joblib` (colinéarité de l'ancien schéma). AUC quasi identique (0,578 vs 0,580).
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.build_dataset import DATA_DIR
from src.interpretability import (BASE_LABELS, CAREER, FIELD, GOAL, CATEGORIES, fit_clean_logit,
                                  label, load_context)

ROOT = DATA_DIR.parent
APP_DIR = ROOT / "app"
INTERESTS = CATEGORIES["Intérêts"]
INTEREST_FR = {k: BASE_LABELS[k].replace("aime ", "").replace("les ", "").replace("le ", "").replace("la ", "").replace("l'", "")
               for k in INTERESTS}

# Variables modifiables dans le simulateur : (variable, min, max, pas)
SIMULATOR = [
    ("imprace_A", 1, 10, 1), ("exphappy_A", 1, 10, 1), ("age_A", 18, 45, 1),
    ("movies_A", 1, 10, 1), ("gaming_A", 1, 10, 1), ("clubbing_A", 1, 10, 1), ("yoga_A", 1, 10, 1),
    ("sinc1_1_A", 0, 50, 1), ("age_B", 18, 45, 1), ("sports_B", 1, 10, 1), ("tv_B", 1, 10, 1),
    ("shar1_1_B", 0, 50, 1), ("fun1_1_B", 0, 50, 1), ("int_corr", -0.8, 0.9, 0.01),
]


def _flatten_tree(node, feat_index):
    """Arbre XGBoost (dump JSON) -> tableaux plats [feature, seuil, oui, non, manquant, feuille]."""
    nodes = {}

    def walk(n):
        nid = n["nodeid"]
        if "leaf" in n:
            nodes[nid] = [-1, 0.0, -1, -1, -1, n["leaf"]]
        else:
            nodes[nid] = [feat_index[n["split"]], n["split_condition"], n["yes"], n["no"], n["missing"], 0.0]
            for c in n["children"]:
                walk(c)

    walk(node)
    size = max(nodes) + 1
    return [nodes.get(i, [-1, 0, -1, -1, -1, 0.0]) for i in range(size)]


def export_xgb(model, features):
    b = model.get_booster()
    base = float(json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
    idx = {f: i for i, f in enumerate(features)}
    trees = [_flatten_tree(json.loads(t), idx) for t in b.get_dump(dump_format="json")]
    return {"base_margin": float(np.log(base / (1 - base))), "trees": trees}


def xgb_margin_py(x, xgbj):
    """Réplique Python exacte de l'évaluation faite en JavaScript (float32 comme XGBoost)."""
    m = xgbj["base_margin"]
    for t in xgbj["trees"]:
        i = 0
        while t[i][0] >= 0:
            f, thr, yes, no, miss, _ = t[i]
            v = x[f]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                i = miss
            else:
                i = yes if np.float32(v) < np.float32(thr) else no
        m += t[i][5]
    return m


def decode_cat(row, cat, side):
    for code in list(range(1, 19)):
        c = f"{cat}_{side}_{code}"
        if c in row.index and row[c] == 1:
            return code
    return None


def person(row, side):
    """Profil lisible d'un participant (côté A ou B d'une ligne)."""
    g = row[f"gender_{side}"] if side == "A" else (0 if row.cand_female == 1 else 1)
    interests = {k: row[f"{k}_{side}"] for k in INTERESTS if pd.notna(row[f"{k}_{side}"])}
    top = sorted(interests, key=lambda k: -interests[k])[:3]
    f, c, go = decode_cat(row, "field_cd", side), decode_cat(row, "career_c", side), decode_cat(row, "goal", side)
    return {"genre": "femme" if g == 0 else "homme",
            "age": None if pd.isna(row[f"age_{side}"]) else int(row[f"age_{side}"]),
            "etudes": FIELD.get(f, "non renseigné"), "carriere": CAREER.get(c, "non renseigné"),
            "objectif": GOAL.get(go, "non renseigné"),
            "interets": [INTEREST_FR[k] for k in top]}


def build(out=APP_DIR / "app_data.json"):
    ctx = load_context()
    train, test, F, xgb, fd = ctx["train"], ctx["test"], ctx["features"], ctx["xgb"], ctx["feature_dict"]
    test = test.reset_index(drop=True)
    X = test[F]

    # --- XGBoost mitigé : SHAP exact ---------------------------------------------------------
    exp = shap.TreeExplainer(xgb)(X)
    xgb_base = float(exp.base_values[0])

    # --- Logit propre : contributions exactes -------------------------------------------------
    pipe, lf = fit_clean_logit(train, fd)
    assert set(lf) <= set(F), "les variables du logit doivent être un sous-ensemble de celles de XGBoost"
    imp, sc, lr = pipe[0], pipe[1], pipe[-1]
    Z = sc.transform(imp.transform(test[lf]))
    contrib_logit = Z * lr.coef_[0]
    p_logit = pipe.predict_proba(test[lf])[:, 1]

    fidx = {f: i for i, f in enumerate(F)}

    def top_contrib(vals, feats, k=6):
        s = pd.Series(vals, index=feats)
        top = s.abs().sort_values(ascending=False).head(k).index
        return [[fidx[f], round(float(s[f]), 4)] for f in top], round(float(s.drop(top).sum()), 4)

    rows = []
    for i, r in test.iterrows():
        sx, sx_rest = top_contrib(exp.values[i], F)
        sl, sl_rest = top_contrib(contrib_logit[i], lf)
        rows.append({
            "a": int(r.iid), "b": int(r.pid), "s": int(r.wave), "dec": int(r.dec), "m": int(r.match),
            "p": [round(float(p_logit[i]), 4), round(float(r.proba_xgb_mitigated), 4),
                  round(float(r.proba_tabicl_mitigated), 4)],
            "shx": sx, "shx_r": sx_rest, "shl": sl, "shl_r": sl_rest,
            # précision float32 exacte (9 chiffres significatifs) : un arrondi plus grossier peut
            # faire basculer une branche d'arbre et fausser le score recalculé dans le navigateur
            "x": [None if pd.isna(v) else float(f"{np.float32(v):.9g}") for v in X.loc[i].values],
        })

    people = {}
    for _, r in test.iterrows():
        people.setdefault(int(r.iid), person(r, "A") | {"session": int(r.wave)})
        people.setdefault(int(r.pid), person(r, "B") | {"session": int(r.wave)})

    # --- Modèles recalculables -----------------------------------------------------------------
    xgbj = export_xgb(xgb, F)
    logitj = {"idx": [fidx[f] for f in lf], "median": [float(v) for v in imp.statistics_],
              "mean": [float(v) for v in sc.mean_], "scale": [float(v) for v in sc.scale_],
              "coef": [float(v) for v in lr.coef_[0]], "intercept": float(lr.intercept_[0])}

    rep = lambda p: json.loads((ROOT / "reports" / p).read_text())
    data = {
        "features": F, "labels": [label(f) for f in F],
        "simulator": [[fidx[f], lo, hi, st] for f, lo, hi, st in SIMULATOR],
        "derived": {"age_A": fidx["age_A"], "age_B": fidx["age_B"], "age_diff": fidx["age_diff"],
                    "abs_age_diff": fidx["abs_age_diff"]},
        "xgb_base": xgb_base, "logit_base": float(lr.intercept_[0]),
        "rows": rows, "people": people, "xgb": xgbj, "logit": logitj,
        "reports": {"tradeoffs": rep("tradeoffs/summary.json"), "economics": rep("economics/summary.json"),
                    "interpretability": {k: rep("interpretability/summary.json")[k]
                                         for k in ["shap_share_by_side", "borne_selectivite_popularite",
                                                   "shap_vs_lime", "tabicl_surrogate", "variables_consensus", "genre"]},
                    "top_k": pd.read_csv(ROOT / "reports/economics/top_k_par_utilisateur.csv").round(4).to_dict("records")},
    }
    APP_DIR.mkdir(exist_ok=True)

    def clean(o):   # NaN / inf ne sont pas du JSON valide -> null
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    out.write_text(json.dumps(clean(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return data, test, pipe, lf


def verify_parity(data, test, xgb, pipe, lf, F):
    """Les scores recalculés comme dans le navigateur doivent être ceux de Python."""
    import math
    xm = np.array([xgb_margin_py(r["x"], data["xgb"]) for r in data["rows"]])
    px_js = 1 / (1 + np.exp(-xm))
    px_py = xgb.predict_proba(test[F])[:, 1]
    L = data["logit"]
    pl_js = []
    for r in data["rows"]:
        z = L["intercept"]
        for j, fi in enumerate(L["idx"]):
            v = r["x"][fi]
            v = L["median"][j] if v is None else v
            z += L["coef"][j] * (v - L["mean"][j]) / L["scale"][j]
        pl_js.append(1 / (1 + math.exp(-z)))
    pl_py = pipe.predict_proba(test[lf])[:, 1]
    return float(np.abs(px_js - px_py).max()), float(np.abs(np.array(pl_js) - pl_py).max())
