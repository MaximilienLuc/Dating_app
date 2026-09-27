"""
HEC Match — interprétabilité des modèles (bloc Alex).

Couvre les méthodes du cours ISAF, appliquées aux modèles de référence de la soutenance :
  - XGBoost mitigé (models/xgb_mitigated.joblib) : SHAP global et local (TreeExplainer, exact),
    PDP / ICE, LIME, désaccord SHAP vs LIME (Krishna et al.), permutation importance (AUC).
  - Logit (white box) : ré-entraîné proprement sur `features_logit` moins les 4 proxies raciaux
    retirés par Max. Raison : `logit_mitigated.joblib` a été entraîné sur l'ancien schéma
    (shar1_1 + colonnes de référence des dummies) et ses coefficients sont donc pollués par la
    colinéarité (cf. CLAUDE.md). Ses probabilités restent utilisables, pas ses coefficients.
    On affiche odds ratios par écart-type, IC à 95 % et effets marginaux moyens (AME).
  - TabICL mitigé : ne se charge que sur GPU (Colab). On l'interprète via un modèle surrogate
    (arbre peu profond) ajusté sur ses prédictions figées (data/tabicl_mitigated_predictions.parquet),
    avec mesure de fidélité par validation croisée groupée par session.
  - Accord entre modèles : corrélation de rang des importances globales.

Tout est calculé sur l'échantillon TEST (sessions de split.json) sauf l'ajustement du logit
propre (train). Sorties : reports/interpretability/ (PNG pour les slides, CSV et summary.json
pour l'app).
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.build_dataset import DATA_DIR, SEED
from src.performance import (load_mitigated_predictions, load_model_predictions,
                             split_train_test)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = DATA_DIR.parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "reports" / "interpretability"
PROXIES = ["go_out_B", "attr3_1_B", "intel3_1_B", "career_c_B_12"]

# --------------------------------------------------------------------------- #
#  Style (palette de référence du skill dataviz : bleu/rouge divergents,      #
#  encre texte neutre, grille discrète)                                       #
# --------------------------------------------------------------------------- #
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SIDE_COLORS = {"A (décideur)": BLUE, "B (candidat)": ORANGE, "Couple": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# --------------------------------------------------------------------------- #
#  Libellés lisibles pour les slides                                          #
# --------------------------------------------------------------------------- #
BASE_LABELS = {
    "age": "âge", "date": "fréquence de dates (1=souvent)", "go_out": "fréquence de sorties (1=souvent)",
    "imprace": "importance même origine", "imprelig": "importance même religion",
    "exphappy": "optimisme sur la soirée", "income": "revenu médian du quartier",
    "income_missing": "revenu non renseigné",
    "sports": "aime le sport", "tvsports": "aime le sport à la TV", "exercise": "aime la muscu",
    "dining": "aime les restaurants", "museums": "aime les musées", "art": "aime l'art",
    "hiking": "aime la randonnée", "gaming": "aime les jeux vidéo", "clubbing": "aime sortir en boîte",
    "reading": "aime lire", "tv": "aime la TV", "theater": "aime le théâtre", "movies": "aime le cinéma",
    "concerts": "aime les concerts", "music": "aime la musique", "shopping": "aime le shopping",
    "yoga": "aime le yoga",
    "attr1_1": "cherche : physique", "sinc1_1": "cherche : sincérité", "intel1_1": "cherche : intelligence",
    "fun1_1": "cherche : humour", "amb1_1": "cherche : ambition", "shar1_1": "cherche : intérêts communs",
    "attr3_1": "se note : physique", "sinc3_1": "se note : sincérité", "intel3_1": "se note : intelligence",
    "fun3_1": "se note : humour", "amb3_1": "se note : ambition",
}
COUPLE_LABELS = {"int_corr": "Couple : intérêts en commun", "age_diff": "Couple : écart d'âge (B − A)",
                 "abs_age_diff": "Couple : écart d'âge absolu", "fit_score": "Couple : adéquation attentes/profil"}
FIELD = {1: "Droit", 2: "Maths", 3: "Sc. sociales/Psy", 4: "Médecine/Pharma", 5: "Ingénierie",
         6: "Lettres/Journalisme", 7: "Histoire/Philo", 8: "Business/Éco/Finance", 9: "Éducation",
         10: "Sciences", 11: "Travail social", 12: "Undergrad/indécis", 13: "Sc. Po/Relations int.",
         14: "Cinéma", 15: "Beaux-arts", 16: "Langues", 17: "Architecture", 18: "Autre"}
CAREER = {1: "Avocat", 2: "Recherche", 3: "Psychologue", 4: "Médecin", 5: "Ingénieur",
          6: "Arts/Spectacle", 7: "Finance/Conseil/Business", 8: "Immobilier", 9: "Humanitaire",
          10: "Indécis", 11: "Travail social", 12: "Orthophonie", 13: "Politique", 14: "Sport pro",
          15: "Autre", 16: "Journalisme", 17: "Architecture"}
GOAL = {1: "soirée fun", 2: "rencontrer du monde", 3: "obtenir un date", 4: "relation sérieuse",
        5: "pour le dire", 6: "autre"}
CATEGORIES = {
    "Intérêts": ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming",
                 "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"],
    "Ce que je cherche": ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"],
    "Comment je me note": ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"],
}


def side_of(f):
    if f in COUPLE_LABELS:
        return "Couple"
    return "A (décideur)" if (f.endswith("_A") or "_A_" in f) else "B (candidat)"


def base_of(f):
    for cat in ("field_cd", "career_c", "goal"):
        if f.startswith(cat + "_"):
            return cat
    return f[:-2] if f.endswith(("_A", "_B")) else f


def category_of(f):
    if f in COUPLE_LABELS:
        return "Variables de couple"
    b = base_of(f)
    for cat, members in CATEGORIES.items():
        if b in members:
            return cat
    if b in ("field_cd", "career_c", "goal"):
        return "Études / carrière / objectif"
    return "Démographie / habitudes"


def label(f):
    if f in COUPLE_LABELS:
        return COUPLE_LABELS[f]
    who = "A" if side_of(f) == "A (décideur)" else "B"
    for cat, mapping, name in (("field_cd", FIELD, "études"), ("career_c", CAREER, "carrière"),
                               ("goal", GOAL, "objectif")):
        if f.startswith(cat + "_"):
            code = f.split("_")[-1]
            val = "non renseigné" if code == "NA" else mapping.get(int(code), code)
            return f"{who} : {name} = {val}"
    return f"{who} : {BASE_LABELS.get(base_of(f), base_of(f))}"


def is_dummy(f):
    return base_of(f) in ("field_cd", "career_c", "goal") or f.startswith("income_missing")


def value_label(f, v):
    """Libellé « variable = valeur » lisible pour les explications locales."""
    if pd.isna(v):
        return f"{label(f)} : non renseigné"
    if is_dummy(f):
        return label(f) if v == 1 else label(f).replace(" = ", " ≠ ")
    if base_of(f) == "income":
        return f"{label(f)} = {v:,.0f} $".replace(",", " ")
    return f"{label(f)} = {v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) and v % 1 else f"{label(f)} = {v:.0f}"


def lime_rule_label(f, rule):
    """Traduit une règle LIME (« field_cd_B_16=0 », « 3.00 < age_A <= 5.00 ») en libellé lisible."""
    if is_dummy(f):
        return label(f) if rule.strip().endswith("=1") else label(f).replace(" = ", " ≠ ")
    return rule.replace(f, label(f))


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)
    return OUT_DIR / f"{name}.png"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------- #
#  Chargement                                                                 #
# --------------------------------------------------------------------------- #
def load_context():
    data, feature_dict = load_model_predictions()
    data, mitigated_features = load_mitigated_predictions(data)
    train, test, split = split_train_test(data)
    xgb = joblib.load(MODELS_DIR / "xgb_mitigated.joblib")
    return {"data": data, "feature_dict": feature_dict, "features": mitigated_features,
            "train": train, "test": test, "split": split, "xgb": xgb}


# --------------------------------------------------------------------------- #
#  1. SHAP — XGBoost mitigé                                                   #
# --------------------------------------------------------------------------- #
def shap_values(model, X):
    import shap
    explainer = shap.TreeExplainer(model)
    exp = explainer(X)
    # Vérification de l'additivité (propriété d'efficacité de Shapley) en log-odds
    margin = model.predict(X, output_margin=True)
    err = np.abs(exp.values.sum(1) + explainer.expected_value - margin).max()
    assert err < 1e-3, f"additivité SHAP violée ({err})"
    return exp, float(explainer.expected_value)


def global_importance(shap_vals, features):
    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_vals).mean(0)})
    imp["label"] = imp.feature.map(label)
    imp["side"] = imp.feature.map(side_of)
    imp["category"] = imp.feature.map(category_of)
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def direction_table(shap_vals, X, features, top):
    """Sens de l'effet : corrélation de rang entre la valeur de la variable et sa valeur SHAP."""
    rows = []
    V = pd.DataFrame(shap_vals, columns=features, index=X.index)
    for f in top:
        ok = X[f].notna()
        rho = spearmanr(X.loc[ok, f], V.loc[ok, f])[0] if X.loc[ok, f].nunique() > 1 else np.nan
        rows.append({"feature": f, "label": label(f), "spearman_value_vs_shap": round(rho, 2),
                     "effet": "↑ plus de oui" if rho > 0.3 else "↓ moins de oui" if rho < -0.3 else "non monotone"})
    return pd.DataFrame(rows)


def plot_global_importance(imp, n=15):
    top = imp.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(top.label, top.mean_abs_shap, color=top.side.map(SIDE_COLORS), height=0.62)
    ax.set_xlabel("Impact moyen sur la prédiction (|SHAP| moyen, log-odds)")
    ax.set_title(f"XGBoost mitigé — les {n} variables les plus influentes (SHAP, test)")
    ax.grid(axis="y", visible=False)
    present = [s for s in SIDE_COLORS if s in set(top.side)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=SIDE_COLORS[s]) for s in present]
    ax.legend(handles, present, loc="lower right", frameon=False)
    return _save(fig, "01_shap_global_top15")


def plot_side_and_category(imp):
    by_cat = imp.groupby(["category", "side"]).share.sum().unstack(fill_value=0)
    by_cat = by_cat.loc[by_cat.sum(1).sort_values().index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    side = imp.groupby("side").share.sum().reindex(SIDE_COLORS.keys())
    axes[0].bar(side.index, side.values * 100, color=[SIDE_COLORS[s] for s in side.index], width=0.6)
    for i, v in enumerate(side.values):
        axes[0].text(i, v * 100 + 1, f"{v:.0%}", ha="center", color=INK, fontweight="bold")
    axes[0].set_ylabel("% de l'importance totale")
    axes[0].set_title("Qui explique la décision ?")
    axes[0].grid(axis="x", visible=False)
    axes[0].set_ylim(0, max(side.values) * 118)
    left = np.zeros(len(by_cat))
    for s in SIDE_COLORS:
        if s in by_cat:
            axes[1].barh(by_cat.index, by_cat[s] * 100, left=left, color=SIDE_COLORS[s], label=s,
                         height=0.6, edgecolor=SURFACE, linewidth=2)
            left += by_cat[s].values * 100
    axes[1].set_xlabel("% de l'importance totale (|SHAP| moyen)")
    axes[1].set_title("Par famille de variables")
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, "02_shap_by_side_and_category")


def plot_beeswarm(exp, X, imp, n=12):
    import shap
    exp_lab = exp[:, list(imp.feature.head(n))]
    exp_lab.feature_names = list(imp.label.head(n))
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(exp_lab, max_display=n, show=False, color_bar_label="Valeur de la variable")
    plt.title("XGBoost mitigé — sens des effets (SHAP summary plot, test)", loc="left", fontweight="bold")
    plt.xlabel("Valeur SHAP (log-odds) — à droite : pousse vers « oui »")
    fig = plt.gcf()
    return _save(fig, "03_shap_beeswarm")


# --------------------------------------------------------------------------- #
#  2. Explications locales (SHAP + LIME) sur des couples précis               #
# --------------------------------------------------------------------------- #
def pick_examples(test, proba_col="proba_xgb_mitigated"):
    """Deux couples illustratifs : un « oui » bien prédit, un « non » bien prédit."""
    t = test.reset_index(drop=True)
    yes = t[(t.dec == 1)].sort_values(proba_col, ascending=False).index[5]
    no = t[(t.dec == 0)].sort_values(proba_col).index[5]
    return {"oui_bien_predit": int(yes), "non_bien_predit": int(no)}


def describe_couple(row):
    g = {0: "femme", 1: "homme"}
    race = {1: "noir·e", 2: "blanc·he", 3: "latino", 4: "asiatique", 6: "autre"}
    a = f"A : {g.get(row.gender_A, '?')}, {row.age_A:.0f} ans"
    b = f"B : {'femme' if row.cand_female == 1 else 'homme'}, {row.age_B:.0f} ans"
    return f"{a} · {b} · origine de B : {race.get(row.cand_race, '?')} · décision réelle : {'OUI' if row.dec == 1 else 'NON'}"


def plot_local_shap(exp, base_value, test, idx, features, name, n=8):
    vals = pd.Series(exp.values[idx], index=features)
    top = vals.abs().sort_values(ascending=False).head(n).index
    rest = vals.drop(top).sum()
    contrib = pd.concat([vals[top], pd.Series({"Toutes les autres variables": rest})])
    contrib = contrib.iloc[::-1]
    xrow = test.iloc[idx]
    labels = [value_label(f, xrow[f]) if f in features else f for f in contrib.index]
    p0, p1 = sigmoid(base_value), sigmoid(base_value + vals.sum())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(labels, contrib.values, color=[BLUE if v > 0 else RED for v in contrib.values], height=0.6)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Contribution SHAP (log-odds) — bleu : vers « oui », rouge : vers « non »")
    ax.set_title(f"Pourquoi ce score ? Moyenne {p0:.0%} → ce couple {p1:.0%}")
    ax.text(0, -0.16, describe_couple(xrow), transform=ax.transAxes, color=INK2, fontsize=10)
    ax.grid(axis="y", visible=False)
    return _save(fig, name), {"proba_base": round(p0, 3), "proba_couple": round(p1, 3),
                              "description": describe_couple(xrow),
                              "top_contributions": {label(f): round(float(vals[f]), 3) for f in top}}


def make_lime(train, features, model):
    from lime.lime_tabular import LimeTabularExplainer
    imputer = SimpleImputer(strategy="median").fit(train[features])
    Xtr = imputer.transform(train[features])
    # Les variables indicatrices (dummies, revenu manquant) sont déclarées catégorielles :
    # LIME les ré-échantillonne alors selon leur fréquence au lieu de les discrétiser.
    cat_idx = [i for i, f in enumerate(features)
               if base_of(f) in ("field_cd", "career_c", "goal") or f.startswith("income_missing")]
    explainer = LimeTabularExplainer(Xtr, feature_names=features, class_names=["non", "oui"],
                                     categorical_features=cat_idx, mode="classification",
                                     discretize_continuous=True, random_state=SEED)

    def predict_fn(Z):
        return model.predict_proba(pd.DataFrame(Z, columns=features))

    return explainer, imputer, predict_fn


def lime_explain(explainer, imputer, predict_fn, x_row, features, n=8, num_samples=3000):
    x = imputer.transform(x_row[features].to_frame().T)[0]
    e = explainer.explain_instance(x, predict_fn, num_features=n, num_samples=num_samples)
    # e.as_map()[1] : (indice variable, poids) ; e.as_list() : (règle discrétisée, poids)
    return [(features[i], rule, w) for (i, w), (rule, _) in zip(e.as_map()[1], e.as_list())], e.score


def plot_shap_vs_lime(shap_row, lime_list, features, name, title):
    s = pd.Series(shap_row, index=features)
    s_top = s.abs().sort_values(ascending=False).head(8).index
    l_top = [f for f, _, _ in lime_list]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    sv = s[s_top].iloc[::-1]
    axes[0].barh([label(f) for f in sv.index], sv.values, color=[BLUE if v > 0 else RED for v in sv.values], height=0.6)
    axes[0].set_title("SHAP (exact, TreeExplainer)")
    lv = pd.Series({lime_rule_label(f, rule): w for f, rule, w in lime_list}).iloc[::-1]
    axes[1].barh(lv.index, lv.values, color=[BLUE if v > 0 else RED for v in lv.values], height=0.6)
    axes[1].set_title("LIME (modèle linéaire local)")
    for ax in axes:
        ax.axvline(0, color=INK2, linewidth=1)
        ax.grid(axis="y", visible=False)
    common = len(set(s_top) & set(l_top))
    fig.suptitle(f"{title} — {common}/8 variables en commun dans les deux top 8", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, name)


def shap_lime_agreement(exp, test, explainer, imputer, predict_fn, features, n_couples=100, k=5):
    """Désaccord SHAP vs LIME (Krishna et al., 2022) : accord des top-k variables et des signes."""
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(test), size=n_couples, replace=False)
    rows = []
    for i in idxs:
        s = pd.Series(exp.values[i], index=features)
        s_top = list(s.abs().sort_values(ascending=False).head(k).index)
        lime_list, _ = lime_explain(explainer, imputer, predict_fn, test.iloc[i], features, n=k, num_samples=2000)
        l_top = [f for f, _, _ in lime_list]
        inter = set(s_top) & set(l_top)
        sign = [np.sign(s[f]) == np.sign(w) for f, _, w in lime_list if f in inter]
        rows.append({"idx": int(i), "feature_agreement": len(inter) / k,
                     "rank1_agreement": s_top[0] == l_top[0],
                     "sign_agreement": np.mean(sign) if sign else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  3. PDP / ICE                                                               #
# --------------------------------------------------------------------------- #
def pdp_ice(model, X, feature, grid_size=20, n_ice=150):
    x = X[feature].dropna()
    grid = np.unique(np.quantile(x, np.linspace(0.02, 0.98, grid_size)))
    rng = np.random.default_rng(SEED)
    sample = X.iloc[rng.choice(len(X), size=min(n_ice, len(X)), replace=False)]
    ice = np.empty((len(sample), len(grid)))
    pdp = np.empty(len(grid))
    for j, v in enumerate(grid):
        Xg = X.copy(); Xg[feature] = v
        pdp[j] = model.predict_proba(Xg)[:, 1].mean()
        Xs = sample.copy(); Xs[feature] = v
        ice[:, j] = model.predict_proba(Xs)[:, 1]
    return grid, pdp, ice


def plot_pdp_ice(model, X, feats, name):
    fig, axes = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 4.2), sharey=True)
    out = {}
    for ax, f in zip(np.atleast_1d(axes), feats):
        grid, pdp, ice = pdp_ice(model, X, f)
        ice_c = ice - ice[:, [0]] + pdp[0]   # ICE centrées sur la PDP au premier point de grille
        for line in ice_c:
            ax.plot(grid, line, color=INK2, alpha=0.08, linewidth=0.8)
        ax.plot(grid, pdp, color=BLUE, linewidth=2.5, label="PDP (moyenne)")
        ax.set_title(label(f), fontsize=11)
        ax.set_xlabel("valeur")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        out[f] = {"grid": [round(float(g), 3) for g in grid], "pdp": [round(float(p), 4) for p in pdp],
                  "pdp_range_pp": round(float((pdp.max() - pdp.min()) * 100), 1)}
    np.atleast_1d(axes)[0].set_ylabel("P(oui) prédite")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left")
    fig.suptitle("XGBoost mitigé — PDP (bleu) et ICE centrées (gris, 150 couples)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, name), out


# --------------------------------------------------------------------------- #
#  4. Permutation importance (contribution à la PERFORMANCE)                 #
# --------------------------------------------------------------------------- #
def permutation_auc(model, X, y, n_repeats=10):
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=SEED, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "label": [label(f) for f in X.columns],
                         "auc_drop": r.importances_mean, "auc_drop_std": r.importances_std}
                        ).sort_values("auc_drop", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  5. Logit white box propre                                                  #
# --------------------------------------------------------------------------- #
def fit_clean_logit(train, feature_dict):
    feats = [f for f in feature_dict["features_logit"] if f not in PROXIES]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, random_state=SEED))
    pipe.fit(train[feats], train["dec"])
    return pipe, feats


def logit_table(pipe, feats, train, test, n_boot=200):
    """Coefficients standardisés (par écart-type), odds ratios, IC 95 % par bootstrap de
    SESSIONS d'entraînement (cohérent avec le split par wave), et effets marginaux moyens."""
    coef = pipe[-1].coef_[0]
    waves = train.wave.unique()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        w = rng.choice(waves, size=len(waves), replace=True)
        b = pd.concat([train[train.wave == wi] for wi in w])
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=SEED)).fit(b[feats], b["dec"])
        boots.append(p[-1].coef_[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    # AME par écart-type : moyenne de p(1-p) * beta_std sur le test
    Z = pipe[:-1].transform(test[feats])
    p = pipe[-1].predict_proba(Z)[:, 1]
    ame = (p * (1 - p)).mean() * coef
    sd = pipe[1].scale_
    tab = pd.DataFrame({"feature": feats, "label": [label(f) for f in feats], "coef_per_sd": coef,
                        "odds_ratio_per_sd": np.exp(coef), "or_ci_low": np.exp(lo), "or_ci_high": np.exp(hi),
                        "significant_95": (lo > 0) | (hi < 0), "sign_stable_pct": (np.sign(boots) == np.sign(coef)).mean(0),
                        "ame_pp_per_sd": ame * 100, "sd_original_units": sd})
    return tab.reindex(tab.coef_per_sd.abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_logit_forest(tab, n=15):
    top = tab.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    y = np.arange(len(top))
    colors = [(BLUE if o > 1 else RED) if s else "#a3a29e" for o, s in zip(top.odds_ratio_per_sd, top.significant_95)]
    ax.hlines(y, top.or_ci_low, top.or_ci_high, color=colors, linewidth=2)
    ax.scatter(top.odds_ratio_per_sd, y, color=colors, s=46, zorder=3)
    ax.axvline(1, color=INK2, linewidth=1)
    ax.set_yticks(y, top.label)
    ax.set_xscale("log")
    ticks = [0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75]
    ticks = [t for t in ticks if top.or_ci_low.min() * 0.95 <= t <= top.or_ci_high.max() * 1.05]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.set_xlabel("Odds ratio pour +1 écart-type (IC 95 %, bootstrap de sessions) — gris : non significatif")
    ax.set_title(f"Logit (white box) — les {n} plus gros effets")
    ax.grid(axis="y", visible=False)
    return _save(fig, "08_logit_odds_ratios")


# --------------------------------------------------------------------------- #
#  6. TabICL — modèle surrogate                                               #
# --------------------------------------------------------------------------- #
def tabicl_surrogate(test, features, proba_col="proba_tabicl_mitigated", depth=3):
    X = test[features].fillna(test[features].median())
    y = test[proba_col].values
    gkf = GroupKFold(n_splits=len(test.wave.unique()))
    oof = np.empty(len(y))
    for tr_i, va_i in gkf.split(X, y, groups=test.wave):
        m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=40, random_state=SEED).fit(X.iloc[tr_i], y[tr_i])
        oof[va_i] = m.predict(X.iloc[va_i])
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=40, random_state=SEED).fit(X, y)
    fidelity = {"r2_in_sample": round(r2_score(y, tree.predict(X)), 3),
                "r2_cv_by_session": round(r2_score(y, oof), 3),
                "spearman_cv": round(spearmanr(y, oof)[0], 3),
                "auc_surrogate_vs_dec": round(roc_auc_score(test.dec, tree.predict(X)), 3),
                "auc_tabicl_vs_dec": round(roc_auc_score(test.dec, y), 3)}
    imp = pd.DataFrame({"feature": features, "label": [label(f) for f in features],
                        "importance": tree.feature_importances_}).sort_values("importance", ascending=False)
    return tree, fidelity, imp


def tabicl_surrogate_xgb(test, features, proba_col="proba_tabicl_mitigated"):
    """Surrogate plus souple (XGBoost de profondeur 2) : moins lisible qu'un arbre, mais bien plus
    fidèle. On l'explique ensuite par SHAP -> meilleure approximation globale de TabICL.
    N.B. : seules les lignes TEST sont utilisées. Sur les lignes d'entraînement, les prédictions
    de TabICL sont « in-context » (AUC 0,92 contre 0,63 en test) : elles décrivent la mémoire du
    modèle, pas sa fonction de décision."""
    import shap
    from xgboost import XGBRegressor

    def make():
        return XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                            random_state=SEED)

    X, y = test[features], test[proba_col].values
    oof = np.empty(len(y))
    for tr_i, va_i in GroupKFold(n_splits=len(test.wave.unique())).split(X, y, groups=test.wave):
        oof[va_i] = make().fit(X.iloc[tr_i], y[tr_i]).predict(X.iloc[va_i])
    model = make().fit(X, y)
    sv = shap.TreeExplainer(model)(X).values
    imp = pd.DataFrame({"feature": features, "label": [label(f) for f in features],
                        "importance": np.abs(sv).mean(0)}).sort_values("importance", ascending=False)
    fidelity = {"r2_cv_by_session": round(r2_score(y, oof), 3), "spearman_cv": round(spearmanr(y, oof)[0], 3)}
    return model, fidelity, imp.reset_index(drop=True)


def plot_surrogate(tree, features, fidelity):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    plot_tree(tree, feature_names=[label(f) for f in features], filled=False, rounded=True,
              impurity=False, precision=2, fontsize=9, ax=ax)
    r2 = max(fidelity['r2_cv_by_session'], 0.0)
    ax.set_title(f"TabICL mitigé résumé par un arbre de profondeur 3 — fidélité R² = {r2:.2f} sur des sessions "
                 f"non vues\n« value » = P(oui) moyenne prédite par TabICL dans la branche", loc="left",
                 fontsize=12, fontweight="bold")
    return _save(fig, "09_tabicl_surrogate_tree")


# --------------------------------------------------------------------------- #
#  7. Structure du problème : sélectivité de A et popularité de B             #
# --------------------------------------------------------------------------- #
def pickiness_popularity_bound(test):
    """Borne haute illustrative (pas un modèle déployable) : si l'on connaissait la sélectivité
    de A et la popularité de B, estimées en leave-one-out sur les AUTRES rendez-vous de la
    soirée (la ligne prédite est exclue), quelle AUC atteindrait-on ?"""
    t = test.copy()
    for key, name in (("iid", "a_loo"), ("pid", "b_loo")):
        s = t.groupby(key).dec.transform("sum"); n = t.groupby(key).dec.transform("count")
        t[name] = (s - t.dec) / (n - 1)
    ok = t.a_loo.notna() & t.b_loo.notna()
    return {"auc_selectivite_A": round(roc_auc_score(t.dec[ok], t.a_loo[ok]), 3),
            "auc_popularite_B": round(roc_auc_score(t.dec[ok], t.b_loo[ok]), 3),
            "auc_A_plus_B": round(roc_auc_score(t.dec[ok], t.a_loo[ok] + t.b_loo[ok]), 3)}


def model_agreement(imp_xgb, logit_tab, surrogate_imp, perm, k=15):
    """Corrélation de rang entre les importances globales des différents modèles/méthodes, et
    variables « consensus » (dans le top-k d'au moins 3 des 4 méthodes)."""
    s = pd.DataFrame({
        "XGB · SHAP": imp_xgb.set_index("feature").mean_abs_shap,
        "XGB · permutation (AUC)": perm.set_index("feature").auc_drop,
        "Logit · |coef| std.": logit_tab.set_index("feature").coef_per_sd.abs(),
        "TabICL · surrogate SHAP": surrogate_imp.set_index("feature").importance,
    }).dropna()   # les 4 proxies et shar1_1 (absents du logit) sortent de la comparaison
    rho = s.corr(method="spearman").round(2)
    ranks = s.rank(ascending=False)
    in_top = (ranks <= k)
    consensus = pd.DataFrame({"label": [label(f) for f in s.index], "n_methods_topk": in_top.sum(1)},
                             index=s.index).join(ranks.astype(int))
    consensus = consensus[consensus.n_methods_topk >= 3].sort_values("n_methods_topk", ascending=False)
    return rho, consensus


def plot_agreement(rho):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#1c5cab"])
    ax.imshow(rho.values, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(rho)), rho.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(rho)), rho.index)
    for i in range(len(rho)):
        for j in range(len(rho)):
            v = rho.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else INK, fontsize=11)
    ax.grid(False)
    ax.set_title("Les méthodes s'accordent-elles ?\n(corrélation de rang des importances)")
    return _save(fig, "10_agreement_between_methods")


def gender_proxy_check(train, test, features, proba_cols):
    """Le genre n'est jamais en entrée, mais est-il reconstructible à partir des variables ?"""
    out = {}
    for side, target in (("B", "cand_female"), ("A", "gender_A")):
        fs = [f for f in features if f.endswith(f"_{side}") or f"_{side}_" in f]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000)).fit(train[fs], train[target])
        out[f"auc_devine_genre_{side}"] = round(roc_auc_score(test[target], m.predict_proba(test[fs])[:, 1]), 3)
    rates = {}
    for c in proba_cols:
        thr = test[c].median()   # l'app recommande la moitié des profils les mieux notés
        r = test.groupby("cand_female")[c].apply(lambda s: (s > thr).mean())
        rates[c] = {"femmes": round(r.get(1), 3), "hommes": round(r.get(0), 3)}
    rates["oui_reels"] = {"femmes": round(test[test.cand_female == 1].dec.mean(), 3),
                          "hommes": round(test[test.cand_female == 0].dec.mean(), 3)}
    out["taux_recommandation_top50pct"] = rates
    return out
