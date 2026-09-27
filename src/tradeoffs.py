"""
HEC Match — synthèse des arbitrages (bloc Alex) : 6 modèles × 4 dimensions, mesurés de façon HOMOGÈNE.

Pourquoi recalculer ? Les blocs existants utilisent des protocoles différents selon les modèles
(stabilité : modèles de base seulement ; fairness : seuils différents par modèle). Ici, une seule
règle de décision pour tous : l'app montre à chaque utilisateur ses 3 candidats les mieux notés
(cf. 06_economics, c'est là que le modèle crée de la valeur).

- Stabilité : protocole de Rémi (src/stability.py) — N réentraînements sur des SESSIONS
  d'entraînement tirées avec remise, mêmes hyperparamètres v0, évaluation sur le même test.
  Mesures : dispersion de l'AUC, part de décisions qui basculent au seuil 0,5 (comparable à Rémi),
  et part des 3 profils recommandés à chaque utilisateur qui change (propre à l'app).
  Modèles mitigés : réentraînés sur le schéma de Max (mitigated_features.json, 160 variables).
  TabICL : non réentraînable hors GPU -> non mesuré.
- Fairness : taux d'exposition (être dans le top 3 de quelqu'un) par genre et origine du candidat,
  rapporté au groupe de référence (hommes / candidats blancs), comparé au même ratio calculé sur
  les « oui » réels. Ratio < 0,8 = seuil d'alerte usuel (règle des 4/5).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.build_dataset import DATA_DIR, SEED
from src.performance import (add_legacy_reference_dummies, load_mitigated_predictions,
                             load_model_predictions, split_train_test)
from src.stability import model_factory, session_sample

ROOT = DATA_DIR.parent
OUT_DIR = ROOT / "reports" / "tradeoffs"
MODELS = ["logit", "xgb", "tabicl", "logit_mitigated", "xgb_mitigated", "tabicl_mitigated"]
NAMES = {"logit": "Logit", "xgb": "XGBoost", "tabicl": "TabICL", "logit_mitigated": "Logit mitigé",
         "xgb_mitigated": "XGBoost mitigé", "tabicl_mitigated": "TabICL mitigé"}
RACE = {1: "Noirs", 2: "Blancs", 3: "Latinos", 4: "Asiatiques", 6: "Autres"}
K = 3


def load_context():
    data, fd = load_model_predictions()
    data, mf = load_mitigated_predictions(data)
    train, test, _ = split_train_test(data)
    feats = {"logit": fd["features_logit"], "xgb": fd["features"],
             "logit_mitigated": mf, "xgb_mitigated": mf}
    return {"train": train, "test": test, "features": feats}


def topk_mask(df, col, k=K):
    """Booléen : le candidat est dans les k mieux notés de son utilisateur A."""
    rank = df.groupby("iid")[col].rank(method="first", ascending=False)
    return (rank <= k).to_numpy()


# --------------------------------------------------------------------------- #
#  Stabilité                                                                  #
# --------------------------------------------------------------------------- #
def stability_refits(train, test, features, n_refits=100, seed=SEED, threshold=0.5):
    rng = np.random.default_rng(seed)
    draws = [session_sample(train.wave, rng) for _ in range(n_refits)]
    y, yt = train.dec.to_numpy(), test.dec.to_numpy()
    rows = []
    for name, feats in features.items():
        kind = "logit" if name.startswith("logit") else "xgb"
        ref = model_factory(kind, seed).fit(train[feats], y).predict_proba(test[feats])[:, 1]
        ref_top = topk_mask(test.assign(_p=ref), "_p")
        for b, idx in enumerate(draws):
            p = model_factory(kind, seed).fit(train[feats].iloc[idx], y[idx]).predict_proba(test[feats])[:, 1]
            top = topk_mask(test.assign(_p=p), "_p")
            rows.append({"model": name, "replicate": b, "auc": roc_auc_score(yt, p),
                         "flip_rate_05": float(np.mean((p >= threshold) != (ref >= threshold))),
                         "top3_change": float(1 - (top & ref_top).sum() / ref_top.sum()),
                         "spearman_scores": float(pd.Series(p).corr(pd.Series(ref), method="spearman"))})
    return pd.DataFrame(rows)


def stability_summary(refits):
    g = refits.groupby("model")
    return pd.DataFrame({
        "auc_p05": g.auc.quantile(0.05), "auc_p95": g.auc.quantile(0.95),
        "auc_ecart_p95_p05": g.auc.quantile(0.95) - g.auc.quantile(0.05),
        "bascule_seuil_05": g.flip_rate_05.mean(), "top3_change": g.top3_change.mean(),
        "correlation_scores": g.spearman_scores.mean()})


# --------------------------------------------------------------------------- #
#  Fairness d'exposition (top 3 par utilisateur)                              #
# --------------------------------------------------------------------------- #
def exposure_ratios(test, shown):
    """Taux d'exposition de chaque groupe / groupe de référence."""
    t = test.assign(_s=shown)
    out = {}
    g = t.groupby("cand_female")._s.mean()
    out["Femmes / hommes"] = g.get(1) / g.get(0)
    r = t.groupby("cand_race")._s.mean()
    for code, lab in RACE.items():
        if code != 2 and code in r:
            out[f"{lab} / Blancs"] = r[code] / r[2]
    return out


def fairness_table(test, models=MODELS):
    rows = {"Décisions réelles (oui)": exposure_ratios(test, test.dec.to_numpy() == 1)}
    for m in models:
        rows[NAMES[m]] = exposure_ratios(test, topk_mask(test, f"proba_{m}"))
    return pd.DataFrame(rows).T


def auc_ci(test, col, n_boot=2000, seed=SEED):
    """IC de l'AUC par bootstrap de sessions de test (comme Rémi)."""
    rng = np.random.default_rng(seed)
    y, p = test.dec.to_numpy(), test[col].to_numpy()
    vals = [roc_auc_score(y[i], p[i]) for i in (session_sample(test.wave, rng) for _ in range(n_boot))]
    return np.percentile(vals, [2.5, 97.5])


def exposure_ratio_ci(test, shown, group_col, group, ref, n_boot=2000, seed=SEED):
    """Ratio d'exposition groupe / référence avec IC par bootstrap de sessions de test."""
    waves = np.sort(test.wave.unique())
    t = test.assign(_s=shown)
    S = np.array([[t[(t.wave == w) & (t[group_col] == group)]._s.sum(), ((t.wave == w) & (t[group_col] == group)).sum(),
                   t[(t.wave == w) & (t[group_col] == ref)]._s.sum(), ((t.wave == w) & (t[group_col] == ref)).sum()]
                  for w in waves], dtype=float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(waves), np.full(len(waves), 1 / len(waves)), size=n_boot)
    sums = counts @ S
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = (sums[:, 0] / sums[:, 1]) / (sums[:, 2] / sums[:, 3])
    tot = S.sum(0)
    return (tot[0] / tot[1]) / (tot[2] / tot[3]), *np.nanpercentile(boots, [2.5, 97.5])


def fairness_ci_table(test, models=MODELS, groups=((4, "Asiatiques"), (3, "Latinos"), (1, "Noirs"))):
    rows = []
    policies = {"Décisions réelles (oui)": test.dec.to_numpy() == 1}
    policies.update({NAMES[m]: topk_mask(test, f"proba_{m}") for m in models})
    for name, shown in policies.items():
        for code, lab in groups:
            r, lo, hi = exposure_ratio_ci(test, shown, "cand_race", code, 2)
            rows.append({"strategie": name, "groupe": lab, "ratio": r, "ic_bas": lo, "ic_haut": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Tableau de synthèse et graphiques                                          #
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GOOD, WARN, BAD, NA = "#d7eed7", "#fbecc9", "#f7d6d3", "#efeeea"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)
    return OUT_DIR / f"{name}.png"


def distortion(ratios):
    """Écart moyen à la parité : moyenne des |log(ratio)| sur les groupes (0 = parité parfaite)."""
    return float(np.mean(np.abs(np.log(np.asarray(ratios, dtype=float)))))


def plot_fairness_forest(fc):
    order = ["Décisions réelles (oui)", "Logit", "Logit mitigé", "XGBoost", "XGBoost mitigé", "TabICL", "TabICL mitigé"]
    groups = ["Asiatiques", "Latinos", "Noirs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, g in zip(axes, groups):
        d = fc[fc.groupe == g].set_index("strategie").loc[order].iloc[::-1]
        y = np.arange(len(d))
        signif = (d.ic_haut < 1) | (d.ic_bas > 1)
        cols = [INK2 if s.startswith("Décisions") else (RED if sg else BLUE) for s, sg in zip(d.index, signif)]
        ax.hlines(y, d.ic_bas, d.ic_haut, color=cols, linewidth=2)
        ax.scatter(d.ratio, y, color=cols, s=45, zorder=3)
        ax.axvline(1, color=INK, linewidth=1)
        ax.axvspan(0.8, 1.25, color=GRID, alpha=0.5, zorder=0)
        ax.set_yticks(y, d.index)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2], ["×0,25", "×0,5", "×1", "×2"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.1, 3)
        ax.set_title(f"Candidats {g.lower()} / blancs", fontsize=12)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Chances d'être dans le top 3 d'un utilisateur, par origine du candidat (IC 95 %, sessions)\n"
                 "zone grise : règle des 4/5 · rouge : écart significatif à la parité", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, "02_fairness_exposition_top3")


def plot_stability(summary):
    order = ["logit", "logit_mitigated", "xgb", "xgb_mitigated"]
    s = summary.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    labels = [NAMES[m] for m in order][::-1]
    colors = [BLUE, BLUE, ORANGE, ORANGE][::-1]
    for ax, col, title in ((axes[0], "bascule_seuil_05", "Décisions qui basculent (seuil 0,5)"),
                           (axes[1], "top3_change", "Profils du top 3 qui changent")):
        v = s[col].iloc[::-1] * 100
        ax.barh(labels, v, color=colors, height=0.55)
        for i, x in enumerate(v):
            ax.text(x + 0.8, i, f"{x:.0f} %", va="center", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max(v) * 1.25)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Stabilité : réentraîné sur d'autres sessions (100 tirages), le modèle change-t-il d'avis ?\n"
                 "TabICL : non mesurable (réentraînement impossible hors GPU)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, "03_stabilite")


def plot_scorecard(card):
    """card : DataFrame index = modèles, colonnes = critères ; chaque cellule (texte, niveau)."""
    colors = {"bon": GOOD, "moyen": WARN, "faible": BAD, "na": NA}
    n_r, n_c = card.shape
    fig, ax = plt.subplots(figsize=(2.35 * n_c + 1.8, 0.62 * n_r + 1.2))
    ax.set_xlim(0, n_c); ax.set_ylim(0, n_r); ax.invert_yaxis(); ax.axis("off")
    for j, c in enumerate(card.columns):
        ax.text(j + 0.5, -0.15, c, ha="center", va="bottom", fontweight="bold", fontsize=10.5)
    for i, m in enumerate(card.index):
        ax.text(-0.05, i + 0.5, m, ha="right", va="center", fontweight="bold", fontsize=11)
        for j, c in enumerate(card.columns):
            txt, lvl = card.loc[m, c]
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, color=colors[lvl], linewidth=0))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0, n_r + 0.35, "Vert : meilleur du lot · Jaune : intermédiaire · Rouge : point faible · Gris : non mesuré",
            fontsize=9, color=INK2)
    return _save(fig, "01_tableau_arbitrages")
