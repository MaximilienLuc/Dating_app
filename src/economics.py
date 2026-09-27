"""
HEC Match — valeur économique des modèles (bloc Alex), en complément de src/performance.py.

Question : un modèle rapporte-t-il plus qu'une stratégie naïve ?
  1. Stratégies de référence : montrer tout le monde, ne montrer personne, oracle.
     Les P&L des 6 modèles (seuils optimisés par GroupKFold sur le train dans
     reports/performance/summary_all_models.csv) sont comparés à ces références, avec un
     intervalle de confiance par bootstrap apparié de SESSIONS de test.
  2. Seuil de rentabilité théorique : montrer un profil rapporte en espérance
     p·X − (1−p)·Y, ne pas le montrer coûte p·Z (occasion manquée). On montre donc si
     p > p* = Y / (X + Y + Z). Si le taux de oui dépasse p* partout, montrer tout le monde
     est optimal et aucun modèle ne peut faire mieux.
  3. Sensibilité aux coûts : valeur ajoutée des modèles (règle de décision p > p*, aucun
     réglage sur le test) quand le coût d'une mauvaise recommandation Y varie.
  4. Contrainte de capacité : une app ne montre que k profils par utilisateur. Le modèle
     sert alors à CLASSER les candidats de chaque utilisateur (précision@k, AUC intra-utilisateur).

Conventions de coûts identiques à src/performance.calculate_pnl (source de vérité :
data/economic_assumptions.json). Tout est évalué sur l'échantillon TEST.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.build_dataset import DATA_DIR, SEED
from src.performance import (calculate_pnl, load_economic_assumptions,
                             load_mitigated_predictions, load_model_predictions,
                             split_train_test)

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})

ROOT = DATA_DIR.parent
OUT_DIR = ROOT / "reports" / "economics"
MODELS = ["logit", "xgb", "tabicl", "logit_mitigated", "xgb_mitigated", "tabicl_mitigated"]
MITIGATED = ["logit_mitigated", "xgb_mitigated", "tabicl_mitigated"]
NAMES = {"logit": "Logit", "xgb": "XGBoost", "tabicl": "TabICL",
         "logit_mitigated": "Logit mitigé", "xgb_mitigated": "XGBoost mitigé",
         "tabicl_mitigated": "TabICL mitigé"}
MODEL_COLORS = {"logit_mitigated": BLUE, "xgb_mitigated": ORANGE, "tabicl_mitigated": AQUA}
N_BOOT = 2000


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)
    return OUT_DIR / f"{name}.png"


def load_context():
    data, _ = load_model_predictions()
    data, _ = load_mitigated_predictions(data)
    train, test, _ = split_train_test(data)
    a = load_economic_assumptions()
    thresholds = pd.read_csv(ROOT / "reports" / "performance" / "summary_all_models.csv"
                             ).set_index("model").optimal_threshold.to_dict()
    return {"train": train, "test": test, "X": a["X"], "Y": a["Y"], "Z": a.get("Z", 0.0),
            "thresholds": thresholds}


def break_even(X, Y, Z):
    """Probabilité de oui au-delà de laquelle montrer un profil est rentable en espérance."""
    return Y / (X + Y + Z)


# --------------------------------------------------------------------------- #
#  Bootstrap apparié de sessions                                              #
# --------------------------------------------------------------------------- #
def session_bootstrap(test, wave_stats, combine, n_boot=N_BOOT, seed=SEED):
    """Bootstrap apparié de SESSIONS de test (même logique que la stabilité de Rémi).

    wave_stats(df_session) -> vecteur de sommes additives pour une session (ex. P&L total,
    nombre de lignes) ; combine(vecteur sommé) -> statistique. Chaque tirage rééchantillonne
    les sessions avec remise, ce qui revient à pondérer les vecteurs par des comptes
    multinomiaux : rapide et exact. Renvoie (valeur observée, IC 2,5 %, IC 97,5 %)."""
    waves = np.sort(test.wave.unique())
    S = np.array([wave_stats(test[test.wave == w]) for w in waves], dtype=float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(waves), np.full(len(waves), 1 / len(waves)), size=n_boot)
    sums = counts @ S
    boots = np.array([combine(v) for v in sums])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return combine(S.sum(0)), lo, hi


# --------------------------------------------------------------------------- #
#  1. Modèles vs stratégies naïves                                            #
# --------------------------------------------------------------------------- #
def naive_pnl(y, X, Y, Z):
    y = np.asarray(y)
    P, N = int(y.sum()), int(len(y) - y.sum())
    return {"Montrer tout le monde": X * P - Y * N, "Ne montrer personne": -Z * P,
            "Oracle (parfait)": X * P}


def policy_table(test, X, Y, Z, thresholds):
    """P&L test de chaque modèle à son seuil optimisé (GroupKFold sur train, Blanquette) et à
    la règle théorique p > p*, comparé à « montrer tout le monde »."""
    n = len(test)
    per1000 = 1000 / n
    p_star = break_even(X, Y, Z)
    show_all = lambda d: calculate_pnl(d.dec, np.ones(len(d)), 0.5, X, Y, Z)
    rows = []
    for name, v in naive_pnl(test.dec, X, Y, Z).items():
        rows.append({"strategie": name, "seuil": np.nan, "part_montree": np.nan,
                     "pnl_test": v, "pnl_pour_1000": v * per1000,
                     "vs_montrer_tout_pour_1000": (v - show_all(test)) * per1000,
                     "ic_bas": np.nan, "ic_haut": np.nan})
    rows[0]["part_montree"], rows[1]["part_montree"] = 1.0, 0.0
    rows[2]["part_montree"] = test.dec.mean()
    for m in MODELS:
        col = f"proba_{m}"
        for label, t in (("seuil optimisé (CV train)", thresholds[m]), ("règle p > p*", p_star)):
            ws = lambda d, t=t: [calculate_pnl(d.dec, d[col], t, X, Y, Z) - show_all(d), len(d)]
            obs, lo, hi = session_bootstrap(test, ws, lambda v: v[0] * 1000 / v[1])
            pnl = calculate_pnl(test.dec, test[col], t, X, Y, Z)
            rows.append({"strategie": f"{NAMES[m]} — {label}", "seuil": t,
                         "part_montree": float((test[col] > t).mean()), "pnl_test": pnl,
                         "pnl_pour_1000": pnl * per1000, "vs_montrer_tout_pour_1000": obs,
                         "ic_bas": lo, "ic_haut": hi})
    return pd.DataFrame(rows)


def yes_rate_by_decile(test, models=MITIGATED, q=10):
    out = {}
    for m in models:
        dec = pd.qcut(test[f"proba_{m}"].rank(method="first"), q, labels=False) + 1
        out[m] = test.groupby(dec).dec.mean()
    return pd.DataFrame(out)


def plot_break_even(rates, p_star, base_rate):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = rates.index.values
    for m in rates:
        ax.plot(x, rates[m] * 100, marker="o", markersize=7, linewidth=2, color=MODEL_COLORS[m], label=NAMES[m])
    ax.axhline(p_star * 100, color=RED, linewidth=1.6, linestyle="--")
    ax.text(10.15, p_star * 100, f"seuil de rentabilité\np* = {p_star:.0%}", color=RED, va="center", fontsize=10)
    ax.axhline(base_rate * 100, color=INK2, linewidth=1, linestyle=":")
    ax.text(10.15, base_rate * 100, f"taux de oui\nmoyen {base_rate:.0%}", color=INK2, va="center", fontsize=10)
    ax.set_xticks(x, [f"D{i}" for i in x])
    ax.set_xlabel("Décile du score du modèle (D1 = les 10 % de couples jugés les moins prometteurs)")
    ax.set_ylabel("Taux de « oui » réel")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, max(70, rates.values.max() * 100 + 5))
    ax.set_xlim(0.6, 10.4)
    ax.set_title("Seuls les 10 % de couples les moins prometteurs passent sous le seuil de rentabilité")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "01_taux_oui_par_decile_vs_seuil")


def plot_policy_bars(table, X, Y, Z):
    t = table[table.strategie.str.contains("seuil optimisé")].copy()
    t["nom"] = t.strategie.str.replace(" — seuil optimisé (CV train)", "", regex=False)
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [BLUE if v >= 0 else RED for v in t.vs_montrer_tout_pour_1000]
    ax.barh(t.nom, t.vs_montrer_tout_pour_1000, color=colors, height=0.55)
    ax.errorbar(t.vs_montrer_tout_pour_1000, np.arange(len(t)),
                xerr=[t.vs_montrer_tout_pour_1000 - t.ic_bas, t.ic_haut - t.vs_montrer_tout_pour_1000],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
    ax.axvline(0, color=INK, linewidth=1)
    for i, (v, s) in enumerate(zip(t.vs_montrer_tout_pour_1000, t.part_montree)):
        ax.text(ax.get_xlim()[1], i, f"  montre {s:.0%}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("€ gagnés en plus de « montrer tout le monde », pour 1 000 profils (IC 95 %, bootstrap de sessions)")
    ax.set_title(f"Valeur ajoutée des modèles avec les coûts actuels (+{X:g} € / −{Y:g} € / −{Z:g} €)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "02_valeur_ajoutee_vs_montrer_tout")


# --------------------------------------------------------------------------- #
#  2. Sensibilité au coût d'une mauvaise recommandation                       #
# --------------------------------------------------------------------------- #
def cost_sensitivity(test, X, Z, Y_grid=None, models=MITIGATED, n_boot=N_BOOT):
    """Pour chaque Y : règle de décision p > p*(Y) (pas de réglage sur le test), valeur ajoutée
    vs la MEILLEURE stratégie naïve (tout montrer ou ne rien montrer), pour 1 000 profils."""
    if Y_grid is None:
        Y_grid = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6]
    rows = []
    for Y in Y_grid:
        p_star = break_even(X, Y, Z)

        for m in models:
            col = f"proba_{m}"
            ws = lambda d: [calculate_pnl(d.dec, d[col], p_star, X, Y, Z),
                            calculate_pnl(d.dec, np.ones(len(d)), 0.5, X, Y, Z),
                            calculate_pnl(d.dec, np.zeros(len(d)), 0.5, X, Y, Z), len(d)]
            obs, lo, hi = session_bootstrap(test, ws, lambda v: (v[0] - max(v[1], v[2])) * 1000 / v[3],
                                            n_boot=n_boot)
            rows.append({"Y": Y, "p_star": p_star, "model": m, "valeur_ajoutee_pour_1000": obs,
                         "ic_bas": lo, "ic_haut": hi, "part_montree": float((test[col] > p_star).mean())})
    return pd.DataFrame(rows)


def plot_cost_sensitivity(sens, X, Y_current, Z):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for m in MITIGATED:
        g = sens[sens.model == m]
        ax.plot(g.p_star * 100, g.valeur_ajoutee_pour_1000, marker="o", color=MODEL_COLORS[m], linewidth=2, label=NAMES[m])
        ax.fill_between(g.p_star * 100, g.ic_bas, g.ic_haut, color=MODEL_COLORS[m], alpha=0.10, linewidth=0)
    ax.axhline(0, color=INK, linewidth=1)
    cur = break_even(X, Y_current, Z) * 100
    ax.axvline(cur, color=INK2, linestyle="--", linewidth=1)
    ax.text(cur, ax.get_ylim()[1], f" hypothèses actuelles\n p* = {cur:.0f} %", color=INK2, va="top", fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_xlabel(f"Seuil de rentabilité p* = Y / (X + Y + Z), avec X = {X:g} € et Z = {Z:g} €, Y variable\n"
                  "(à droite : une mauvaise recommandation coûte cher)")
    ax.set_ylabel("€ en plus de la meilleure stratégie naïve\npour 1 000 profils")
    ax.set_title("Le modèle crée de la valeur surtout quand une mauvaise recommandation coûte cher")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "03_sensibilite_cout_mauvaise_reco")


# --------------------------------------------------------------------------- #
#  3. Contrainte de capacité : k profils par utilisateur                      #
# --------------------------------------------------------------------------- #
def precision_at_k(test, col, k):
    """Part de « oui » parmi les k candidats les mieux classés de chaque utilisateur A,
    moyennée sur les utilisateurs. col=None -> hasard (espérance = taux de oui de A)."""
    return float(np.mean(_user_precisions(test, col, k)))


def _user_precisions(test, col, k):
    vals = []
    for _, g in test.groupby("iid"):
        kk = min(k, len(g))
        if col is None:
            vals.append(g.dec.mean())
        elif col == "oracle":
            vals.append(min(kk, g.dec.sum()) / kk)
        else:
            vals.append(g.nlargest(kk, col).dec.mean())
    return vals


def within_user_auc(test, col):
    aucs = [roc_auc_score(g.dec, g[col]) for _, g in test.groupby("iid") if g.dec.nunique() == 2]
    return float(np.mean(aucs))


def topk_table(test, X, Y, ks=(1, 2, 3, 5), models=MODELS, n_boot=N_BOOT):
    """Précision@k par stratégie, gain vs hasard (points, IC par sessions) et traduction en
    euros pour 1 000 profils MONTRÉS : X par oui, −Y par non. Sous contrainte de capacité, le
    nombre de profils montrés est fixé : l'occasion manquée Z ne départage pas les stratégies."""
    rows = []
    for k in ks:
        rnd = precision_at_k(test, None, k)
        orc = precision_at_k(test, "oracle", k)
        rows.append({"k": k, "strategie": "Hasard", "precision": rnd, "gain_vs_hasard_pp": 0.0, "ic_bas": np.nan, "ic_haut": np.nan})
        rows.append({"k": k, "strategie": "Oracle", "precision": orc, "gain_vs_hasard_pp": (orc - rnd) * 100, "ic_bas": np.nan, "ic_haut": np.nan})
        for m in models:
            col = f"proba_{m}"
            ws = lambda d: [np.sum(_user_precisions(d, col, k)), np.sum(_user_precisions(d, None, k)),
                            d.iid.nunique()]
            obs, lo, hi = session_bootstrap(test, ws, lambda v: (v[0] - v[1]) / v[2] * 100, n_boot=n_boot)
            rows.append({"k": k, "strategie": NAMES[m], "precision": precision_at_k(test, col, k),
                         "gain_vs_hasard_pp": obs, "ic_bas": lo, "ic_haut": hi})
    tab = pd.DataFrame(rows)
    tab["eur_pour_1000_montres"] = (X * tab.precision - Y * (1 - tab.precision)) * 1000
    tab["gain_eur_pour_1000_vs_hasard"] = (X + Y) * tab.gain_vs_hasard_pp * 10
    tab["gain_eur_ic_bas"] = (X + Y) * tab.ic_bas * 10
    tab["gain_eur_ic_haut"] = (X + Y) * tab.ic_haut * 10
    return tab


def plot_topk(tab, k_focus=3):
    t = tab[(tab.k == k_focus) & tab.strategie.isin(["Hasard"] + [NAMES[m] for m in MITIGATED] + ["Oracle"])]
    order = ["Hasard"] + [NAMES[m] for m in MITIGATED] + ["Oracle"]
    t = t.set_index("strategie").loc[order]
    colors = ["#a3a29e"] + [MODEL_COLORS[m] for m in MITIGATED] + [INK2]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(t.index, t.precision * 100, color=colors, width=0.6)
    for i, (s, row) in enumerate(t.iterrows()):
        txt = f"{row.precision:.0%}"
        if s not in ("Hasard", "Oracle"):
            txt += f"\n+{row.gain_vs_hasard_pp:.1f} pts\n[{row.ic_bas:+.1f} ; {row.ic_haut:+.1f}]"
        ax.text(i, row.precision * 100 + 1.5, txt, ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("Part de « oui » parmi les profils montrés")
    ax.axhline(t.loc["Hasard"].precision * 100, color="#a3a29e", linewidth=1, linestyle=":")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 115)
    ax.set_title(f"Si l'app ne montre que {k_focus} profils par utilisateur (classement par le modèle)")
    ax.text(0, -0.2, "Entre crochets : IC 95 % du gain par rapport au hasard (bootstrap de sessions de test)",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.grid(axis="x", visible=False)
    return _save(fig, f"04_top{k_focus}_par_utilisateur")


def plot_topk_euros(tab, k_focus=3):
    order = ["Hasard"] + [NAMES[m] for m in MITIGATED] + ["Oracle"]
    t = tab[tab.k == k_focus].set_index("strategie").loc[order]
    colors = ["#a3a29e"] + [MODEL_COLORS[m] for m in MITIGATED] + [INK2]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(t.index, t.eur_pour_1000_montres, color=colors, width=0.6)
    base = t.loc["Hasard"].eur_pour_1000_montres
    for i, (s, row) in enumerate(t.iterrows()):
        txt = f"{row.eur_pour_1000_montres:.0f} €"
        if s != "Hasard":
            txt += f"\n({row.eur_pour_1000_montres / base - 1:+.0%})"
        ax.text(i, row.eur_pour_1000_montres + 15, txt, ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("€ pour 1 000 profils montrés")
    ax.set_ylim(0, t.eur_pour_1000_montres.max() * 1.25)
    ax.set_title(f"Ce que rapporte le classement, à {k_focus} profils montrés par utilisateur")
    ax.grid(axis="x", visible=False)
    return _save(fig, f"05_top{k_focus}_euros")


def plot_topk_curve(tab):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for m in MITIGATED:
        g = tab[tab.strategie == NAMES[m]]
        ax.plot(g.k, g.precision * 100, marker="o", color=MODEL_COLORS[m], linewidth=2, label=NAMES[m])
    for s, c, ls in (("Hasard", "#a3a29e", ":"), ("Oracle", INK2, "--")):
        g = tab[tab.strategie == s]
        ax.plot(g.k, g.precision * 100, color=c, linestyle=ls, linewidth=1.5, label=s)
    ax.set_xticks(sorted(tab.k.unique()))
    ax.set_xlabel("Nombre de profils montrés par utilisateur (k)")
    ax.set_ylabel("Part de « oui » parmi les profils montrés")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_title("Précision@k selon le nombre de profils montrés")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(30, 100)
    return _save(fig, "06_precision_at_k")
