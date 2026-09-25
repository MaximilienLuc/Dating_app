"""HEC Match — Decomposition Biais Societal vs Biais Algorithmique.

Distingue deux niveaux de biais raciaux :
  1. BIAIS SOCIÉTAL : écart des taux de oui réels (y_true) par groupe
     racial — reflète les préférences humaines documentées par Fisman &
     Iyengar (Columbia 2002-2004), indépendamment de tout algorithme.
  2. BIAIS ALGORITHMIQUE : biais_algo(g) = gap_modele(g) - gap_données(g).
     Ce que le modèle ajoute (ou retire) au-delà du biais sociétal.

Modèles évalués : XGBoost baseline, Logit baseline, TabICL (proba
pré-calculées sur GPU Colab — models/tabicl.joblib non portable en CPU).

Sorties :
  - data/fairness_bias_decomposition.json
  - reports/fairness/bias_decomposition.png
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import evaluate_fairness
from src.mitigation import optimize_threshold_pnl

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "fairness"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 0.10
ALPHA = 0.05
RACE_LABELS = {1: "black", 2: "caucasian", 3: "latino", 4: "asian", 6: "other"}
CAUCASIAN_CODE = 2


def rates_by_race(values: np.ndarray, cand_race: np.ndarray) -> dict:
    out = {}
    for code, label in RACE_LABELS.items():
        mask = cand_race == code
        if mask.sum() > 0:
            out[label] = {"rate": float(values[mask].mean()), "n": int(mask.sum())}
    return out


def run() -> dict:
    # ---- Données ---------------------------------------------------------
    data = pd.read_parquet(DATA_DIR / "clean.parquet")
    with open(DATA_DIR / "features.json") as f:
        contract = json.load(f)
    with open(DATA_DIR / "split.json") as f:
        split = json.load(f)

    FEATURES = contract["features"]
    FEATURES_LOGIT = contract["features_logit"]

    train = data[data["wave"].isin(split["train_waves"])].copy()
    test = data[data["wave"].isin(split["test_waves"])].copy()

    y_train = train["dec"].values
    y_test = test["dec"].values
    race_test = test["cand_race"].values

    pa_test = {
        "cand_female": test["cand_female"].values,
        "cand_race": race_test,
    }

    # ---- TabICL (probas pré-calculées GPU Colab) -------------------------
    tabicl_full = pd.read_parquet(DATA_DIR / "tabicl_predictions.parquet")
    test_m = test.reset_index(drop=False).merge(
        tabicl_full[["iid", "pid", "wave", "tabicl_proba"]],
        on=["iid", "pid", "wave"], how="left",
    )
    p_tabicl_test = test_m["tabicl_proba"].values.astype(float)
    n_miss = np.isnan(p_tabicl_test).sum()
    if n_miss > 0:
        print(f"  [TabICL] {n_miss} valeurs manquantes → imputation médiane")
        p_tabicl_test = np.where(np.isnan(p_tabicl_test),
                                  np.nanmedian(p_tabicl_test), p_tabicl_test)

    # TabICL proba sur train pour calibrer t*
    train_m = train.reset_index(drop=False).merge(
        tabicl_full[["iid", "pid", "wave", "tabicl_proba"]],
        on=["iid", "pid", "wave"], how="left",
    )
    p_tabicl_train = train_m["tabicl_proba"].values.astype(float)
    p_tabicl_train = np.where(np.isnan(p_tabicl_train),
                               np.nanmedian(p_tabicl_train), p_tabicl_train)

    # ---- Modèles baseline ------------------------------------------------
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    logit = joblib.load(MODELS_DIR / "logit.joblib")

    p_xgb_test   = xgb.predict_proba(test[FEATURES])[:, 1]
    p_logit_test  = logit.predict_proba(test[FEATURES_LOGIT])[:, 1]

    # Seuils optimaux P&L (calibrés sur train)
    p_xgb_train  = xgb.predict_proba(train[FEATURES])[:, 1]
    p_logit_train = logit.predict_proba(train[FEATURES_LOGIT])[:, 1]
    t_xgb, _   = optimize_threshold_pnl(y_train, p_xgb_train)
    t_logit, _ = optimize_threshold_pnl(y_train, p_logit_train)
    t_tabicl, _ = optimize_threshold_pnl(y_train, p_tabicl_train)
    print(f"Seuils optimaux P&L : XGB={t_xgb:.2f} | Logit={t_logit:.2f} | TabICL={t_tabicl:.2f}")

    # ---- 1. BIAIS SOCIÉTAL -----------------------------------------------
    print("\n" + "=" * 65)
    print("NIVEAU 1 — BIAIS SOCIÉTAL (taux de oui dans les données brutes)")
    print("=" * 65)
    raw_rates = rates_by_race(y_test, race_test)
    cauc_rate_data = raw_rates["caucasian"]["rate"]
    for label, info in raw_rates.items():
        gap = (cauc_rate_data - info["rate"]) * 100
        print(f"  {label:12s}: {info['rate']:.1%}  gap_vs_cauc={gap:+.1f}pp  n={info['n']}")

    # ---- 2. BIAIS ALGORITHMIQUE à t=0.50 et t* ---------------------------
    group_keys = ["black", "latino", "asian", "other"]
    models_config = [
        ("XGBoost",  p_xgb_test,   0.50, t_xgb),
        ("Logit",    p_logit_test,  0.50, t_logit),
        ("TabICL",   p_tabicl_test, 0.50, t_tabicl),
    ]

    results = {
        "raw_rates_data": {k: v for k, v in raw_rates.items()},
        "models": {},
    }

    for mname, proba, t_std, t_opt in models_config:
        for t_label, threshold in [("t=0.50", t_std), (f"t*={t_opt:.2f}", t_opt)]:
            key = f"{mname} ({t_label})"
            y_pred = (proba >= threshold).astype(int)
            expo_rates = rates_by_race(y_pred, race_test)
            tost = evaluate_fairness(y_pred, pa_test, delta=DELTA, alpha=ALPHA)
            cauc_expo = expo_rates["caucasian"]["rate"]

            print(f"\n  [{key}] expo_cauc={cauc_expo:.1%}  "
                  f"global_fair={tost['global_fairness']}")

            grp_results = {}
            for label in group_keys:
                tost_key = f"caucasian_vs_{label}"
                gap_data  = (cauc_rate_data - raw_rates[label]["rate"]) * 100
                gap_model = (cauc_expo - expo_rates.get(label, {}).get("rate", float("nan"))) * 100
                bias_algo = gap_model - gap_data
                is_fair   = tost["details"]["race"].get(tost_key, {}).get("is_fair", None)
                grp_results[label] = {
                    "gap_data": round(gap_data, 2),
                    "gap_model": round(gap_model, 2),
                    "bias_algo": round(bias_algo, 2),
                    "is_fair_tost": is_fair,
                    "n": raw_rates[label]["n"],
                }
                tag = "OK  " if is_fair else "FAIL"
                print(f"    [{tag}] vs {label:8s}: data={gap_data:+.1f}pp | "
                      f"model={gap_model:+.1f}pp | algo={bias_algo:+.1f}pp")

            results["models"][key] = {
                "threshold": round(threshold, 2),
                "exposure_rate_cauc": round(cauc_expo, 4),
                "global_fairness": tost["global_fairness"],
                "groups": grp_results,
            }

    # Export JSON
    with open(DATA_DIR / "fairness_bias_decomposition.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExported: {DATA_DIR / 'fairness_bias_decomposition.json'}")

    # ---- Visualisation ---------------------------------------------------
    # On plot uniquement les 3 modèles au seuil 0.50 pour la lisibilité
    plot_models = {
        "XGBoost": ("XGBoost (t=0.50)", "#ff7f0e"),
        "Logit":   ("Logit (t=0.50)",   "#1f77b4"),
        "TabICL":  ("TabICL (t=0.50)",  "#e377c2"),
    }
    glabels = ["vs Black", "vs Latino", "vs Asian", "vs Other"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    fig.suptitle(
        "Décomposition du Biais Racial : Sociétal vs Algorithmique\n"
        "HEC Match — Speed Dating (Fisman & Iyengar, Columbia 2002-2004)",
        fontsize=13, fontweight="bold"
    )

    # ---- Plot A : biais total ----------------------------------------
    ax = axes[0]
    ax.axvspan(-DELTA * 100, DELTA * 100, color="#d4edda", alpha=0.4,
               label="Zone équivalence ±10 pp (TOST)")
    ax.axvline(0, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(-DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)
    ax.axvline( DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)

    y_pos = np.arange(len(group_keys))
    # Biais sociétal (fond gris hachuré)
    gaps_data = [(cauc_rate_data - raw_rates[g]["rate"]) * 100 for g in group_keys]
    ax.barh(y_pos, gaps_data, 0.55, label="Biais sociétal (données brutes)",
            color="#adb5bd", alpha=0.65, hatch="//", edgecolor="white", linewidth=0.5)

    bar_h = 0.14
    offsets = [-0.22, 0.0, 0.22]
    for (mlabel, (mkey, mcolor)), offset in zip(plot_models.items(), offsets):
        gaps_m = [results["models"][mkey]["groups"][g]["gap_model"] for g in group_keys]
        fair_f = [results["models"][mkey]["groups"][g]["is_fair_tost"] for g in group_keys]
        bars = ax.barh(y_pos + offset, gaps_m, bar_h, label=mlabel, color=mcolor, alpha=0.88)
        for i, (g, is_f) in enumerate(zip(gaps_m, fair_f)):
            if is_f:
                ax.scatter(g, y_pos[i] + offset, marker="*", s=55,
                           color="#155724", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(glabels, fontsize=11)
    ax.set_xlabel("Écart d'exposition caucasien − groupe (pp)", fontsize=10)
    ax.set_title("Écart total\n(données brutes vs prédictions à t=0.50)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.set_xlim(-28, 22)
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.98, 0.02, "★ = test TOST passé (équivalent)", transform=ax.transAxes,
            ha="right", fontsize=8, color="#155724", style="italic")

    # ---- Plot B : biais algorithmique pur --------------------------------
    ax2 = axes[1]
    ax2.axvline(0, color="#6c757d", linestyle="--", linewidth=1.5, alpha=0.8,
                label="0 pp = modèle neutre")
    ax2.axvspan(-5, 5, color="#fff3cd", alpha=0.35, label="±5 pp (biais faible)")

    for (mlabel, (mkey, mcolor)), offset in zip(plot_models.items(), offsets):
        bias_algo = [results["models"][mkey]["groups"][g]["bias_algo"] for g in group_keys]
        ax2.barh(y_pos + offset, bias_algo, bar_h, label=mlabel, color=mcolor, alpha=0.88)
        for i, b in enumerate(bias_algo):
            ha = "left" if b >= 0 else "right"
            nudge = 0.25 if b >= 0 else -0.25
            ax2.text(b + nudge, y_pos[i] + offset, f"{b:+.1f}",
                     va="center", ha=ha, fontsize=8, color=mcolor, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(glabels, fontsize=11)
    ax2.set_xlabel("Biais algo pur (pp) = écart modèle − écart données brutes", fontsize=10)
    ax2.set_title("Biais algorithmique pur\n(ce que le modèle ajoute au-delà du biais sociétal)",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax2.set_xlim(-18, 18)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = REPORTS_DIR / "bias_decomposition.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    return results


if __name__ == "__main__":
    run()
