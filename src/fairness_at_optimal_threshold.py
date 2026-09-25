"""Re-evaluate TOST fairness at the P&L-optimal deployment threshold."""
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


def run() -> dict:
    data = pd.read_parquet(DATA_DIR / "clean.parquet")
    with open(DATA_DIR / "features.json") as f:
        contract = json.load(f)
    with open(DATA_DIR / "split.json") as f:
        split = json.load(f)
    with open(DATA_DIR / "mitigated_features.json") as f:
        mit_contract = json.load(f)

    FEATURES = contract["features"]
    MIT_FEATURES = mit_contract["features"]

    train_mask = data["wave"].isin(split["train_waves"])
    test_mask = data["wave"].isin(split["test_waves"])
    train = data[train_mask]
    test = data[test_mask]

    X_train = train[FEATURES]
    y_train = train["dec"].values
    X_test = test[FEATURES]

    X_train_mit = train[MIT_FEATURES]
    X_test_mit = test[MIT_FEATURES]

    pa_test = {
        "cand_female": test["cand_female"].values,
        "cand_race": test["cand_race"].values,
    }

    xgb_base = joblib.load(MODELS_DIR / "xgb.joblib")
    xgb_mit = joblib.load(MODELS_DIR / "xgb_mitigated.joblib")
    logit_base = joblib.load(MODELS_DIR / "logit.joblib")
    logit_mit = joblib.load(MODELS_DIR / "logit_mitigated.joblib")

    p_tr_xgb_base = xgb_base.predict_proba(X_train)[:, 1]
    p_tr_xgb_mit = xgb_mit.predict_proba(X_train_mit)[:, 1]
    p_tr_logit_base = logit_base.predict_proba(X_train)[:, 1]
    p_tr_logit_mit = logit_mit.predict_proba(X_train_mit)[:, 1]

    t_xgb_base, _ = optimize_threshold_pnl(y_train, p_tr_xgb_base)
    t_xgb_mit, _ = optimize_threshold_pnl(y_train, p_tr_xgb_mit)
    t_logit_base, _ = optimize_threshold_pnl(y_train, p_tr_logit_base)
    t_logit_mit, _ = optimize_threshold_pnl(y_train, p_tr_logit_mit)

    print(f"Optimal thresholds — XGB: base={t_xgb_base:.2f}, mit={t_xgb_mit:.2f} | "
          f"Logit: base={t_logit_base:.2f}, mit={t_logit_mit:.2f}")

    p_te_xgb_base = xgb_base.predict_proba(X_test)[:, 1]
    p_te_xgb_mit = xgb_mit.predict_proba(X_test_mit)[:, 1]
    p_te_logit_base = logit_base.predict_proba(X_test)[:, 1]
    p_te_logit_mit = logit_mit.predict_proba(X_test_mit)[:, 1]

    results = {}
    for name, proba, threshold in [
        ("xgb_baseline",   p_te_xgb_base,   t_xgb_base),
        ("xgb_mitigated",  p_te_xgb_mit,    t_xgb_mit),
        ("logit_baseline",  p_te_logit_base, t_logit_base),
        ("logit_mitigated", p_te_logit_mit,  t_logit_mit),
    ]:
        preds = (proba >= threshold).astype(int)
        tost = evaluate_fairness(preds, pa_test, delta=DELTA, alpha=ALPHA)
        exposure_rate = preds.mean()
        race = tost["details"]["race"]
        n_fair = sum(v["is_fair"] for v in race.values())
        n_total = len(race)

        results[name] = {
            "threshold": round(threshold, 2),
            "exposure_rate": round(float(exposure_rate), 4),
            "global_fairness": tost["global_fairness"],
            "n_racial_tests_passed": n_fair,
            "n_racial_tests_total": n_total,
            "tost_race": race,
        }

        print(f"\n{name.upper()} (t={threshold:.2f}, exposure={exposure_rate:.1%}):")
        print(f"  Racial tests passed: {n_fair}/{n_total}")
        for k, v in race.items():
            tag = "OK  " if v["is_fair"] else "FAIL"
            print(f"  [{tag}] {k:25s}: gap={v['gap']*100:+.2f}pp  n_comp={v['n_comp']}")

    with open(DATA_DIR / "fairness_optimal_threshold.json", "w") as f:
        json.dump(results, f, indent=2)

    groups_order = ["caucasian_vs_asian", "caucasian_vs_latino",
                    "caucasian_vs_black", "caucasian_vs_other"]
    labels = ["vs Asian", "vs Latino", "vs Black", "vs Other"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle(
        "TOST Racial Fairness — Seuil de Deploiement Optimal (delta=10pp)",
        fontsize=13, fontweight="bold"
    )

    COLORS = {
        "xgb_baseline":   ("#ff7f0e", "XGB Baseline"),
        "xgb_mitigated":  ("#2ca02c", "XGB Mitigue"),
        "logit_baseline":  ("#1f77b4", "Logit Baseline"),
        "logit_mitigated": ("#9467bd", "Logit Mitigue"),
    }

    for ax_idx, (ax, model_pair) in enumerate(zip(
        axes,
        [("xgb_baseline", "xgb_mitigated"), ("logit_baseline", "logit_mitigated")]
    )):
        ax.axvspan(-DELTA * 100, DELTA * 100,
                   color="#d4edda", alpha=0.45, label="Zone equivalence +/-10 pp")
        ax.axvline(0, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(-DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)
        ax.axvline( DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)

        y_pos = np.arange(len(groups_order))
        bar_width = 0.35

        for b_idx, model_key in enumerate(model_pair):
            offset = -bar_width / 2 + b_idx * bar_width
            color, lbl = COLORS[model_key]
            gaps = [results[model_key]["tost_race"][g]["gap"] * 100
                    for g in groups_order]
            fair_flags = [results[model_key]["tost_race"][g]["is_fair"]
                          for g in groups_order]
            t_val = results[model_key]["threshold"]

            ax.barh(y_pos + offset, gaps, bar_width,
                    label=f"{lbl} (t={t_val:.2f})", color=color, alpha=0.85)

            for i, (g, is_fair) in enumerate(zip(gaps, fair_flags)):
                marker = "OK" if is_fair else "X"
                ha = "left" if g >= 0 else "right"
                nudge = 0.4 if g >= 0 else -0.4
                ax.text(g + nudge, y_pos[i] + offset, marker,
                        va="center", ha=ha, fontsize=9, fontweight="bold",
                        color="#155724" if is_fair else "#721c24")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Ecart d'exposition (pp)", fontsize=11)
        model_name = "XGBoost" if ax_idx == 0 else "Logit"
        ax.set_title(f"{model_name} — seuil optimal P&L", fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
        ax.set_xlim(-25, 20)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = REPORTS_DIR / "tost_optimal_threshold.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")
    return results


if __name__ == "__main__":
    run()
