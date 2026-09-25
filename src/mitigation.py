"""HEC Match — Bias Mitigation via Proxy Removal using FPDP (Option A).

Identifies candidate proxy variables for cand_race using Fairness Partial
Dependence Plots (FPDP) on the training set, drops them from the feature space,
retrains Logit and XGBoost models, and evaluates the Fairness-Utility Trade-off
(AUC, P&L, TOST racial equivalence).

Exports:
- models/xgb_mitigated.joblib
- models/logit_mitigated.joblib
- data/mitigated_features.json
- data/mitigation_metrics.json
- reports/fairness/mitigation_tradeoff.png
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.metrics import evaluate_fairness

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "fairness"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DELTA = 0.10  # 10 pp equivalence tolerance
ALPHA = 0.05  # 5% significance level for each unilateral test

# HEC Match P&L cost matrix parameters:
GAIN_TP = 2.0   # +2€: profil recommandé qui reçoit un oui
COST_FP = -1.0  # -1€: profil recommandé qui reçoit un non (friction/déception)
COST_FN = -1.0  # -1€: profil caché qui aurait reçu un oui (manque à gagner)


def compute_pnl(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gain_tp: float = GAIN_TP,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> float:
    """Calculate total platform P&L for a given binary decision vector."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp * gain_tp + fp * cost_fp + fn * cost_fn)


def optimize_threshold_pnl(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the decision threshold that maximizes total P&L on the training set."""
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.70, 51)
    best_t = 0.50
    best_pnl = -float("inf")
    for t in thresholds:
        val = compute_pnl(y_true, (y_proba >= t).astype(int))
        if val > best_pnl:
            best_pnl = val
            best_t = float(t)
    return best_t, best_pnl


def identify_candidate_proxies(
    model: Any,
    X_train: pd.DataFrame,
    protected_attrs: dict[str, np.ndarray],
    features: list[str],
    delta: float = DELTA,
    alpha: float = ALPHA,
) -> tuple[list[str], dict[str, Any]]:
    """Identify candidate proxy features via Fairness Partial Dependence Plots (FPDP).

    A feature X_j is flagged as a candidate proxy if forcing X_j = c_k across all
    instances neutralizes the baseline unfairness (all pairwise TOST race tests
    pass, or any previously rejected pairwise test becomes fair).
    """
    X_tr = X_train[features].copy()
    base_preds = model.predict(X_tr)
    base_tost = evaluate_fairness(base_preds, protected_attrs, delta=delta, alpha=alpha)
    baseline_unfair_groups = [
        k for k, v in base_tost["details"]["race"].items() if not v["is_fair"]
    ]

    proxy_candidates: set[str] = set()
    fpdp_log: dict[str, Any] = {}

    for col in features:
        unique_vals = X_tr[col].dropna().unique()
        if len(unique_vals) <= 10:
            grid = np.sort(unique_vals)
        else:
            grid = np.quantile(X_tr[col].dropna(), np.linspace(0.05, 0.95, 19))
            grid = np.unique(grid)

        orig_col = X_tr[col].copy()
        col_flips = []
        all_fair_found = False

        for val in grid:
            X_tr[col] = val
            preds = model.predict(X_tr)
            tost = evaluate_fairness(preds, protected_attrs, delta=delta, alpha=alpha)
            race_details = tost["details"]["race"]

            # 1. Check if all racial tests pass
            if all(r["is_fair"] for r in race_details.values()):
                all_fair_found = True
                col_flips.append({
                    "type": "global_race_fair",
                    "val": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                    "gaps": {k: float(r["gap"]) for k, r in race_details.items()},
                })
                break

            # 2. Check if a previously unfair pairwise test becomes fair
            resolved = [g for g in baseline_unfair_groups if race_details[g]["is_fair"]]
            if resolved:
                col_flips.append({
                    "type": "partial_resolution",
                    "val": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                    "resolved_groups": resolved,
                })

        X_tr[col] = orig_col  # restore

        if all_fair_found or col_flips:
            proxy_candidates.add(col)
            fpdp_log[col] = {
                "all_fair_achieved": all_fair_found,
                "flips": col_flips[:3],
            }

    return sorted(list(proxy_candidates)), fpdp_log


def run_mitigation_pipeline() -> dict[str, Any]:
    """Execute the full Option A pre-processing mitigation pipeline."""
    # -----------------------------------------------------------------------
    # 1. Load Data Contract & Baseline Models
    # -----------------------------------------------------------------------
    data = pd.read_parquet(DATA_DIR / "clean.parquet")
    with open(DATA_DIR / "features.json") as f:
        contract = json.load(f)
    with open(DATA_DIR / "split.json") as f:
        split = json.load(f)

    FEATURES = contract["features"]
    train_mask = data["wave"].isin(split["train_waves"])
    test_mask = data["wave"].isin(split["test_waves"])

    train = data[train_mask].copy()
    test = data[test_mask].copy()

    X_train = train[FEATURES]
    y_train = train["dec"].values
    X_test = test[FEATURES]
    y_test = test["dec"].values

    pa_train = {
        "cand_female": train["cand_female"].values,
        "cand_race": train["cand_race"].values,
    }
    pa_test = {
        "cand_female": test["cand_female"].values,
        "cand_race": test["cand_race"].values,
    }

    xgb_baseline = joblib.load(MODELS_DIR / "xgb.joblib")
    logit_baseline = joblib.load(MODELS_DIR / "logit.joblib")

    # -----------------------------------------------------------------------
    # 2. FPDP Proxy Identification (Step 1)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Identifying Candidate Proxies via FPDP on Training Set...")
    print("=" * 60)
    candidate_proxies, fpdp_details = identify_candidate_proxies(
        xgb_baseline, X_train, pa_train, FEATURES, delta=DELTA, alpha=ALPHA
    )

    print(f"Identified {len(candidate_proxies)} candidate proxy features to drop:")
    for feat in candidate_proxies:
        print(f"  - {feat}: {fpdp_details[feat]}")

    PROXY_FEATURES_TO_DROP = candidate_proxies

    # -----------------------------------------------------------------------
    # 3. Dataset Modification (Step 2)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Creating Mitigated Feature Subsets...")
    print("=" * 60)
    MITIGATED_FEATURES = [f for f in FEATURES if f not in PROXY_FEATURES_TO_DROP]
    X_train_mitigated = X_train[MITIGATED_FEATURES].copy()
    X_test_mitigated = X_test[MITIGATED_FEATURES].copy()
    print(f"Features: {len(FEATURES)} baseline -> {len(MITIGATED_FEATURES)} mitigated (dropped {len(PROXY_FEATURES_TO_DROP)})")

    # -----------------------------------------------------------------------
    # 4. Retrain Mitigated Models (Step 3)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Retraining Mitigated Models...")
    print("=" * 60)

    # Retrain XGBoost
    xgb_mitigated = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
    )
    xgb_mitigated.fit(X_train_mitigated, y_train)

    # Retrain Logit
    logit_mitigated = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=SEED),
    )
    logit_mitigated.fit(X_train_mitigated, y_train)

    # Calibrate optimal P&L threshold on train set
    p_tr_xgb_base = xgb_baseline.predict_proba(X_train)[:, 1]
    p_tr_xgb_mit = xgb_mitigated.predict_proba(X_train_mitigated)[:, 1]
    t_opt_xgb_base, pnl_tr_xgb_base = optimize_threshold_pnl(y_train, p_tr_xgb_base)
    t_opt_xgb_mit, pnl_tr_xgb_mit = optimize_threshold_pnl(y_train, p_tr_xgb_mit)

    p_tr_logit_base = logit_baseline.predict_proba(X_train)[:, 1]
    p_tr_logit_mit = logit_mitigated.predict_proba(X_train_mitigated)[:, 1]
    t_opt_logit_base, _ = optimize_threshold_pnl(y_train, p_tr_logit_base)
    t_opt_logit_mit, _ = optimize_threshold_pnl(y_train, p_tr_logit_mit)

    print(f"XGBoost Optimal P&L threshold: baseline={t_opt_xgb_base:.2f}, mitigated={t_opt_xgb_mit:.2f}")

    # -----------------------------------------------------------------------
    # 5. Evaluate Fairness-Utility Trade-off on Test Set (Step 4)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Fairness-Utility Trade-off on Test Set...")
    print("=" * 60)

    # XGBoost predictions
    p_te_xgb_base = xgb_baseline.predict_proba(X_test)[:, 1]
    p_te_xgb_mit = xgb_mitigated.predict_proba(X_test_mitigated)[:, 1]

    auc_xgb_base = float(roc_auc_score(y_test, p_te_xgb_base))
    auc_xgb_mit = float(roc_auc_score(y_test, p_te_xgb_mit))

    pnl_xgb_base = compute_pnl(y_test, (p_te_xgb_base >= t_opt_xgb_base).astype(int))
    pnl_xgb_mit = compute_pnl(y_test, (p_te_xgb_mit >= t_opt_xgb_mit).astype(int))
    pnl_xgb_base_50 = compute_pnl(y_test, (p_te_xgb_base >= 0.50).astype(int))
    pnl_xgb_mit_50 = compute_pnl(y_test, (p_te_xgb_mit >= 0.50).astype(int))

    tost_xgb_base = evaluate_fairness((p_te_xgb_base >= 0.50).astype(int), pa_test, delta=DELTA, alpha=ALPHA)
    tost_xgb_mit = evaluate_fairness((p_te_xgb_mit >= 0.50).astype(int), pa_test, delta=DELTA, alpha=ALPHA)

    # Logit predictions
    p_te_logit_base = logit_baseline.predict_proba(X_test)[:, 1]
    p_te_logit_mit = logit_mitigated.predict_proba(X_test_mitigated)[:, 1]

    auc_logit_base = float(roc_auc_score(y_test, p_te_logit_base))
    auc_logit_mit = float(roc_auc_score(y_test, p_te_logit_mit))

    pnl_logit_base = compute_pnl(y_test, (p_te_logit_base >= t_opt_logit_base).astype(int))
    pnl_logit_mit = compute_pnl(y_test, (p_te_logit_mit >= t_opt_logit_mit).astype(int))

    tost_logit_base = evaluate_fairness((p_te_logit_base >= 0.50).astype(int), pa_test, delta=DELTA, alpha=ALPHA)
    tost_logit_mit = evaluate_fairness((p_te_logit_mit >= 0.50).astype(int), pa_test, delta=DELTA, alpha=ALPHA)

    metrics_comparison = {
        "candidate_proxies_dropped": PROXY_FEATURES_TO_DROP,
        "n_features_baseline": len(FEATURES),
        "n_features_mitigated": len(MITIGATED_FEATURES),
        "cost_matrix_params": {
            "gain_tp": GAIN_TP,
            "cost_fp": COST_FP,
            "cost_fn": COST_FN,
        },
        "xgb": {
            "baseline": {
                "auc": round(auc_xgb_base, 4),
                "pnl_optimal_threshold": pnl_xgb_base,
                "pnl_threshold_50": pnl_xgb_base_50,
                "optimal_threshold": round(t_opt_xgb_base, 2),
                "tost_race": tost_xgb_base["details"]["race"],
            },
            "mitigated": {
                "auc": round(auc_xgb_mit, 4),
                "pnl_optimal_threshold": pnl_xgb_mit,
                "pnl_threshold_50": pnl_xgb_mit_50,
                "optimal_threshold": round(t_opt_xgb_mit, 2),
                "tost_race": tost_xgb_mit["details"]["race"],
            },
            "delta": {
                "auc": round(auc_xgb_mit - auc_xgb_base, 4),
                "pnl": round(pnl_xgb_mit - pnl_xgb_base, 2),
            },
        },
        "logit": {
            "baseline": {
                "auc": round(auc_logit_base, 4),
                "pnl_optimal_threshold": pnl_logit_base,
                "optimal_threshold": round(t_opt_logit_base, 2),
                "tost_race": tost_logit_base["details"]["race"],
            },
            "mitigated": {
                "auc": round(auc_logit_mit, 4),
                "pnl_optimal_threshold": pnl_logit_mit,
                "optimal_threshold": round(t_opt_logit_mit, 2),
                "tost_race": tost_logit_mit["details"]["race"],
            },
            "delta": {
                "auc": round(auc_logit_mit - auc_logit_base, 4),
                "pnl": round(pnl_logit_mit - pnl_logit_base, 2),
            },
        },
    }

    print("\n--- Summary Comparison (XGBoost) ---")
    print(f"AUC: Baseline = {auc_xgb_base:.4f} -> Mitigated = {auc_xgb_mit:.4f} (Delta = {auc_xgb_mit - auc_xgb_base:+.4f})")
    print(f"P&L: Baseline = {pnl_xgb_base:.0f}€ -> Mitigated = {pnl_xgb_mit:.0f}€ (Delta = {pnl_xgb_mit - pnl_xgb_base:+.0f}€)")
    print("Fairness gaps (Caucasian vs Minority):")
    for group, base_res in tost_xgb_base["details"]["race"].items():
        mit_res = tost_xgb_mit["details"]["race"][group]
        print(f"  {group:22s}: Baseline={base_res['gap']*100:+.2f}% (fair={base_res['is_fair']}) -> Mitigated={mit_res['gap']*100:+.2f}% (fair={mit_res['is_fair']})")

    # -----------------------------------------------------------------------
    # 6. Artifact Export (Step 5)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Exporting Mitigated Models & Artifacts...")
    print("=" * 60)

    # Save models
    joblib.dump(xgb_mitigated, MODELS_DIR / "xgb_mitigated.joblib")
    joblib.dump(logit_mitigated, MODELS_DIR / "logit_mitigated.joblib")

    # Save features contract
    mitigated_features_export = {
        "dropped_proxies": PROXY_FEATURES_TO_DROP,
        "n_dropped": len(PROXY_FEATURES_TO_DROP),
        "features": MITIGATED_FEATURES,
        "n_features": len(MITIGATED_FEATURES),
        "fpdp_diagnostics": fpdp_details,
    }
    with open(DATA_DIR / "mitigated_features.json", "w") as f:
        json.dump(mitigated_features_export, f, indent=2)

    # Save metrics comparison
    with open(DATA_DIR / "mitigation_metrics.json", "w") as f:
        json.dump(metrics_comparison, f, indent=2)

    # -----------------------------------------------------------------------
    # 7. Visualization: Before vs After Mitigation Trade-off Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Plot A: Racial gaps comparison (XGBoost)
    ax_fair = axes[0]
    groups = ["caucasian_vs_asian", "caucasian_vs_latino", "caucasian_vs_black", "caucasian_vs_other"]
    labels = ["vs Asian", "vs Latino", "vs Black", "vs Other"]
    y_pos = np.arange(len(groups))

    gaps_base = [tost_xgb_base["details"]["race"][g]["gap"] * 100 for g in groups]
    gaps_mit = [tost_xgb_mit["details"]["race"][g]["gap"] * 100 for g in groups]

    ax_fair.axvspan(-DELTA * 100, DELTA * 100, color="#d4edda", alpha=0.5, label="Zone d'équivalence [−10 pp, +10 pp]")
    ax_fair.axvline(0, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
    ax_fair.axvline(-DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)
    ax_fair.axvline(DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)

    bar_width = 0.35
    ax_fair.barh(y_pos - bar_width / 2, gaps_base, bar_width, label="Baseline (XGBoost)", color="#ff7f0e", alpha=0.85)
    ax_fair.barh(y_pos + bar_width / 2, gaps_mit, bar_width, label="Mitigé (sans proxys)", color="#2ca02c", alpha=0.85)

    ax_fair.set_yticks(y_pos)
    ax_fair.set_yticklabels(labels, fontsize=11)
    ax_fair.set_xlabel("Écart d'exposition (p̂_cauc − p̂_groupe en pp)", fontsize=11)
    ax_fair.set_title("Évolution des écarts raciaux (TOST δ = 10 pp)\nAvant vs Après suppression des proxys", fontsize=12, fontweight="bold")
    ax_fair.legend(loc="lower left", framealpha=0.9)
    ax_fair.set_xlim(-25, 20)

    # Add text labels on bars
    for i, (gb, gm) in enumerate(zip(gaps_base, gaps_mit)):
        ax_fair.text(gb + (0.5 if gb >= 0 else -0.5), i - bar_width / 2, f"{gb:+.1f}", va="center", ha="left" if gb >= 0 else "right", fontsize=9, fontweight="bold", color="#d95f02")
        ax_fair.text(gm + (0.5 if gm >= 0 else -0.5), i + bar_width / 2, f"{gm:+.1f}", va="center", ha="left" if gm >= 0 else "right", fontsize=9, fontweight="bold", color="#1b9e77")

    # Plot B: Trade-off Utility vs Fairness
    ax_util = axes[1]
    models = ["XGBoost", "Logit"]
    auc_bases = [auc_xgb_base, auc_logit_base]
    auc_mits = [auc_xgb_mit, auc_logit_mit]

    x_indices = np.arange(len(models))
    ax_util.bar(x_indices - 0.18, auc_bases, 0.35, label="Baseline", color="#ff7f0e", alpha=0.85)
    ax_util.bar(x_indices + 0.18, auc_mits, 0.35, label="Mitigé", color="#2ca02c", alpha=0.85)
    ax_util.set_xticks(x_indices)
    ax_util.set_xticklabels(models, fontsize=11)
    ax_util.set_ylabel("AUC Test", fontsize=11)
    ax_util.set_ylim(0.50, 0.65)
    ax_util.set_title("Arbitrage d'utilité : Préservation de l'AUC\nAvant vs Après mitigation", fontsize=12, fontweight="bold")
    ax_util.legend(loc="lower right", framealpha=0.9)

    for i, (ab, am) in enumerate(zip(auc_bases, auc_mits)):
        ax_util.text(i - 0.18, ab + 0.005, f"{ab:.3f}", ha="center", fontsize=9.5, fontweight="bold")
        ax_util.text(i + 0.18, am + 0.005, f"{am:.3f}", ha="center", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "mitigation_tradeoff.png", dpi=300)
    plt.close(fig)

    print("Mitigation trade-off plot saved to:", REPORTS_DIR / "mitigation_tradeoff.png")
    return metrics_comparison


if __name__ == "__main__":
    run_mitigation_pipeline()
