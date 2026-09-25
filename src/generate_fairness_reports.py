"""Generate fairness figures and summary tables for the final report.

Outputs:
- reports/fairness/tost_intervals.png
- reports/fairness/racial_exposure_rates.png
- reports/fairness/tost_summary.csv
- reports/fairness/tost_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "fairness"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data and models
# ---------------------------------------------------------------------------
data = pd.read_parquet(DATA_DIR / "clean.parquet")
with open(DATA_DIR / "features.json") as f:
    contract = json.load(f)
with open(DATA_DIR / "split.json") as f:
    split = json.load(f)

FEATURES = contract["features"]
test_mask = data["wave"].isin(split["test_waves"])
test = data[test_mask].copy()
X_te = test[FEATURES]

logit = joblib.load(MODELS_DIR / "logit.joblib")
xgb = joblib.load(MODELS_DIR / "xgb.joblib")

y_logit = (logit.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
y_xgb = (xgb.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
test["pred_logit"] = y_logit
test["pred_xgb"] = y_xgb

RACE_LABELS = {1: "Black", 2: "Caucasian", 3: "Latino", 4: "Asian", 6: "Other"}
DELTA = 0.10
ALPHA = 0.05

# ---------------------------------------------------------------------------
# 2. Compute TOST statistics
# ---------------------------------------------------------------------------
records = []
for model_name, preds in [("Logit", y_logit), ("XGBoost", y_xgb)]:
    mask_cauc = test["cand_race"] == 2
    p_cauc = float(preds[mask_cauc].mean())
    n_cauc = int(mask_cauc.sum())

    for code, label in RACE_LABELS.items():
        if code == 2:
            continue
        mask_m = test["cand_race"] == code
        p_m = float(preds[mask_m].mean())
        n_m = int(mask_m.sum())
        gap = p_cauc - p_m
        var = (p_cauc * (1 - p_cauc) / n_cauc) + (p_m * (1 - p_m) / n_m)
        sigma = float(np.sqrt(var))
        df = n_cauc + n_m - 2
        tcrit = float(t_dist.ppf(1 - ALPHA, df))
        ci_lower = gap - tcrit * sigma
        ci_upper = gap + tcrit * sigma
        zl = (gap + DELTA) / sigma
        zu = (DELTA - gap) / sigma
        is_fair = bool(zl > tcrit and zu > tcrit)

        records.append({
            "model": model_name,
            "comparison": f"Caucasian vs {label}",
            "group": label,
            "n_cauc": n_cauc,
            "n_min": n_m,
            "p_cauc_pct": round(p_cauc * 100, 2),
            "p_min_pct": round(p_m * 100, 2),
            "gap_pp": round(gap * 100, 2),
            "sigma_pp": round(sigma * 100, 2),
            "ci_lower_pp": round(ci_lower * 100, 2),
            "ci_upper_pp": round(ci_upper * 100, 2),
            "is_fair": is_fair,
        })

df_tost = pd.DataFrame(records)
df_tost.to_csv(REPORTS_DIR / "tost_summary.csv", index=False)
with open(REPORTS_DIR / "tost_summary.json", "w") as f:
    json.dump(records, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Figure 1: TOST Equivalence Confidence Intervals (delta = 10 pp)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

groups = ["Asian", "Latino", "Black", "Other"]
y_positions = np.arange(len(groups))
offset = 0.18

# Green shaded equivalence zone
ax.axvspan(-DELTA * 100, DELTA * 100, color="#d4edda", alpha=0.5, label="Zone d'équivalence [−10 pp, +10 pp]")
ax.axvline(0, color="#6c757d", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(-DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)
ax.axvline(DELTA * 100, color="#28a745", linestyle=":", linewidth=1.5)

models = [("Logit", -offset, "#1f77b4"), ("XGBoost", offset, "#ff7f0e")]

for model_name, pos_offset, color in models:
    sub = df_tost[df_tost["model"] == model_name].set_index("group")
    for i, g in enumerate(groups):
        row = sub.loc[g]
        y = i + pos_offset
        gap = row["gap_pp"]
        ci_l = row["ci_lower_pp"]
        ci_u = row["ci_upper_pp"]
        is_fair = row["is_fair"]

        err_left = gap - ci_l
        err_right = ci_u - gap

        marker_edge = "#28a745" if is_fair else "#dc3545"
        marker_symbol = "o" if is_fair else "X"

        ax.errorbar(
            gap, y,
            xerr=[[err_left], [err_right]],
            fmt=marker_symbol,
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=5,
            capthick=2,
            markersize=9,
            markeredgecolor=marker_edge,
            markeredgewidth=2,
            label=f"{model_name}" if i == 0 else ""
        )

        # Label gap value next to marker
        offset_text = 0.8 if gap >= 0 else -0.8
        ha = "left" if gap >= 0 else "right"
        ax.text(
            gap + offset_text, y + 0.05,
            f"{gap:+.1f} pp",
            va="center", ha=ha,
            fontsize=9.5, fontweight="bold",
            color=color
        )

ax.set_yticks(y_positions)
ax.set_yticklabels([f"vs {g}\n(n={int(test[test['cand_race'] == [k for k,v in RACE_LABELS.items() if v==g][0]].shape[0])})" for g in groups], fontsize=11)
ax.set_xlabel("Écart d'exposition (p̂_cauc − p̂_groupe en points de pourcentage)", fontsize=12, labelpad=8)
ax.set_title("Test d'équivalence TOST (Schuirmann, 1987) — Seuil de tolérance δ = 10 pp\nIntervalles de confiance unilatéraux conjoints à 90% (α = 0.05)", fontsize=13, fontweight="bold", pad=12)

# Custom legend entries
from matplotlib.lines import Line2D
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc="#d4edda", edgecolor="#28a745", label="Zone tolérée [−10 pp, +10 pp]"),
    Line2D([0], [0], color="#1f77b4", lw=2, marker="o", markersize=8, label="Logit (IC 90%)"),
    Line2D([0], [0], color="#ff7f0e", lw=2, marker="o", markersize=8, label="XGBoost (IC 90%)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="#28a745", markeredgewidth=2, markersize=8, label="Équivalence validée (Fair)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markeredgecolor="#dc3545", markeredgewidth=2, markersize=8, label="Non-équivalence (Unfair)"),
]
ax.legend(handles=legend_elements, loc="lower right", framealpha=0.92, fontsize=9.5)
ax.set_xlim(-32, 26)
plt.tight_layout()
fig.savefig(REPORTS_DIR / "tost_intervals.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Figure 2: Group Exposure Rates by Model & Ground Truth
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

plot_groups = ["Caucasian", "Asian", "Latino", "Black", "Other"]
code_map = {v: k for k, v in RACE_LABELS.items()}

rates_truth = [test[test["cand_race"] == code_map[g]]["dec"].mean() * 100 for g in plot_groups]
rates_logit = [test[test["cand_race"] == code_map[g]]["pred_logit"].mean() * 100 for g in plot_groups]
rates_xgb   = [test[test["cand_race"] == code_map[g]]["pred_xgb"].mean() * 100 for g in plot_groups]
counts      = [int((test["cand_race"] == code_map[g]).sum()) for g in plot_groups]

x = np.arange(len(plot_groups))
width = 0.26

rects1 = ax.bar(x - width, rates_truth, width, label="Données réelles (dec)", color="#6c757d", alpha=0.85)
rects2 = ax.bar(x, rates_logit, width, label="Prédictions Logit", color="#1f77b4", alpha=0.9)
rects3 = ax.bar(x + width, rates_xgb, width, label="Prédictions XGBoost", color="#ff7f0e", alpha=0.9)

ax.set_ylabel("Taux d'exposition / recommandation (%)", fontsize=12)
ax.set_title("Taux d'exposition pré-rencontre par groupe racial sur le set de test (n=1 826)\nComparaison : Cible réelle vs. Logit vs. XGBoost", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(plot_groups, counts)], fontsize=11)
ax.set_ylim(0, 65)
ax.legend(framealpha=0.92, fontsize=10.5)

# Value annotations on bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
fig.savefig(REPORTS_DIR / "racial_exposure_rates.png", dpi=300)
plt.close(fig)

print("Figures successfully generated in:", REPORTS_DIR)
