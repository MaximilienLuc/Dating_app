"""HEC Match — fairness metrics (TOST equivalence test).

Implements the Two-One-Sided T-test (Schuirmann, 1987) to *prove*
that exposure rates across protected groups are equivalent within a
tolerance δ.  Unlike a standard t-test ("is there a difference?"),
TOST reverses the burden of proof: the model is **assumed unfair**
until we can demonstrate the gap is negligibly small.

Public API
----------
evaluate_fairness(y_pred, protected_attrs, *, delta=0.05, alpha=0.05)
    → dict   (see "Output Contract" in the docstring)

Private
-------
_compute_tost(preds_ref, preds_comp, delta, alpha)
    → dict   (core math, one pairwise comparison)
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# Race encoding (Speed Dating Data Key.doc)
# Mirrors build_dataset.py line 89:
#   1 Black/African American  |  2 European/Caucasian-American
#   3 Latino/Hispanic American | 4 Asian/Pacific Islander/Asian-American
#   5 Native American  |  6 Other
# ---------------------------------------------------------------------------
_RACE_LABELS: dict[int, str] = {
    1: "black",
    2: "caucasian",
    3: "latino",
    4: "asian",
    5: "native_american",
    6: "other",
}
_CAUCASIAN_CODE = 2


# ===================================================================== #
#  Core TOST (Schuirmann, 1987)                                         #
# ===================================================================== #

def _compute_tost(
    preds_ref: np.ndarray,
    preds_comp: np.ndarray,
    delta: float = 0.05,
    alpha: float = 0.05,
) -> dict:
    """Run a single TOST equivalence test on two arrays of binary predictions.

    Parameters
    ----------
    preds_ref : 1-D array of {0, 1}
        Binary predictions for the **reference** group.
    preds_comp : 1-D array of {0, 1}
        Binary predictions for the **comparison** group.
    delta : float, default 0.05
        Equivalence tolerance (maximum acceptable gap in positive rates).
    alpha : float, default 0.05
        Significance level for each one-sided test.

    Returns
    -------
    dict with keys ``is_fair`` (bool), ``gap`` (float), ``n_ref`` (int),
    ``n_comp`` (int).
    """
    preds_ref = np.asarray(preds_ref, dtype=float)
    preds_comp = np.asarray(preds_comp, dtype=float)

    # ---- 1. Rates & observed gap ----------------------------------------
    p_ref = preds_ref.mean()
    p_comp = preds_comp.mean()
    theta_hat = p_ref - p_comp

    n_ref = len(preds_ref)
    n_comp = len(preds_comp)

    # ---- 2. Small-sample warning (non-blocking) -------------------------
    if n_ref < 30 or n_comp < 30:
        warnings.warn(
            "Low sample size detected (n < 30). "
            "Variance will be high, reducing test reliability."
        )

    # ---- 3. Unpooled variance estimator ---------------------------------
    var = (p_ref * (1 - p_ref) / n_ref) + (p_comp * (1 - p_comp) / n_comp)
    sigma_hat = np.sqrt(var)

    # Edge case: variance is zero (both groups have constant predictions).
    # If the absolute gap is within the tolerance → trivially equivalent.
    # If the gap exceeds delta → clearly non-equivalent (no noise to mask it).
    if sigma_hat == 0.0:
        return {
            "is_fair": bool(abs(theta_hat) < delta),
            "gap": float(theta_hat),
            "n_ref": n_ref,
            "n_comp": n_comp,
        }

    # ---- 4. Z-statistics (lower & upper one-sided tests) ----------------
    z_lower = (theta_hat + delta) / sigma_hat
    z_upper = (delta - theta_hat) / sigma_hat

    # ---- 5. Decision rule -----------------------------------------------
    df = n_ref + n_comp - 2
    t_crit = t_dist.ppf(1 - alpha, df)

    is_fair = bool(z_lower > t_crit and z_upper > t_crit)

    return {
        "is_fair": is_fair,
        "gap": float(theta_hat),
        "n_ref": n_ref,
        "n_comp": n_comp,
    }


# ===================================================================== #
#  Public wrapper — gender + race (pairwise)                            #
# ===================================================================== #

def evaluate_fairness(
    y_pred: np.ndarray,
    protected_attrs: dict[str, np.ndarray],
    *,
    delta: float = 0.05,
    alpha: float = 0.05,
) -> dict:
    """Run the full fairness audit (gender + race) and return a standardized
    result dictionary compatible with the Streamlit app.

    Parameters
    ----------
    y_pred : 1-D array of {0, 1}
        Binary predictions for the test set (thresholded probabilities).
    protected_attrs : dict
        Must contain at least:

        * ``"cand_female"`` : 1-D array (0 = male candidate, 1 = female).
        * ``"cand_race"``   : 1-D array of integer race codes (see
          ``_RACE_LABELS``).

    delta : float, default 0.05
        Equivalence tolerance passed to ``_compute_tost``.
    alpha : float, default 0.05
        Significance level passed to ``_compute_tost``.

    Returns
    -------
    dict
        Structured as::

            {
                "global_fairness": bool,
                "details": {
                    "gender": {"is_fair": …, "gap": …, "n_ref": …, "n_comp": …},
                    "race": {
                        "caucasian_vs_<label>": {…},
                        …
                    }
                }
            }

        ``global_fairness`` is True **only if** both gender and *all*
        pairwise race tests return True.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    cand_female = np.asarray(protected_attrs["cand_female"])
    cand_race = np.asarray(protected_attrs["cand_race"])

    # ---- A. Gender test -------------------------------------------------
    # Reference = men (cand_female == 0), Comparison = women (cand_female == 1)
    mask_male = cand_female == 0
    mask_female = cand_female == 1
    gender_result = _compute_tost(
        y_pred[mask_male], y_pred[mask_female], delta=delta, alpha=alpha
    )

    # ---- B. Race tests (Caucasian vs. each minority, pairwise) ----------
    mask_caucasian = cand_race == _CAUCASIAN_CODE
    preds_caucasian = y_pred[mask_caucasian]

    race_results: dict[str, dict] = {}
    all_race_fair = True

    unique_races = np.unique(cand_race[~np.isnan(cand_race)])
    for code in sorted(unique_races):
        code_int = int(code)
        if code_int == _CAUCASIAN_CODE:
            continue
        label = _RACE_LABELS.get(code_int, f"race_{code_int}")
        key = f"caucasian_vs_{label}"

        mask_minority = cand_race == code_int
        result = _compute_tost(
            preds_caucasian, y_pred[mask_minority], delta=delta, alpha=alpha
        )
        race_results[key] = result

        if not result["is_fair"]:
            all_race_fair = False

    # ---- Aggregate ------------------------------------------------------
    global_fairness = gender_result["is_fair"] and all_race_fair

    return {
        "global_fairness": global_fairness,
        "details": {
            "gender": gender_result,
            "race": race_results,
        },
    }
