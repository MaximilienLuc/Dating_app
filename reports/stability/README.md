# Stability

Three models are evaluated on the same held-out sessions. Only logit and XGBoost are refitted.

| Analysis | Logit | XGBoost | TabICL |
| :--- | :---: | :---: | :---: |
| Fixed-model test AUC and session bootstrap | Yes | Yes | Yes |
| AUC per held-out session | Yes | Yes | Yes |
| Training-session resampling | Yes | Yes | No |
| Prediction changes and decision flips | Yes | Yes | No |
| Coefficient/importance distances | Yes | Yes | No |

## Reproduce

```bash
pip install -r requirements-stability.txt
python -m pytest tests/test_stability.py -q
python -m src.stability --refits 200 --eval-bootstrap 2000
jupyter nbconvert --to notebook --execute 04_stability.ipynb --output 04_stability.executed.ipynb
```

Run from the repository root. The source notebook checks input hashes before displaying results. The script never loads the group's serialized models or trains TabICL. It refits logit/XGBoost with the v0 hyperparameters and current feature contract. TabICL probabilities come from `data/tabicl_predictions.parquet`, joined by `(iid, pid, wave)` with uniqueness and coverage checks. The supplied export has no embedded training/data provenance manifest: its compatibility beyond keys relies on the group's export process; regenerate it when its training inputs change.

## Method

The logit uses `features_logit`; XGBoost uses `features`. Legacy contracts fall back to the shared list, but cloud publication requires the reduced logit list. The split and seed stay fixed.

- **Training sensitivity:** 200 paired draws of entire training sessions with replacement. Refit preprocessing and each estimator, then evaluate on the same fixed test encounters. Hold the estimator seed fixed. This measures training-data sensitivity, not algorithmic seed sensitivity.
- **Test uncertainty:** 2,000 paired draws of entire test sessions with replacement. Predictions are fixed for all three models. Report pooled, encounter-weighted AUC and pairwise AUC differences. This is a session bootstrap, not GroupKFold cross-validation.
- **Explanation sensitivity:** Euclidean and cosine distances from each model's reference vector. Logit coefficients use a common scale (original training-set standard deviations); XGBoost uses normalized gain importance. Compare distances within a model, not between these different vector types.

Decision flips use a provisional threshold of 0.5. No feature, parameter or threshold is selected on the test set. Training-resample ranges describe sensitivity, not classical confidence intervals. There are only six held-out sessions, so test intervals are exploratory. TabICL training stability is unknown, not zero.

The existing logit is already L2-regularized with C=1. Elastic Net or alternative regularization strengths would require a separate, internally validated experiment. An XGBoost performance/stability trade-off likewise requires comparing configurations, not just observing importance variance. Optimal P&L threshold stability is deferred until the group defines its cost matrix and validation protocol.

## Files for the app and slides

- `summary.json`: model coverage, exact feature lists, aggregate results, versions and hashes.
- `refits.csv`: logit/XGBoost training resamples only.
- `test_bootstrap.csv`: all three fixed models and paired differences.
- `by_session.csv`: sample sizes and per-session AUC.
- `logit_coefficients.csv`: coefficient ranges and sign agreement.
- `stability.png`, `coefficients.png`: charts from the executed notebook.
- `provenance.json`: cloud run and source revision.

[Executed notebook](../../04_stability.executed.ipynb)

## Verified cloud results

| Model | AUC | Mean decision flips at 0.5 |
| :--- | ---: | ---: |
| Logit | 0.589 | 16.7% |
| XGBoost | 0.614 | 21.4% |
| TabICL | 0.632 | Not measured |

All three paired 95% percentile intervals for AUC differences include zero. Six test sessions do not establish a clear winner. Decision flips measure sensitivity to training samples, not mistakes against ground truth.

[Cloud run](https://github.com/MaximilienLuc/Dating_app/actions/runs/36034895577): 200 refits per trainable model, 2,000 paired test resamples, 18 passing tests and an executed notebook. No participant-level predictions are exported by this analysis.
