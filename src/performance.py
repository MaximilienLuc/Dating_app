"""
HEC Match — performance prédictive (statistique + économique) des 3 modèles.

Portée volontairement limitée à ce qui est couvert par le cours ISAF : PR-AUC, calibration,
matrice de confusion, XPER (décomposition de l'AUC par feature), et optimisation du seuil de
décision sur un P&L économique. PAS de Brier score ni de log-loss -- non couverts en cours,
écartés volontairement (pas un oubli).

Sources des prédictions par modèle :
  - logit, xgb  : models/logit.joblib, models/xgb.joblib -- rechargés et appliqués directement
    (predict_proba), features via feature_dict["features_logit"] / feature_dict["features"].
  - tabicl      : data/tabicl_predictions.parquet (colonnes iid, pid, wave, tabicl_proba) --
    prédictions déjà calculées sur Colab, JAMAIS de rechargement/refit du modèle en local
    (TabICLClassifier.fit() segfault de façon reproductible sur cette machine, cf.
    docs/soutenance_notes.md "TabICL -- diagnostic du crash CPU et décision"). Conséquence :
    l'optimisation de seuil par GroupKFold pour TabICL n'est PAS un vrai ré-entraînement par
    fold (impossible en local) -- c'est une estimation par sous-échantillonnage des mêmes
    prédictions figées (un seul entraînement, cf. optimize_threshold_groupkfold ci-dessous),
    moins rigoureuse que pour logit/xgb. Documenté explicitement partout où c'est utilisé.

Hyperparamètres logit/xgb dupliqués ici depuis 01_data_models_v0.py (et depuis
src/stability.py, qui fait la même chose) -- dette technique déjà connue et actée pour être
traitée après le gel des modèles (cf. docs/soutenance_notes.md "Coordination équipe").
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.build_dataset import DATA_DIR, SEED

ROOT = DATA_DIR.parent
MODELS_DIR = ROOT / "models"
ECON_ASSUMPTIONS_PATH = DATA_DIR / "economic_assumptions.json"

MODEL_NAMES = ["logit", "xgb", "tabicl"]


# ===================================================================== #
#  Chargement                                                           #
# ===================================================================== #

def load_economic_assumptions(path=None):
    path = Path(path) if path else ECON_ASSUMPTIONS_PATH
    return json.loads(path.read_text())


def save_economic_assumptions(assumptions, path=None):
    path = Path(path) if path else ECON_ASSUMPTIONS_PATH
    path.write_text(json.dumps(assumptions, indent=2) + "\n")


def load_model_predictions():
    """Charge data/clean.parquet + les 3 modèles/prédictions, retourne un DataFrame unique
    avec une colonne proba_<modele> par modèle, pour toutes les lignes, ainsi que
    feature_dict (data/features.json)."""
    data = pd.read_parquet(DATA_DIR / "clean.parquet")
    feature_dict = json.loads((DATA_DIR / "features.json").read_text())
    features, features_logit = feature_dict["features"], feature_dict["features_logit"]

    logit = joblib.load(MODELS_DIR / "logit.joblib")
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")

    data = data.copy()
    data["proba_logit"] = logit.predict_proba(data[features_logit])[:, 1]
    data["proba_xgb"] = xgb.predict_proba(data[features])[:, 1]

    tabicl_preds = pd.read_parquet(DATA_DIR / "tabicl_predictions.parquet")
    data = data.merge(tabicl_preds[["iid", "pid", "wave", "tabicl_proba"]],
                       on=["iid", "pid", "wave"], how="left")
    data = data.rename(columns={"tabicl_proba": "proba_tabicl"})
    return data, feature_dict


def split_train_test(data):
    split = json.loads((DATA_DIR / "split.json").read_text())
    train_waves, test_waves = set(split["train_waves"]), set(split["test_waves"])
    train = data[data["wave"].isin(train_waves)].reset_index(drop=True)
    test = data[data["wave"].isin(test_waves)].reset_index(drop=True)
    return train, test, split


# ===================================================================== #
#  Partie 1 -- performance statistique au-delà de l'AUC                 #
# ===================================================================== #

def pr_auc(y_true, y_proba):
    """Precision-Recall AUC (average precision)."""
    return float(average_precision_score(y_true, y_proba))


def calibration_table(y_true, y_proba, n_bins=10):
    """Probabilité prédite moyenne vs taux réel observé, par bins de proba prédite."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "bin_range": f"[{bins[b]:.2f}, {bins[b + 1]:.2f})",
            "n": int(mask.sum()),
            "mean_predicted": float(y_proba[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def confusion_metrics(y_true, y_proba, threshold):
    """Accuracy/precision/recall/f1 + matrice de confusion, à un seuil donné."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ===================================================================== #
#  Partie 2 -- XPER (décomposition de l'AUC par feature)                #
# ===================================================================== #

def compute_xper(model, X_train, y_train, X_test, y_test, feature_names,
                  eval_metric="AUC", sample_size=200, n_coalition_sampled=500,
                  seed=SEED, top_n=10):
    """XPER sur l'AUC (Sinclair et al.) -- pip install XPER.

    Nécessite le VRAI objet modèle (appelé des milliers de fois sur des coalitions de
    features masquées) -- pas juste des prédictions figées. Fonctionne pour logit/xgb
    (modèle chargé en local). Pour TabICL : voir colab/xper_tabicl.py, pas exécutable ici
    (modèle non chargeable en local, et coût de calcul élevé même sur Colab -- à tenter,
    pas garanti).

    n_coalition_sampled réduit par rapport au défaut de la librairie (~2*p+2048, soit
    ~2360 pour nos 156-158 features) : un test chronométré sur le logit réel avec les
    valeurs par défaut a dépassé 10 minutes sans terminer (interrompu). Réduit à 500 pour
    rester dans un temps raisonnable -- approximation plus grossière, documentée comme
    telle (limite à assumer si citée en soutenance).
    """
    from XPER.compute.Performance import ModelPerformance

    Xtr = np.asarray(X_train)
    ytr = np.asarray(y_train)
    Xte = np.asarray(X_test)
    yte = np.asarray(y_test)

    perf = ModelPerformance(Xtr, ytr, Xte, yte, model, sample_size=sample_size, seed=seed)
    phi, phi_i_j = perf.calculate_XPER_values(
        [eval_metric], N_coalition_sampled=n_coalition_sampled, seed=seed)

    # phi[0] = valeur benchmark de la métrique ; phi[1:] = contribution par feature,
    # dans le même ordre que les colonnes de X_test.
    contrib = pd.DataFrame({"feature": feature_names, "xper_value": phi[1:]})
    contrib["abs_xper_value"] = contrib["xper_value"].abs()
    contrib = contrib.sort_values("abs_xper_value", ascending=False).reset_index(drop=True)
    return {
        "benchmark_auc": float(phi[0]),
        "contributions": contrib,
        "top_features": contrib.head(top_n),
    }


# ===================================================================== #
#  Partie 3 -- optimisation économique (P&L)                            #
# ===================================================================== #

def calculate_pnl(y_true, y_pred_proba, threshold, X, Y):
    """P&L calculé au niveau de la décision INDIVIDUELLE (dec), pas du vrai match mutuel
    (match = A ET B disent oui) -- simplification assumée et documentée (cf.
    docs/soutenance_notes.md) : croiser deux décisions d'une même paire demanderait de
    revenir aux deux lignes (A->B et B->A) et sort du cadre "un score, une recommandation".
    "Match manqué" devient ici "occasion de oui manquée" au niveau individuel : un profil
    montré qui aurait dit oui (dec=1) mais n'est pas recommandé (proba <= threshold) ne
    coûte ni ne rapporte rien dans cette version simplifiée (pas de coût d'opportunité
    modélisé pour l'instant).

    +X si le profil est recommandé (proba > threshold) ET dec == 1
    -Y si le profil est recommandé (proba > threshold) ET dec == 0
     0 si le profil n'est pas recommandé (proba <= threshold)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    recommended = y_pred_proba > threshold
    gains = X * float(np.sum(recommended & (y_true == 1)))
    costs = Y * float(np.sum(recommended & (y_true == 0)))
    return gains - costs


def _refit_predict(model_name, fold_train, fold_val, features, seed=SEED):
    """Ré-entraîne logit/xgb sur fold_train (mêmes hyperparamètres que
    01_data_models_v0.py) et retourne predict_proba sur fold_val. N'existe pas pour
    TabICL (voir optimize_threshold_groupkfold)."""
    if model_name == "logit":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=5000))
    elif model_name == "xgb":
        model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
    else:
        raise ValueError(f"Pas de ré-entraînement possible pour {model_name}")
    model.fit(fold_train[features], fold_train["dec"])
    return model.predict_proba(fold_val[features])[:, 1]


def optimize_threshold_groupkfold(train, model_name, X, Y, features=None,
                                   proba_col=None, thresholds=None, n_splits=5, seed=SEED):
    """Grille de seuils (0.05 à 0.95, pas 0.05) optimisée sur TRAIN via GroupKFold
    (5 folds sur wave, même découpage que l'AUC). Retourne (seuil_optimal, DataFrame des
    P&L moyens par seuil).

    logit/xgb : `features` requis, ré-entraîne le modèle à chaque fold (vrai GroupKFold).
    tabicl    : `proba_col` requis (colonne de proba déjà calculée dans `train`, un seul
    entraînement fixe) -- PAS un vrai ré-entraînement par fold, juste une évaluation du
    P&L restreinte aux indices de validation de chaque fold sur ces prédictions figées.
    Moins rigoureux que logit/xgb : à documenter comme limite si cité en soutenance.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    gkf = GroupKFold(n_splits=n_splits)
    groups = train["wave"]
    pnl_per_threshold = {t: [] for t in thresholds}

    for tr_idx, val_idx in gkf.split(train, train["dec"], groups=groups):
        fold_val = train.iloc[val_idx]
        if model_name == "tabicl":
            val_proba = fold_val[proba_col].to_numpy()
        else:
            fold_train = train.iloc[tr_idx]
            val_proba = _refit_predict(model_name, fold_train, fold_val, features, seed=seed)
        y_val = fold_val["dec"].to_numpy()
        for t in thresholds:
            pnl_per_threshold[t].append(calculate_pnl(y_val, val_proba, t, X, Y))

    grid = pd.DataFrame({
        "threshold": thresholds,
        "mean_pnl": [np.mean(pnl_per_threshold[t]) for t in thresholds],
        "std_pnl": [np.std(pnl_per_threshold[t]) for t in thresholds],
    })
    best_threshold = float(grid.loc[grid["mean_pnl"].idxmax(), "threshold"])
    return best_threshold, grid


def run_robustness_test(train, test, models_config, scenarios, n_splits=5, seed=SEED):
    """Refait l'optimisation de seuil pour plusieurs scénarios (X, Y), en réécrivant
    TEMPORAIREMENT data/economic_assumptions.json à chaque scénario (jamais de valeur X/Y
    en dur dans le code -- seul ce fichier est la source de vérité), puis restaure les
    valeurs d'origine à la fin (try/finally).

    models_config : dict {name: {"features": [...] } ou {"proba_col": "..."}}
    scenarios : liste de dicts {"X":.., "Y":.., "label":..}
    Retourne un DataFrame : scenario, modele, seuil_optimal, pnl_test_seuil_optimal.
    """
    original = load_economic_assumptions()
    rows = []
    try:
        for scenario in scenarios:
            save_economic_assumptions({"X": scenario["X"], "Y": scenario["Y"],
                                       "note": f"scénario robustesse temporaire : {scenario['label']}"})
            assumptions = load_economic_assumptions()
            X, Y = assumptions["X"], assumptions["Y"]
            for name, cfg in models_config.items():
                best_t, _ = optimize_threshold_groupkfold(
                    train, name, X, Y, features=cfg.get("features"),
                    proba_col=cfg.get("proba_col"), n_splits=n_splits, seed=seed)
                proba_col = cfg.get("proba_col") or f"proba_{name}"
                pnl_test = calculate_pnl(test["dec"], test[proba_col], best_t, X, Y)
                rows.append({"scenario": scenario["label"], "X": X, "Y": Y, "model": name,
                            "optimal_threshold": best_t, "pnl_test_at_optimal": pnl_test})
    finally:
        save_economic_assumptions(original)
    return pd.DataFrame(rows)


# ===================================================================== #
#  Orchestration -- calcule tout et sauvegarde reports/performance/     #
# ===================================================================== #

def run_full_analysis(output_dir=None, run_xper=True, xper_sample_size=500,
                       robustness_scenarios=None, seed=SEED):
    """Calcule les parties 1, 2, 3 pour les 3 modèles et sauvegarde tout dans
    reports/performance/. Retourne un dict avec tous les résultats en mémoire
    (utilisé par notebooks/05_performance.ipynb pour la narration)."""
    output_dir = Path(output_dir) if output_dir else (ROOT / "reports" / "performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    data, feature_dict = load_model_predictions()
    train, test, split = split_train_test(data)
    features, features_logit = feature_dict["features"], feature_dict["features_logit"]
    assumptions = load_economic_assumptions()
    X, Y = assumptions["X"], assumptions["Y"]

    proba_cols = {"logit": "proba_logit", "xgb": "proba_xgb", "tabicl": "proba_tabicl"}
    feats_by_model = {"logit": features_logit, "xgb": features, "tabicl": features}

    results = {"models": {}}

    # --- Partie 3 : seuil optimal par GroupKFold (train), P&L test au seuil optimal vs 0.5 ---
    thresholds_grid = {}
    for name in MODEL_NAMES:
        cfg_features = feats_by_model[name] if name != "tabicl" else None
        cfg_proba_col = proba_cols[name] if name == "tabicl" else None
        best_t, grid = optimize_threshold_groupkfold(
            train, name, X, Y, features=cfg_features, proba_col=cfg_proba_col, seed=seed)
        grid.to_csv(output_dir / f"threshold_grid_{name}.csv", index=False)
        thresholds_grid[name] = grid

        proba_test = test[proba_cols[name]]
        pnl_optimal = calculate_pnl(test["dec"], proba_test, best_t, X, Y)
        pnl_default = calculate_pnl(test["dec"], proba_test, 0.5, X, Y)
        results["models"].setdefault(name, {})
        results["models"][name].update({
            "optimal_threshold": best_t,
            "pnl_test_optimal": pnl_optimal,
            "pnl_test_default_05": pnl_default,
        })

    # --- Partie 1 : PR-AUC, calibration, matrice de confusion au seuil optimal ---
    calibration_tables = {}
    for name in MODEL_NAMES:
        y_test, proba_test = test["dec"], test[proba_cols[name]]
        results["models"][name]["pr_auc"] = pr_auc(y_test, proba_test)
        calib = calibration_table(y_test, proba_test)
        calib.to_csv(output_dir / f"calibration_{name}.csv", index=False)
        calibration_tables[name] = calib
        results["models"][name]["confusion_at_optimal"] = confusion_metrics(
            y_test, proba_test, results["models"][name]["optimal_threshold"])
    results["calibration_tables"] = calibration_tables

    # --- Partie 2 : XPER (logit, xgb -- tabicl non calculable en local, voir colab/xper_tabicl.py) ---
    xper_results = {}
    if run_xper:
        logit_model = joblib.load(MODELS_DIR / "logit.joblib")
        xgb_model = joblib.load(MODELS_DIR / "xgb.joblib")
        for name, model, feats in [("logit", logit_model, features_logit),
                                    ("xgb", xgb_model, features)]:
            xper_results[name] = compute_xper(
                model, train[feats], train["dec"], test[feats], test["dec"],
                feature_names=feats, sample_size=xper_sample_size, seed=seed)
            xper_results[name]["contributions"].to_csv(
                output_dir / f"xper_values_{name}.csv", index=False)
        xper_results["tabicl"] = {"status": "not_computed",
                                  "reason": "modèle non chargeable en local (segfault CPU) ; "
                                            "nécessite des milliers d'appels predict_proba, "
                                            "pas seulement des prédictions figées -- tenté sur "
                                            "Colab via colab/xper_tabicl.py, résultat non garanti"}
    results["xper"] = xper_results

    # --- Robustesse : scénarios alternatifs de X/Y ---
    if robustness_scenarios is None:
        robustness_scenarios = [
            {"X": 1, "Y": 1, "label": "X=1/Y=1"},
            {"X": 3, "Y": 0.3, "label": "X=3/Y=0.3"},
        ]
    models_config = {
        "logit": {"features": features_logit}, "xgb": {"features": features},
        "tabicl": {"proba_col": "proba_tabicl"},
    }
    robustness = run_robustness_test(train, test, models_config, robustness_scenarios, seed=seed)
    robustness.to_csv(output_dir / "robustness_test.csv", index=False)
    results["robustness"] = robustness

    # --- Tableau récapitulatif à 3 modèles ---
    summary_rows = []
    for name in MODEL_NAMES:
        m = results["models"][name]
        top_features = (xper_results[name]["top_features"]["feature"].tolist()
                        if run_xper and name in ("logit", "xgb") else None)
        summary_rows.append({
            "model": name,
            "pr_auc": round(m["pr_auc"], 4),
            "optimal_threshold": m["optimal_threshold"],
            "pnl_test_optimal": round(m["pnl_test_optimal"], 2),
            "pnl_test_default_05": round(m["pnl_test_default_05"], 2),
            "accuracy_at_optimal": round(m["confusion_at_optimal"]["accuracy"], 3),
            "precision_at_optimal": round(m["confusion_at_optimal"]["precision"], 3),
            "recall_at_optimal": round(m["confusion_at_optimal"]["recall"], 3),
            "f1_at_optimal": round(m["confusion_at_optimal"]["f1"], 3),
            "top_xper_features": ", ".join(top_features[:5]) if top_features else "n/a",
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_3_models.csv", index=False)
    results["summary"] = summary

    return results
