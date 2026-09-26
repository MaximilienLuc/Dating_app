"""
HEC Match — performance prédictive (statistique + économique), modèles de base ET mitigés.

Portée volontairement limitée à ce qui est couvert par le cours ISAF : PR-AUC, calibration,
matrice de confusion, XPER (décomposition de l'AUC ou du coût économique par feature), et
optimisation du seuil de décision sur un P&L économique. PAS de Brier score ni de log-loss --
non couverts en cours, écartés volontairement (pas un oubli).

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
  - logit_mitigated, xgb_mitigated : models/logit_mitigated.joblib, models/xgb_mitigated.joblib
    (Max, src/mitigation.py -- 4 proxies raciaux retirés via FPDP). PAS de version TabICL
    mitigée (n'existe pas). ATTENTION : ces modèles ont été entraînés sur l'ANCIEN encodage à
    164 features (avant notre correction drop_first=True), qui contient 6 colonnes de référence
    (field_cd/career_c/goal _1, A et B) absentes de data/clean.parquet actuel.
    add_legacy_reference_dummies() les reconstruit EXACTEMENT (encodage one-hot exhaustif :
    référence = 1 - somme des autres catégories de la même variable/côté -- pas une
    approximation) pour pouvoir appliquer ces modèles sans attendre que Max ré-entraîne sur le
    nouveau schéma.

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

BASELINE_MODEL_NAMES = ["logit", "xgb", "tabicl"]
MITIGATED_MODEL_NAMES = ["logit_mitigated", "xgb_mitigated", "tabicl_mitigated"]
MODEL_NAMES = BASELINE_MODEL_NAMES  # rétro-compatibilité (notebook, appels existants)

# Reconstruction de l'ancien encodage (164 features, avant drop_first=True) pour les modèles
# mitigés de Max. Catégories réellement observées (2..N + "NA"), la catégorie "1" (référence,
# supprimée par notre drop_first) se déduit par complément à 1.
_LEGACY_CAT_OTHER_CODES = {
    "field_cd": [str(i) for i in range(2, 19)] + ["NA"],
    "career_c": [str(i) for i in range(2, 18)] + ["NA"],
    "goal": [str(i) for i in range(2, 7)] + ["NA"],
}


# ===================================================================== #
#  Chargement                                                           #
# ===================================================================== #

def load_economic_assumptions(path=None):
    path = Path(path) if path else ECON_ASSUMPTIONS_PATH
    return json.loads(path.read_text())


def save_economic_assumptions(assumptions, path=None):
    path = Path(path) if path else ECON_ASSUMPTIONS_PATH
    path.write_text(json.dumps(assumptions, indent=2) + "\n")


def add_legacy_reference_dummies(data):
    """Reconstruit les 6 colonnes de référence (_1) de field_cd/career_c/goal (A et B),
    retirées par notre correction drop_first=True mais nécessaires pour charger les modèles
    mitigés de Max (entraînés sur l'ancien schéma à 164 features). Reconstruction EXACTE, pas
    une approximation : encodage one-hot exhaustif, donc référence = 1 - somme des autres
    catégories de la même variable/côté (chaque ligne appartient à exactement une catégorie)."""
    data = data.copy()
    for cat, other_codes in _LEGACY_CAT_OTHER_CODES.items():
        for side in "AB":
            ref_col = f"{cat}_{side}_1"
            other_cols = [f"{cat}_{side}_{code}" for code in other_codes]
            data[ref_col] = 1 - data[other_cols].sum(axis=1)
    return data


def load_model_predictions():
    """Charge data/clean.parquet + les 3 modèles/prédictions de base, retourne un DataFrame
    unique avec une colonne proba_<modele> par modèle, pour toutes les lignes, ainsi que
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


def load_mitigated_predictions(data):
    """Ajoute proba_logit_mitigated/proba_xgb_mitigated/proba_tabicl_mitigated à `data` (déjà
    chargé par load_model_predictions). Retourne (data, mitigated_features).

    logit_mitigated/xgb_mitigated : rechargés et appliqués directement (predict_proba), comme
    les modèles de base -- voir add_legacy_reference_dummies pour la reconstruction du schéma.
    tabicl_mitigated : prédictions déjà calculées sur Colab
    (data/tabicl_mitigated_predictions.parquet, colonne tabicl_mit_proba), entraîné par
    colab/train_tabicl_mitigated.py -- JAMAIS de rechargement/refit du modèle en local, même
    contrainte CUDA que TabICL de base (cf. load_model_predictions).
    """
    mitigated_features = json.loads((DATA_DIR / "mitigated_features.json").read_text())["features"]
    data_legacy = add_legacy_reference_dummies(data)

    logit_mit = joblib.load(MODELS_DIR / "logit_mitigated.joblib")
    xgb_mit = joblib.load(MODELS_DIR / "xgb_mitigated.joblib")

    data = data_legacy  # garde les 6 colonnes de référence reconstruites (nécessaires pour
                        # ré-entraîner logit_mitigated/xgb_mitigated par fold, cf. GroupKFold)
    data["proba_logit_mitigated"] = logit_mit.predict_proba(data[mitigated_features])[:, 1]
    data["proba_xgb_mitigated"] = xgb_mit.predict_proba(data[mitigated_features])[:, 1]

    tabicl_mit_preds = pd.read_parquet(DATA_DIR / "tabicl_mitigated_predictions.parquet")
    data = data.merge(tabicl_mit_preds[["iid", "pid", "wave", "tabicl_mit_proba"]],
                       on=["iid", "pid", "wave"], how="left")
    data = data.rename(columns={"tabicl_mit_proba": "proba_tabicl_mitigated"})
    return data, mitigated_features


def split_train_test(data):
    split = json.loads((DATA_DIR / "split.json").read_text())
    train_waves, test_waves = set(split["train_waves"]), set(split["test_waves"])
    train = data[data["wave"].isin(train_waves)].reset_index(drop=True)
    test = data[data["wave"].isin(test_waves)].reset_index(drop=True)
    return train, test, split


def build_model_specs(feature_dict, mitigated_features=None):
    """Décrit chaque modèle : colonne de proba déjà calculée dans `data`, liste de features
    (None + non refittable pour TabICL, qui n'a que des prédictions figées), et si un vrai
    ré-entraînement par fold GroupKFold est possible."""
    specs = {
        "logit": {"proba_col": "proba_logit", "features": feature_dict["features_logit"],
                  "refittable": True, "refit_kind": "logit"},
        "xgb": {"proba_col": "proba_xgb", "features": feature_dict["features"],
                "refittable": True, "refit_kind": "xgb"},
        "tabicl": {"proba_col": "proba_tabicl", "features": None, "refittable": False,
                   "refit_kind": None},
    }
    if mitigated_features is not None:
        specs["logit_mitigated"] = {"proba_col": "proba_logit_mitigated",
                                     "features": mitigated_features, "refittable": True,
                                     "refit_kind": "logit"}
        specs["xgb_mitigated"] = {"proba_col": "proba_xgb_mitigated",
                                   "features": mitigated_features, "refittable": True,
                                   "refit_kind": "xgb"}
        specs["tabicl_mitigated"] = {"proba_col": "proba_tabicl_mitigated", "features": None,
                                      "refittable": False, "refit_kind": None}
    return specs


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
#  Partie 2 -- XPER (décomposition de l'AUC ou du coût par feature)     #
# ===================================================================== #

def compute_xper(model, X_train, y_train, X_test, y_test, feature_names,
                  eval_metric="AUC", cost_fp=None, cost_fn=None,
                  sample_size=100, n_coalition_sampled=150, seed=SEED, top_n=10):
    """XPER (Sinclair et al.) -- pip install XPER.

    eval_metric="AUC" : décomposition de la performance statistique (discrimination).
    eval_metric="MC"  : décomposition du COÛT DE MAUVAISE CLASSIFICATION -- nécessite cost_fp
      et cost_fn (coûts positifs, magnitude ; alignés sur src/mitigation.py de Max :
      COST_FP=1.0, COST_FN=1.0). Ne modélise PAS de gain pour un vrai positif (contrairement à
      notre calculate_pnl) -- ce n'est donc pas un XPER exact de notre P&L, mais la décomposition
      la plus proche que XPER propose nativement (seules 5 métriques supportées :
      AUC/BS/Balanced_accuracy/Accuracy/MC -- vérifié dans le code source de la librairie).

    Nécessite le VRAI objet modèle (appelé des centaines de fois sur des coalitions de
    features masquées) -- pas juste des prédictions figées. Fonctionne pour logit/xgb
    (modèle chargé en local, base ou mitigé). Pour TabICL : voir colab/xper_tabicl.py, pas
    exécutable ici (modèle non chargeable en local, et coût de calcul élevé même sur Colab).

    n_coalition_sampled réduit par rapport au défaut de la librairie (~2*p+2048, soit ~2360
    pour nos 156-160 features) : un premier test chronométré avec les valeurs par défaut a
    dépassé 10 min sans terminer ; un deuxième essai à 500 coalitions/sample_size=200 a mis
    9h17 (essentiellement de la veille système, ~2h14 de calcul réel pour XGBoost -- cf.
    docs/soutenance_notes.md). Réduit encore à 150/100 : 18.6s pour XGBoost -- approximation
    plus grossière, documentée comme telle (limite à assumer si citée en soutenance).

    ATTENTION -- bug vérifié dans XPER==<version installée> : les paramètres CFP/CFN de
    `calculate_XPER_values`/`evaluate` sont INVERSÉS par rapport à leur docstring ("CFP: Cost
    of false positive", "CFN: Cost of false negative"). Vérifié empiriquement (script ad hoc,
    logistic regression synthétique, comparaison à un coût attendu -(CFP*FP+CFN*FN)/N) sur
    `Optimisation.py` (branche `Eval_Metric == ["MC"]`, les deux occurrences) ET sur
    `Performance.evaluate()` : le paramètre nommé `CFP` multiplie en réalité l'indicatrice de
    FAUX NÉGATIF, et `CFN` multiplie l'indicatrice de FAUX POSITIF. Sans incidence tant que
    cost_fp == cost_fn (notre cas jusqu'ici, Y=Z=1 partout) ; devient significatif dès que les
    deux diffèrent (cf. compute_xper_pnl ci-dessous, qui en tient compte explicitement).
    """
    from XPER.compute.Performance import ModelPerformance

    Xtr = np.asarray(X_train)
    ytr = np.asarray(y_train)
    Xte = np.asarray(X_test)
    yte = np.asarray(y_test)

    perf = ModelPerformance(Xtr, ytr, Xte, yte, model, sample_size=sample_size, seed=seed)
    kwargs = {}
    if eval_metric == "MC":
        if cost_fp is None or cost_fn is None:
            raise ValueError('eval_metric="MC" nécessite cost_fp et cost_fn (obligatoires '
                             "dans l'API XPER pour cette métrique)")
        kwargs.update(CFP=cost_fp, CFN=cost_fn)
    phi, phi_i_j = perf.calculate_XPER_values(
        [eval_metric], N_coalition_sampled=n_coalition_sampled, seed=seed, **kwargs)

    # phi[0] = valeur benchmark de la métrique ; phi[1:] = contribution par feature,
    # dans le même ordre que les colonnes de X_test.
    contrib = pd.DataFrame({"feature": feature_names, "xper_value": phi[1:]})
    contrib["abs_xper_value"] = contrib["xper_value"].abs()
    contrib = contrib.sort_values("abs_xper_value", ascending=False).reset_index(drop=True)
    return {
        "benchmark_value": float(phi[0]),
        "eval_metric": eval_metric,
        "contributions": contrib,
        "top_features": contrib.head(top_n),
    }


def compute_xper_pnl(model, X_train, y_train, X_test, y_test, feature_names, X, Y, Z,
                      sample_size=100, n_coalition_sampled=150, seed=SEED, top_n=10):
    """XPER décomposant exactement notre P&L (calculate_pnl), pas juste le coût de
    mauvaise classification brut de Max (compute_xper(eval_metric="MC")).

    Dérivation : calculate_pnl = X*TP - Y*FP - Z*FN, et TP = P - FN (P = nombre de positifs
    réels, constant, ne dépend ni du modèle ni des features). Donc :
        P&L = X*P - (X+Z)*FN - Y*FP
    Le terme X*P est une constante non attribuable à une feature (n'affecte que le benchmark
    XPER, pas les contributions par feature). Décomposer P&L revient donc EXACTEMENT (pas une
    approximation) à décomposer un coût de mauvaise classification avec un coût de FN de
    (X+Z) et un coût de FP de Y.

    À cause du bug vérifié de XPER (CFP/CFN inversés en interne -- voir compute_xper), pour
    obtenir un coût de FN = (X+Z) et un coût de FP = Y dans le calcul réel, il faut appeler la
    librairie avec CFP=(X+Z) et CFN=Y (et NON l'inverse, malgré ce que suggèrent les noms).
    C'est ce que fait cette fonction -- ne pas "corriger" cet appel sans relire la vérification
    empirique documentée dans compute_xper.
    """
    return compute_xper(
        model, X_train, y_train, X_test, y_test, feature_names,
        eval_metric="MC", cost_fp=(X + Z), cost_fn=Y,
        sample_size=sample_size, n_coalition_sampled=n_coalition_sampled, seed=seed,
        top_n=top_n)


# ===================================================================== #
#  Partie 3 -- optimisation économique (P&L)                            #
# ===================================================================== #

def calculate_pnl(y_true, y_pred_proba, threshold, X, Y, Z=0.0):
    """P&L calculé au niveau de la décision INDIVIDUELLE (dec), pas du vrai match mutuel
    (match = A ET B disent oui) -- simplification assumée et documentée (cf.
    docs/soutenance_notes.md) : croiser deux décisions d'une même paire demanderait de
    revenir aux deux lignes (A->B et B->A) et sort du cadre "un score, une recommandation".

    +X si le profil est recommandé (proba > threshold) ET dec == 1                  (TP)
    -Y si le profil est recommandé (proba > threshold) ET dec == 0                  (FP)
    -Z si le profil N'EST PAS recommandé (proba <= threshold) ET dec == 1           (FN,
       "occasion manquée" -- valorisée depuis l'alignement sur les estimations de Max,
       Z=0 dans notre toute première version, cf. data/economic_assumptions.json)
     0 si le profil n'est pas recommandé ET dec == 0 (décision correcte, pas de coût)  (TN)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    recommended = y_pred_proba > threshold
    tp = recommended & (y_true == 1)
    fp = recommended & (y_true == 0)
    fn = (~recommended) & (y_true == 1)
    return X * float(np.sum(tp)) - Y * float(np.sum(fp)) - Z * float(np.sum(fn))


def _refit_predict(refit_kind, fold_train, fold_val, features, seed=SEED):
    """Ré-entraîne logit/xgb sur fold_train (mêmes hyperparamètres que
    01_data_models_v0.py) et retourne predict_proba sur fold_val. N'existe pas pour
    TabICL (voir optimize_threshold_groupkfold)."""
    if refit_kind == "logit":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=5000))
    elif refit_kind == "xgb":
        model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
    else:
        raise ValueError(f"Pas de ré-entraînement possible pour {refit_kind}")
    model.fit(fold_train[features], fold_train["dec"])
    return model.predict_proba(fold_val[features])[:, 1]


def optimize_threshold_groupkfold(train, X, Y, Z=0.0, features=None, proba_col=None,
                                   refit_kind=None, thresholds=None, n_splits=5, seed=SEED):
    """Grille de seuils (0.05 à 0.95, pas 0.05) optimisée sur TRAIN via GroupKFold
    (5 folds sur wave, même découpage que l'AUC). Retourne (seuil_optimal, DataFrame des
    P&L moyens par seuil).

    Modèles refittable=True (logit/xgb, base ou mitigés) : `features` + `refit_kind`
    requis, ré-entraîne le modèle à chaque fold (vrai GroupKFold).
    TabICL (refittable=False) : `proba_col` requis (colonne de proba déjà calculée dans
    `train`, un seul entraînement fixe) -- PAS un vrai ré-entraînement par fold, juste une
    évaluation du P&L restreinte aux indices de validation de chaque fold sur ces
    prédictions figées. Moins rigoureux : à documenter comme limite si cité en soutenance.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    gkf = GroupKFold(n_splits=n_splits)
    groups = train["wave"]
    pnl_per_threshold = {t: [] for t in thresholds}

    for tr_idx, val_idx in gkf.split(train, train["dec"], groups=groups):
        fold_val = train.iloc[val_idx]
        if refit_kind is None:
            val_proba = fold_val[proba_col].to_numpy()
        else:
            fold_train = train.iloc[tr_idx]
            val_proba = _refit_predict(refit_kind, fold_train, fold_val, features, seed=seed)
        y_val = fold_val["dec"].to_numpy()
        for t in thresholds:
            pnl_per_threshold[t].append(calculate_pnl(y_val, val_proba, t, X, Y, Z))

    grid = pd.DataFrame({
        "threshold": thresholds,
        "mean_pnl": [np.mean(pnl_per_threshold[t]) for t in thresholds],
        "std_pnl": [np.std(pnl_per_threshold[t]) for t in thresholds],
    })
    best_threshold = float(grid.loc[grid["mean_pnl"].idxmax(), "threshold"])
    return best_threshold, grid


def run_robustness_test(train, test, model_specs, scenarios, n_splits=5, seed=SEED):
    """Refait l'optimisation de seuil pour plusieurs scénarios (X, Y, Z), en réécrivant
    TEMPORAIREMENT data/economic_assumptions.json à chaque scénario (jamais de valeur X/Y/Z
    en dur dans le code -- seul ce fichier est la source de vérité), puis restaure les
    valeurs d'origine à la fin (try/finally).

    model_specs : dict {name: spec} -- voir build_model_specs().
    scenarios : liste de dicts {"X":.., "Y":.., "Z":.., "label":..}
    Retourne un DataFrame : scenario, modele, seuil_optimal, pnl_test_seuil_optimal.
    """
    original = load_economic_assumptions()
    rows = []
    try:
        for scenario in scenarios:
            save_economic_assumptions({
                "X": scenario["X"], "Y": scenario["Y"], "Z": scenario.get("Z", 0.0),
                "note": f"scénario robustesse temporaire : {scenario['label']}",
            })
            assumptions = load_economic_assumptions()
            X, Y, Z = assumptions["X"], assumptions["Y"], assumptions.get("Z", 0.0)
            for name, spec in model_specs.items():
                best_t, _ = optimize_threshold_groupkfold(
                    train, X, Y, Z, features=spec["features"], proba_col=spec["proba_col"],
                    refit_kind=spec["refit_kind"], n_splits=n_splits, seed=seed)
                pnl_test = calculate_pnl(test["dec"], test[spec["proba_col"]], best_t, X, Y, Z)
                rows.append({"scenario": scenario["label"], "X": X, "Y": Y, "Z": Z,
                            "model": name, "optimal_threshold": best_t,
                            "pnl_test_at_optimal": pnl_test})
    finally:
        save_economic_assumptions(original)
    return pd.DataFrame(rows)


# ===================================================================== #
#  Orchestration -- calcule tout et sauvegarde reports/performance/     #
# ===================================================================== #

def run_full_analysis(output_dir=None, include_mitigated=True, run_xper=True,
                       xper_metrics=("AUC", "MC"), xper_sample_size=100,
                       xper_n_coalition_sampled=150, robustness_scenarios=None, seed=SEED):
    """Calcule les parties 1, 2, 3 pour les modèles de base ET (si include_mitigated) les
    modèles mitigés de Max, sauvegarde tout dans reports/performance/. Retourne un dict avec
    tous les résultats en mémoire (utilisé par notebooks/05_performance.ipynb)."""
    output_dir = Path(output_dir) if output_dir else (ROOT / "reports" / "performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    data, feature_dict = load_model_predictions()
    mitigated_features = None
    if include_mitigated:
        data, mitigated_features = load_mitigated_predictions(data)
    train, test, split = split_train_test(data)

    model_specs = build_model_specs(feature_dict, mitigated_features)
    model_names = list(model_specs.keys())

    assumptions = load_economic_assumptions()
    X, Y, Z = assumptions["X"], assumptions["Y"], assumptions.get("Z", 0.0)

    results = {"models": {}, "model_names": model_names}

    # --- Partie 3 : seuil optimal par GroupKFold (train), P&L test au seuil optimal vs 0.5 ---
    for name in model_names:
        spec = model_specs[name]
        best_t, grid = optimize_threshold_groupkfold(
            train, X, Y, Z, features=spec["features"], proba_col=spec["proba_col"],
            refit_kind=spec["refit_kind"], seed=seed)
        grid.to_csv(output_dir / f"threshold_grid_{name}.csv", index=False)

        proba_test = test[spec["proba_col"]]
        pnl_optimal = calculate_pnl(test["dec"], proba_test, best_t, X, Y, Z)
        pnl_default = calculate_pnl(test["dec"], proba_test, 0.5, X, Y, Z)
        results["models"].setdefault(name, {})
        results["models"][name].update({
            "optimal_threshold": best_t,
            "pnl_test_optimal": pnl_optimal,
            "pnl_test_default_05": pnl_default,
        })

    # --- Partie 1 : PR-AUC, calibration, matrice de confusion au seuil optimal ---
    for name in model_names:
        spec = model_specs[name]
        y_test, proba_test = test["dec"], test[spec["proba_col"]]
        results["models"][name]["pr_auc"] = pr_auc(y_test, proba_test)
        calib = calibration_table(y_test, proba_test)
        calib.to_csv(output_dir / f"calibration_{name}.csv", index=False)
        results["models"][name]["confusion_at_optimal"] = confusion_metrics(
            y_test, proba_test, results["models"][name]["optimal_threshold"])

    # --- Partie 2 : XPER (logit/xgb, base + mitigés -- tabicl non calculable en local) ---
    xper_results = {}
    if run_xper:
        model_objects = {
            "logit": joblib.load(MODELS_DIR / "logit.joblib"),
            "xgb": joblib.load(MODELS_DIR / "xgb.joblib"),
        }
        if include_mitigated:
            model_objects["logit_mitigated"] = joblib.load(MODELS_DIR / "logit_mitigated.joblib")
            model_objects["xgb_mitigated"] = joblib.load(MODELS_DIR / "xgb_mitigated.joblib")

        for name, model in model_objects.items():
            feats = model_specs[name]["features"]
            xper_results.setdefault(name, {})
            for metric in xper_metrics:
                cost_kwargs = {}
                if metric == "MC":
                    # CFP/CFN inversés dans XPER (voir compute_xper) : cost_fp=Z, cost_fn=Y
                    # applique bien Y au FP et Z au FN, comme documenté dans mitigation.py.
                    cost_kwargs = {"cost_fp": Z, "cost_fn": Y}
                res = compute_xper(
                    model, train[feats], train["dec"], test[feats], test["dec"],
                    feature_names=feats, eval_metric=metric,
                    sample_size=xper_sample_size, n_coalition_sampled=xper_n_coalition_sampled,
                    seed=seed, **cost_kwargs)
                xper_results[name][metric] = res
                res["contributions"].to_csv(
                    output_dir / f"xper_{metric.lower()}_{name}.csv", index=False)
        xper_results["tabicl"] = {"status": "not_computed",
                                  "reason": "modèle non chargeable en local (segfault CPU) ; "
                                            "nécessite des centaines d'appels predict_proba, "
                                            "pas seulement des prédictions figées -- tenté sur "
                                            "Colab via colab/xper_tabicl.py, résultat non garanti"}
        if include_mitigated:
            xper_results["tabicl_mitigated"] = {"status": "not_computed",
                                                "reason": "même contrainte que tabicl (modèle "
                                                          "CUDA-only, non chargeable en local)"}
    results["xper"] = xper_results

    # --- Robustesse : scénarios alternatifs de X/Y/Z ---
    if robustness_scenarios is None:
        robustness_scenarios = [
            {"X": 1, "Y": 1, "Z": 1, "label": "X=1/Y=1/Z=1"},
            {"X": 3, "Y": 0.3, "Z": 0.3, "label": "X=3/Y=0.3/Z=0.3"},
        ]
    robustness = run_robustness_test(train, test, model_specs, robustness_scenarios, seed=seed)
    robustness.to_csv(output_dir / "robustness_test.csv", index=False)
    results["robustness"] = robustness

    # --- Tableau récapitulatif ---
    summary_rows = []
    for name in model_names:
        m = results["models"][name]
        top_features = None
        if run_xper and name in xper_results and "AUC" in xper_results.get(name, {}):
            top_features = xper_results[name]["AUC"]["top_features"]["feature"].tolist()
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
            "top_xper_auc_features": ", ".join(top_features[:5]) if top_features else "n/a",
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_all_models.csv", index=False)
    results["summary"] = summary

    return results
