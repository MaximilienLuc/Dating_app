"""
HEC Match -- TabICL MITIGE (sans proxies raciaux) sur Colab.

Ce script est la version debiaisee de colab/train_tabicl.py.
Il retire les 4 variables proxy identifiees par FPDP (Fairness Partial Dependence Plot)
qui etaient a l'origine du biais racial dans les predictions TabICL de base.

Variables retirees (PROXY_FEATURES_TO_DROP) :
  - attr3_1_B    : auto-evaluation d'attractivite du candidat B (proxy race via preferences)
  - career_c_B_12: code carriere specifique (disparite groupe Autres)
  - go_out_B     : frequence de sorties (disparite groupe Autres)
  - intel3_1_B   : auto-evaluation d'intelligence du candidat B (disparite groupe Autres)

Fichiers a uploader dans Colab AVANT de lancer
(icone dossier a gauche > uploader, depuis le repo local) :
  - data/clean.parquet        -> clean.parquet
  - data/features.json        -> features.json
  - data/split.json           -> split.json

Fichiers produits par ce script (a telecharger depuis Colab) :
  - tabicl_mitigated.joblib               -> a mettre dans models/
  - tabicl_mitigated_predictions.parquet  -> a mettre dans data/

IMPORTANT -- runtime Colab requis : Modifier > Parametres du notebook > GPU (T4 ou mieux).
TabICLClassifier.fit() (tabicl==2.2.0) segfault de facon reproductible sur CPU.
"""
!pip install -q tabicl

import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
import joblib
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

SEED = 42

# 4 proxies raciaux identifies par FPDP dans src/mitigation.py
PROXY_FEATURES_TO_DROP = [
    "attr3_1_B",      # auto-evaluation attractivite candidat B
    "career_c_B_12",  # code carriere specifique candidat B
    "go_out_B",       # frequence de sorties candidat B
    "intel3_1_B",     # auto-evaluation intelligence candidat B
]

# ---------------------------------------------------------------------------
print("=== Environnement ===")
print("platform:", platform.platform(), "|", platform.machine())
print("python:", platform.python_version())
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("sklearn:", sklearn.__version__, "| tabicl:", importlib.metadata.version("tabicl"))
assert torch.cuda.is_available(), (
    "Pas de GPU disponible -- Modifier > Parametres du notebook > GPU, puis relancer."
)
print()

# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------
data = pd.read_parquet("clean.parquet")
feature_dict = json.load(open("features.json"))
split = json.load(open("split.json"))

TARGET = feature_dict["target"]

# Features de base (comme XGBoost, pas features_logit -- TabICL n'est pas sensible a la
# collinearite des 6 preferences). On retire ensuite les 4 proxies raciaux.
FEATURES_BASE = feature_dict["features"]  # 158 features
FEATURES_MIT = [f for f in FEATURES_BASE if f not in PROXY_FEATURES_TO_DROP]  # 154 features

# Verifier que les proxies etaient bien presents dans la liste de base
dropped = [f for f in PROXY_FEATURES_TO_DROP if f in FEATURES_BASE]
missing = [f for f in PROXY_FEATURES_TO_DROP if f not in FEATURES_BASE]
print(f"Features de base : {len(FEATURES_BASE)} | Features mitigees : {len(FEATURES_MIT)}")
print(f"Proxies retires ({len(dropped)}) : {dropped}")
if missing:
    print(f"[ATTENTION] Proxies introuvables dans features.json (deja absents?) : {missing}")
print()

# ---------------------------------------------------------------------------
# 2. Split identique (via split.json)
# ---------------------------------------------------------------------------
train_waves, test_waves = set(split["train_waves"]), set(split["test_waves"])
assert not (train_waves & test_waves), "waves communes train/test -- probleme de split.json"

train = data[data["wave"].isin(train_waves)].reset_index(drop=True)
test  = data[data["wave"].isin(test_waves)].reset_index(drop=True)

X_tr, y_tr = train[FEATURES_MIT], train[TARGET]
X_te, y_te = test[FEATURES_MIT],  test[TARGET]
print(f"train : {X_tr.shape} | test : {X_te.shape}")

# ---------------------------------------------------------------------------
# 3. Entrainement TabICL mitige sur le train complet
# ---------------------------------------------------------------------------
print("\n=== Entrainement TabICL MITIGE (train complet, device=cuda) ===")
model = make_pipeline(
    SimpleImputer(strategy="median"),
    TabICLClassifier(random_state=SEED, device="cuda"),
)
model.fit(X_tr, y_tr)
print("fit OK")

# ---------------------------------------------------------------------------
# 4. AUC test
# ---------------------------------------------------------------------------
proba_test = model.predict_proba(X_te)[:, 1]
auc_test = roc_auc_score(y_te, proba_test)
print(f"\nAUC test (split unique) : {auc_test:.3f}")
print("  (Reference modele de base non mitige : 0.632)")

# ---------------------------------------------------------------------------
# 5. GroupKFold 5 folds sur wave
# ---------------------------------------------------------------------------
print("\n=== GroupKFold (5 folds sur wave) ===")
X_all, y_all, groups_all = data[FEATURES_MIT], data[TARGET], data["wave"]
gkf = GroupKFold(n_splits=5)
aucs = []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, groups=groups_all), start=1):
    X_tr_f, X_te_f = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    y_tr_f, y_te_f = y_all.iloc[tr_idx], y_all.iloc[te_idx]
    model_f = make_pipeline(
        SimpleImputer(strategy="median"),
        TabICLClassifier(random_state=SEED, device="cuda"),
    )
    model_f.fit(X_tr_f, y_tr_f)
    p_f = model_f.predict_proba(X_te_f)[:, 1]
    auc_f = roc_auc_score(y_te_f, p_f)
    aucs.append(auc_f)
    print(f"Fold {fold} : AUC = {auc_f:.3f}")

aucs = np.array(aucs)
print(f"\nGroupKFold (5 folds) : AUC moyenne = {aucs.mean():.3f} +/- {aucs.std():.3f}")
print("  (Reference modele de base : 0.591 +/- 0.038)")

# ---------------------------------------------------------------------------
# 6. Sauvegarde du modele (CUDA-only)
# ---------------------------------------------------------------------------
joblib.dump(model, "tabicl_mitigated.joblib")
print("\nModele sauvegarde : tabicl_mitigated.joblib (CUDA uniquement)")

# ---------------------------------------------------------------------------
# 7. Export des predictions sur le dataset COMPLET (train + test)
#    avec iid, pid, wave pour jointure dans les scripts d'analyse de fairness
# ---------------------------------------------------------------------------
print("\n=== Export predictions sur dataset complet ===")
X_full = data[FEATURES_MIT]
proba_full = model.predict_proba(X_full)[:, 1]

preds_df = pd.DataFrame({
    "iid":                 data["iid"],
    "pid":                 data["pid"],
    "wave":                data["wave"],
    "tabicl_mit_proba":    proba_full,
})
preds_df.to_parquet("tabicl_mitigated_predictions.parquet", index=False)
print(f"Predictions exportees : {len(preds_df)} lignes -> tabicl_mitigated_predictions.parquet")

# ---------------------------------------------------------------------------
# 8. Telecharger les deux fichiers produits depuis Colab
# ---------------------------------------------------------------------------
from google.colab import files
files.download("tabicl_mitigated.joblib")
files.download("tabicl_mitigated_predictions.parquet")
print("\nTelechargement lance. Placer les fichiers dans :")
print("  tabicl_mitigated.joblib              -> match_HEC/models/")
print("  tabicl_mitigated_predictions.parquet -> match_HEC/data/")
