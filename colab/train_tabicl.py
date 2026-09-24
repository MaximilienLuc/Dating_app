"""
HEC Match -- TabICL sur Colab (script autonome, aucune dependance a src/).

A UPLOADER MANUELLEMENT dans Colab avant de lancer (icone dossier a gauche,
ou `from google.colab import files; files.upload()`), depuis le repo local :
  - data/clean.parquet   -> clean.parquet
  - data/features.json   -> features.json
  - data/split.json      -> split.json   (garantit un split train/test IDENTIQUE bit a bit,
                                           sans dependre de la determinisme de
                                           GroupShuffleSplit entre environnements/versions
                                           de sklearn)

Reproduit le meme split par wave et les features COMPLETES du pipeline principal
(feature_dict["features"], PAS le sous-ensemble feature_dict["features_logit"] -- TabICL
garde les 6 preferences comme XGBoost, n'etant pas sensible a leur collinearite). Le nombre
de features suit ce que contient le features.json uploade (158 depuis la correction des
dummies drop_first=True -- pas de valeur codee en dur ici).

IMPORTANT -- runtime Colab requis : Modifier > Parametres du notebook > GPU.
`TabICLClassifier.fit()` (tabicl==2.2.0) segfault de facon reproductible sur CPU -- confirme
a la fois sur Mac Apple Silicon ET sur Colab en CPU (x86_64). Seul device="cuda" fonctionne.
Consequence : le tabicl.joblib produit est verrouille sur l'etat CUDA et ne se charge PAS sur
une machine sans CUDA (plante des joblib.load, avant meme predict_proba). C'est accepte pour ce
projet : usage ponctuel de TabICL via Colab, pas de réentrainement frequent, pas besoin qu'il
tourne ailleurs. Voir src/tabicl_model.py et docs/soutenance_notes.md pour le detail.
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

print("=== Environnement ===")
print("platform:", platform.platform(), "|", platform.machine())
print("python:", platform.python_version())
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("sklearn:", sklearn.__version__, "| tabicl:", importlib.metadata.version("tabicl"))
assert torch.cuda.is_available(), (
    "Pas de GPU disponible sur ce runtime Colab -- device='cuda' est obligatoire "
    "(le chemin CPU de tabicl segfault). Modifier > Parametres du notebook > GPU, puis relancer."
)
print()

# ---------------------------------------------------------------------------
# 1. Chargement (fichiers uploades manuellement, a la racine du runtime Colab)
# ---------------------------------------------------------------------------
data = pd.read_parquet("clean.parquet")
feature_dict = json.load(open("features.json"))
split = json.load(open("split.json"))

FEATURES = feature_dict["features"]  # features completes (comme XGBoost), pas features_logit
TARGET = feature_dict["target"]

print(f"data: {data.shape} | features: {len(FEATURES)}")

# ---------------------------------------------------------------------------
# 2. Split identique a 01_data_models_v0.py, via data/split.json (pas recalcule)
# ---------------------------------------------------------------------------
train_waves, test_waves = set(split["train_waves"]), set(split["test_waves"])
assert not (train_waves & test_waves), "waves communes train/test"

train = data[data["wave"].isin(train_waves)].reset_index(drop=True)
test = data[data["wave"].isin(test_waves)].reset_index(drop=True)

X_tr, y_tr = train[FEATURES], train[TARGET]
X_te, y_te = test[FEATURES], test[TARGET]
print(f"train: {X_tr.shape} | test: {X_te.shape}")

# ---------------------------------------------------------------------------
# 3. Entrainement TabICL sur le train complet
#    (pipeline imputer+TabICL, comme le logit, pour que predict_proba marche
#    directement sur des features brutes -- TabICL ne gere pas les NaN nativement)
# ---------------------------------------------------------------------------
print("\n=== Entrainement TabICL (train complet, device=cuda) ===")
model = make_pipeline(
    SimpleImputer(strategy="median"),
    TabICLClassifier(random_state=SEED, device="cuda"),
)
model.fit(X_tr, y_tr)
print("fit OK")

# ---------------------------------------------------------------------------
# 4. AUC test (meme methode que logit/xgb dans 01_data_models_v0.py)
# ---------------------------------------------------------------------------
proba_test = model.predict_proba(X_te)[:, 1]
auc_test = roc_auc_score(y_te, proba_test)
print(f"\nAUC test (split unique) : {auc_test:.3f}")

# ---------------------------------------------------------------------------
# 5. GroupKFold 5 folds sur wave (meme protocole que l'audit de 01_data_models_v0.py)
# ---------------------------------------------------------------------------
print("\n=== GroupKFold (5 folds sur wave) ===")
X_all, y_all, groups_all = data[FEATURES], data[TARGET], data["wave"]
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

# ---------------------------------------------------------------------------
# 6. Sauvegarde du modele (celui entraine sur le train complet, etape 3)
#    NOTE : ce fichier ne se chargera QUE sur une machine avec CUDA (voir en-tete).
# ---------------------------------------------------------------------------
joblib.dump(model, "tabicl.joblib")
print("\nModele sauvegarde : tabicl.joblib (CUDA uniquement -- ne se chargera pas hors Colab/GPU)")
