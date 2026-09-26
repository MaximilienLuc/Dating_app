"""
HEC Match -- XPER sur TabICL (AUC et P&L), sur Colab (script autonome, aucune dependance a src/).

ATTENTION avant de lancer : XPER interroge le modele reel des CENTAINES/MILLIERS de fois
(echantillonnage de coalitions de features masquees), pas juste une fois comme
predict_tabicl.py. Sur logit (rapide, local), meme avec des parametres reduits, ca prend deja
~20s pour ~156-158 features. TabICL est un modele en in-context learning (un appel
predict_proba = un forward pass sur tout le train comme contexte) -- chaque coalition risque
d'etre beaucoup plus lente qu'un logit. Pas de garantie que ca termine dans un temps
raisonnable, meme sur GPU Colab. Parametres ci-dessous deja reduits (N_coalition_sampled=300,
sample_size=100) pour limiter le risque -- si c'est encore trop lent, interromps et on
documente XPER-TabICL comme non calculable plutot que d'insister.

Ce script calcule XPER-PNL en priorite (decomposition EXACTE de notre P&L reel, pas juste l'AUC
-- voir derivation dans src/performance.py::compute_xper_pnl et docs/soutenance_notes.md
"Performance predictive"). XPER-AUC est calcule en second si le temps le permet.

Lance d'abord sur `tabicl.joblib` (base) SEULEMENT. Une fois que t'as confirme que ca termine
dans un temps raisonnable, relance avec MODEL_FILE="tabicl_mitigated.joblib" (section a la fin)
pour le modele mitige -- ne lance pas les deux d'un coup, le risque de timeout est deja
incertain pour un seul.

A UPLOADER MANUELLEMENT dans Colab avant de lancer :
  - data/clean.parquet                  -> clean.parquet
  - data/features.json                  -> features.json
  - data/split.json                     -> split.json
  - data/economic_assumptions.json      -> economic_assumptions.json
  - models/tabicl.joblib                -> tabicl.joblib

Runtime Colab requis : GPU.

PYARROW : pas besoin d'epingler de version (requirements.txt local est a pyarrow==25.0.1 --
lit sans probleme un parquet ecrit par n'importe quelle version de Colab, la compatibilite
ascendante suffit). Voir colab/predict_tabicl.py pour le detail de l'incident precedent.
"""
!pip install -q tabicl XPER

import json
import time
import numpy as np
import pandas as pd
import joblib

from XPER.compute.Performance import ModelPerformance

data = pd.read_parquet("clean.parquet")
feature_dict = json.load(open("features.json"))
split = json.load(open("split.json"))
assumptions = json.load(open("economic_assumptions.json"))
FEATURES = feature_dict["features"]  # TabICL garde les features completes, pas features_logit

X_cost, Y_cost, Z_cost = assumptions["X"], assumptions["Y"], assumptions.get("Z", 0.0)
print(f"Hypotheses economiques : X={X_cost} Y={Y_cost} Z={Z_cost}")

train_waves, test_waves = set(split["train_waves"]), set(split["test_waves"])
train = data[data["wave"].isin(train_waves)].reset_index(drop=True)
test = data[data["wave"].isin(test_waves)].reset_index(drop=True)

MODEL_FILE = "tabicl.joblib"  # <-- change en "tabicl_mitigated.joblib" pour la 2e passe
model = joblib.load(MODEL_FILE)
print(f"modele charge OK : {MODEL_FILE}")

X_train = train[FEATURES].to_numpy()
y_train = train["dec"].to_numpy()
X_test = test[FEATURES].to_numpy()
y_test = test["dec"].to_numpy()

# Parametres reduits par rapport aux defauts (voir avertissement en tete de fichier) :
SAMPLE_SIZE = 100
N_COALITION_SAMPLED = 300

# ---------------------------------------------------------------------------------------
# XPER-PNL : decomposition EXACTE de notre P&L reel (calculate_pnl), pas juste un cout de
# classification brut. Derivation (voir src/performance.py::compute_xper_pnl) :
#   P&L = X*TP - Y*FP - Z*FN = X*P - (X+Z)*FN - Y*FP   (P = positifs reels, constant)
# Decomposer le P&L revient donc exactement a decomposer un cout de classification avec
# cout de FN = (X+Z) et cout de FP = Y.
#
# BUG VERIFIE dans XPER : les parametres CFP/CFN sont INVERSES en interne par rapport a leur
# docstring -- CFP pondere en realite les FAUX NEGATIFS, CFN pondere les FAUX POSITIFS (verifie
# empiriquement, voir src/performance.py::compute_xper pour le detail). Donc pour obtenir
# FN=(X+Z) et FP=Y dans le calcul reel, on passe CFP=(X+Z) et CFN=Y (pas l'inverse).
# ---------------------------------------------------------------------------------------
print(f"\n=== XPER-PNL ({MODEL_FILE}) ===")
print(f"sample_size={SAMPLE_SIZE}, N_coalition_sampled={N_COALITION_SAMPLED}")
print("Si aucune sortie apres 10-15 minutes, interromps (Runtime > Interrompre l'execution) "
      "et on documente XPER-TabICL comme non calculable plutot que d'attendre indefiniment.")

t0 = time.time()
perf = ModelPerformance(X_train, y_train, X_test, y_test, model, sample_size=SAMPLE_SIZE, seed=42)
phi_pnl, _ = perf.calculate_XPER_values(
    ["MC"], CFP=(X_cost + Z_cost), CFN=Y_cost,
    N_coalition_sampled=N_COALITION_SAMPLED, seed=42)
elapsed = time.time() - t0
print(f"XPER-PNL termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")

contrib_pnl = pd.DataFrame({"feature": FEATURES, "xper_value": phi_pnl[1:]})
contrib_pnl["abs_xper_value"] = contrib_pnl["xper_value"].abs()
contrib_pnl = contrib_pnl.sort_values("abs_xper_value", ascending=False).reset_index(drop=True)

print(f"\nBenchmark P&L (phi[0]) : {phi_pnl[0]:.4f}")
print("\nTop 10 features par contribution XPER-PNL :")
print(contrib_pnl.head(10).to_string())

out_name = "xper_pnl_" + MODEL_FILE.replace(".joblib", "") + ".csv"
contrib_pnl.to_csv(out_name, index=False)
print(f"\nFichier ecrit : {out_name}")

# ---------------------------------------------------------------------------------------
# XPER-AUC : seulement si le P&L ci-dessus a termine dans un temps raisonnable. Meme cout de
# calcul attendu (meme nombre de coalitions/sample_size) -- double le temps total.
# ---------------------------------------------------------------------------------------
print(f"\n=== XPER-AUC ({MODEL_FILE}) ===")
t0 = time.time()
phi_auc, _ = perf.calculate_XPER_values(["AUC"], N_coalition_sampled=N_COALITION_SAMPLED, seed=42)
elapsed = time.time() - t0
print(f"XPER-AUC termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")

contrib_auc = pd.DataFrame({"feature": FEATURES, "xper_value": phi_auc[1:]})
contrib_auc["abs_xper_value"] = contrib_auc["xper_value"].abs()
contrib_auc = contrib_auc.sort_values("abs_xper_value", ascending=False).reset_index(drop=True)

print(f"\nBenchmark AUC (phi[0]) : {phi_auc[0]:.4f}")
print("\nTop 10 features par contribution XPER-AUC :")
print(contrib_auc.head(10).to_string())

out_name_auc = "xper_auc_" + MODEL_FILE.replace(".joblib", "") + ".csv"
contrib_auc.to_csv(out_name_auc, index=False)
print(f"\nFichier ecrit : {out_name_auc}")

print("\nTelecharge les 2 fichiers CSV depuis le panneau de fichiers Colab et place-les dans "
      "reports/performance/ en local. Une fois tabicl.joblib confirme dans un temps "
      "raisonnable, change MODEL_FILE = 'tabicl_mitigated.joblib' plus haut, uploade ce joblib "
      "a la place, et relance tout le script pour la 2e passe (mitigee).")
