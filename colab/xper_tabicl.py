"""
HEC Match -- XPER sur TabICL (AUC et P&L), sur Colab (script autonome, aucune dependance a src/).

VERDICT FINAL (26/09) -- NE PAS RELANCER : tente 3 fois. Les 2 premieres (N_coalition_sampled=
300 puis 15) ont fait planter le kernel Colab par saturation memoire. La 3e (N_coalition_
sampled=3, sample_size=10, parametres actuels du fichier) a termine sans planter mais produit
un resultat statistiquement degenere -- seulement 8 valeurs uniques sur 158 features dans les
deux metriques (beaucoup de features a exactement 0), un artefact du sous-echantillonnage
extreme plutot qu'un vrai signal. XPER-TabICL est acte comme non exploitable en pratique sur ce
tier Colab -- voir docs/soutenance_notes.md "Performance predictive". Fichiers de la tentative
gardes dans reports/performance/xper_{pnl,auc}_tabicl.csv, explicitement annotes comme non
exploitables. Ne pas relancer sans un changement structurel (ex. sous-echantillonner fortement
le train comme contexte -- change alors le modele evalue, a documenter comme tel).

ATTENTION avant de lancer : XPER interroge le modele reel des CENTAINES/MILLIERS de fois
(echantillonnage de coalitions de features masquees), pas juste une fois comme
predict_tabicl.py. Sur logit (rapide, local), meme avec des parametres reduits, ca prend deja
~20s pour ~156-158 features. TabICL est un modele en in-context learning (un appel
predict_proba = un forward pass sur tout le train comme contexte) -- chaque coalition risque
d'etre beaucoup plus lente ET plus gourmande en memoire qu'un logit.

CAUSE CONFIRMEE D'UN 1ER CRASH (OOM, kernel redemarre) : XPER paralllise le calcul des
coalitions avec un ThreadPoolExecutor a max_workers=60, CODE EN DUR dans la librairie
(XPER/compute/EM.py, pas configurable via l'API publique). Avec TabICL, chaque tache concurrente
= un forward pass complet du transformer sur tout le train comme contexte -- jusqu'a 60
d'entre eux simultanement peut saturer la RAM/VRAM disponible sur Colab, meme en GPU. Le seul
levier disponible : N_coalition_sampled borne le nombre de taches total, donc le borner
suffisamment bas (en dessous de 60) borne aussi le nombre de forward pass concurrents reels.
Parametres ci-dessous deja reduits en consequence (N_coalition_sampled=15, sample_size=20) --
si ca replante quand meme, documente XPER-TabICL comme non calculable (meme sur Colab GPU)
plutot que d'insister davantage : c'est un point optionnel, pas bloquant pour la soutenance.

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

# DERNIER TEST, extreme : a rejete un 2e crash OOM meme a 15/20 -- ce n'est probablement pas
# (seulement) les 60 threads concurrents, plutot le fait qu'UN SEUL forward pass TabICL avec
# ~6500 lignes de train comme contexte est deja tres lourd (attention transformer scale au
# carre avec la taille du contexte). Si CE test replante aussi, conclusion : XPER-TabICL non
# calculable sur ce tier Colab -- arreter la, documenter, ne pas insister davantage (optionnel,
# non bloquant pour la soutenance).
SAMPLE_SIZE = 10
N_COALITION_SAMPLED = 3

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
print("Surveille la RAM (icone en haut a droite de Colab) pendant l'execution. Si elle sature "
      "a nouveau ou si rien ne s'affiche apres 5-10 minutes, interromps (Runtime > Interrompre "
      "l'execution) et on documente XPER-TabICL comme non calculable plutot que d'insister.")

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
