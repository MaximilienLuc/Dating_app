"""
HEC Match -- XPER sur TabICL, tentative sur Colab (script autonome, aucune dependance a src/).

ATTENTION avant de lancer : XPER interroge le modele reel des CENTAINES/MILLIERS de fois
(echantillonnage de coalitions de features masquees), pas juste une fois comme
predict_tabicl.py. Sur logit (rapide, local), meme avec des parametres reduits, ca a deja
pris plusieurs minutes pour ~156 features. TabICL est un modele en in-context learning (un
appel predict_proba = un forward pass sur tout le train comme contexte) -- chaque coalition
risque d'etre beaucoup plus lente qu'un logit. Pas de garantie que ca termine dans un temps
raisonnable, meme sur GPU Colab. Parametres ci-dessous deja reduits (N_coalition_sampled=300,
sample_size=100) pour limiter le risque -- si c'est encore trop lent, interromps et on
documente XPER-TabICL comme non calculable plutot que d'insister.

A UPLOADER MANUELLEMENT dans Colab avant de lancer :
  - data/clean.parquet   -> clean.parquet
  - data/features.json   -> features.json
  - data/split.json      -> split.json
  - models/tabicl.joblib -> tabicl.joblib

Runtime Colab requis : GPU. pyarrow epingle a 19.0.0 comme pour les autres scripts (voir
colab/predict_tabicl.py pour le detail du probleme rencontre sinon).
"""
!pip install -q "pyarrow==19.0.0" tabicl XPER

import json
import time
import numpy as np
import pandas as pd
import joblib
import pyarrow

assert pyarrow.__version__ == "19.0.0", (
    f"pyarrow {pyarrow.__version__} charge au lieu de 19.0.0 -- redemarre le runtime "
    "(Runtime > Redemarrer la session) puis relance ce script en entier."
)

from XPER.compute.Performance import ModelPerformance

data = pd.read_parquet("clean.parquet")
feature_dict = json.load(open("features.json"))
split = json.load(open("split.json"))
FEATURES = feature_dict["features"]  # TabICL garde les features completes, pas features_logit

train_waves, test_waves = set(split["train_waves"]), set(split["test_waves"])
train = data[data["wave"].isin(train_waves)].reset_index(drop=True)
test = data[data["wave"].isin(test_waves)].reset_index(drop=True)

model = joblib.load("tabicl.joblib")
print("modele charge OK")

X_train = train[FEATURES].to_numpy()
y_train = train["dec"].to_numpy()
X_test = test[FEATURES].to_numpy()
y_test = test["dec"].to_numpy()

# Parametres reduits par rapport aux defauts (voir avertissement en tete de fichier) :
SAMPLE_SIZE = 100
N_COALITION_SAMPLED = 300

print(f"Lancement XPER (sample_size={SAMPLE_SIZE}, N_coalition_sampled={N_COALITION_SAMPLED})...")
print("Si aucune sortie apres 10-15 minutes, interromps (Runtime > Interrompre l'execution) "
      "et on documente XPER-TabICL comme non calculable plutot que d'attendre indefiniment.")

t0 = time.time()
perf = ModelPerformance(X_train, y_train, X_test, y_test, model, sample_size=SAMPLE_SIZE, seed=42)
phi, phi_i_j = perf.calculate_XPER_values(["AUC"], N_coalition_sampled=N_COALITION_SAMPLED, seed=42)
elapsed = time.time() - t0
print(f"XPER termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")

contrib = pd.DataFrame({"feature": FEATURES, "xper_value": phi[1:]})
contrib["abs_xper_value"] = contrib["xper_value"].abs()
contrib = contrib.sort_values("abs_xper_value", ascending=False).reset_index(drop=True)

print(f"\nBenchmark AUC (phi[0]) : {phi[0]:.4f}")
print("\nTop 10 features par contribution XPER :")
print(contrib.head(10).to_string())

contrib.to_csv("xper_values_tabicl.csv", index=False)
print("\nFichier ecrit : xper_values_tabicl.csv -- telecharge-le et donne le chemin pour l'integrer.")
