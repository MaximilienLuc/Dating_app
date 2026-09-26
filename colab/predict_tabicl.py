"""
HEC Match -- Génère data/tabicl_predictions.parquet sur Colab (script autonome, aucune
dependance a src/).

Pourquoi ce fichier : TabICL ne charge pas / ne s'entraine pas sur les machines de l'equipe
(segfault CPU, voir docs/soutenance_notes.md "TabICL -- diagnostic du crash CPU et decision").
Ce script calcule predict_proba() UNE FOIS sur Colab (modele deja entraine, models/tabicl.joblib)
et exporte les probabilites pour toutes les lignes -- pour que Remi (stabilite) puisse les
utiliser sans avoir besoin de reentrainer TabICL lui-meme.

LIMITE IMPORTANTE, a transmettre : ce fichier ne contient que les predictions d'UN SEUL modele
(celui entraine sur le train complet). Il permet l'analyse de stabilite cote TEST de Remi
(bootstrap de sessions sur des predictions deja figees -- ce qu'il fait deja pour logit/xgb
dans la boucle `evaluations`, sans reentrainement). Il NE PERMET PAS l'analyse de stabilite
cote TRAIN (les 200 refits sur echantillons bootstrap de train, qui redonnent un vecteur
d'importance a chaque fois) -- ca necessiterait de reentrainer TabICL 200 fois sur Colab, pas
encore fait. A clarifier avec Remi : son analyse TabICL sera donc partielle (incertitude du
test set), pas complete (pas de distance de vecteur d'importance bootstrap), sauf si on relance
un script dedie pour ca plus tard.

A UPLOADER MANUELLEMENT dans Colab avant de lancer :
  - data/clean.parquet   -> clean.parquet
  - data/features.json   -> features.json
  - models/tabicl.joblib -> tabicl.joblib   (le modele deja entraine, pas besoin de refit)

Runtime Colab requis : GPU (le joblib est verrouille sur l'etat CUDA, meme pour predict_proba).

PYARROW : plus besoin d'epingler de version ici (requirements.txt est passe a pyarrow==25.0.1
en local suite a l'incident sur tabicl_mitigated_predictions.parquet -- ecrit par le pyarrow par
defaut de Colab, illisible par l'ancien pyarrow 19 local : "Repetition level histogram size
mismatch"). Un pyarrow recent en local lit sans probleme un parquet ecrit par un pyarrow plus
ancien (compatibilite ascendante) -- seul le sens inverse posait probleme. Si jamais l'erreur
revient malgre tout, mettre a jour pyarrow en local (`pip install -U pyarrow`) plutot que de
repingler Colab a une vieille version.
"""
import json
import numpy as np
import pandas as pd
import joblib

data = pd.read_parquet("clean.parquet")
feature_dict = json.load(open("features.json"))
FEATURES = feature_dict["features"]

print(f"data: {data.shape} | features: {len(FEATURES)}")

model = joblib.load("tabicl.joblib")
print("modele charge OK")

X = data[FEATURES]
proba = model.predict_proba(X)[:, 1]

out = data[["iid", "pid", "wave"]].copy()
out["tabicl_proba"] = proba
out.to_parquet("tabicl_predictions.parquet", index=False)

print(f"tabicl_predictions.parquet ecrit : {out.shape}")
print(out.head().to_string())
