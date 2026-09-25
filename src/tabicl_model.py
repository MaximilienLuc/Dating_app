"""
HEC Match — modèle TabICL (Tabular Foundation Model, in-context learning).

Features : feature_dict["features"] (src/build_dataset.py) -- les mêmes 164 variables brutes
que XGBoost, les 6 préférences complètes (attr1_1...shar1_1) incluses -- pas le sous-ensemble
réduit du logit (feature_dict["features_logit"], retrait shar1_1) : TabICL n'est pas un modèle
linéaire, pas sensible à leur colinéarité parfaite par construction.

Split : identique aux deux autres modèles -- GroupShuffleSplit par wave (seed=42), cf.
data/split.json.

Attributs protégés exclus des features : identique aux deux autres (gender_A/B, race_A,
cand_race, cand_female, same_race -- audit uniquement, jamais en entrée du modèle).

Proxy connu : income_A/income_B ~ race_A/cand_race (ANOVA sur individus uniques, cf.
docs/soutenance_notes.md) -- reste en feature, à surveiller pour l'audit FPDP de Blanquette.

Limites d'interprétabilité propres à ce modèle : TabICL est une boîte noire par in-context
learning (pas d'entraînement au sens classique -- le train set sert de contexte au moment de
l'inférence) -- aucun coefficient, aucune structure d'arbres à inspecter. L'interprétabilité
passe uniquement par des méthodes post-hoc model-agnostic (SHAP, permutation importance), comme
pour XGBoost mais sans même la structure d'arbres en filet de sécurité.

CONTRAINTE OPÉRATIONNELLE CRITIQUE -- à lire avant de toucher à ce fichier :
`TabICLClassifier.fit()` (tabicl==2.2.0) provoque un segfault natif (SIGSEGV) reproductible sur
le chemin CPU de la librairie -- confirmé À LA FOIS sur Mac Apple Silicon (M4, torch 2.14.0) ET
sur Colab en CPU (x86_64). Ce n'est donc PAS un bug spécifique à Apple Silicon : le chemin CPU
de tabicl est cassé partout où on l'a testé. Seul l'entraînement avec device="cuda" sur Colab
fonctionne.

Le .joblib qui en résulte est verrouillé sur l'état CUDA (torch==2.11.0+cu128 lors du test) :
le charger sur une machine sans CUDA plante aussi, dès joblib.load(), avant même predict_proba().
models/tabicl.joblib n'est donc PAS portable -- utilisable uniquement sur Colab (ou toute
machine avec CUDA). Décision d'équipe : accepté, l'usage de TabICL est ponctuel (pas de
réentraînement fréquent), pas besoin qu'il tourne ailleurs pour l'instant.

Ne PAS appeler train_tabicl() sur ce Mac (ou toute machine sans CUDA) -- ça plantera le process
(segfault, pas une exception Python interceptable). Utiliser colab/train_tabicl.py, à coller
dans une cellule Colab avec un runtime CUDA. Voir docs/soutenance_notes.md pour le détail du
diagnostic (recherche binaire sur les colonnes, comparaison synthétique vs réel, etc.).
"""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.build_dataset import SEED


def train_tabicl(X_tr, y_tr, seed=SEED):
    """Entraîne TabICL (pipeline imputer + TabICLClassifier). Ne fonctionne QUE sur une
    machine avec CUDA disponible (voir avertissement ci-dessus) -- ne pas appeler en local."""
    from tabicl import TabICLClassifier  # import différé : la lib ne doit pas être requise
                                          # pour le reste du pipeline (logit/xgb) sur ce Mac

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        TabICLClassifier(random_state=seed, device="cuda"),
    )
    model.fit(X_tr, y_tr)
    return model
