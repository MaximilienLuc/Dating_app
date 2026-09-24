# match_HEC
Dating app for HEC students

## Installation

Créer un venv et installer les dépendances : `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

Le dataset brut (`data/Speed Dating Data.csv` et son dictionnaire `data/Speed Dating Data Key.doc`) n'est pas versionné (voir `.gitignore`) : à télécharger depuis Kaggle (« Speed Dating Experiment ») et à placer manuellement dans `data/`.

## TabICL (modèle 3/3) — nécessite Colab

`TabICLClassifier.fit()` (`tabicl==2.2.0`) provoque un segfault natif (SIGSEGV) reproductible sur le chemin CPU de la librairie — confirmé à la fois sur Mac Apple Silicon (M4, torch 2.14.0) et sur Colab en CPU (x86_64). **Ce n'est donc pas un bug spécifique à Apple Silicon** : le chemin CPU de `tabicl` est cassé partout où on l'a testé à ce jour. Seul l'entraînement avec `device="cuda"` sur Colab fonctionne.

Le `models/tabicl.joblib` qui en résulte est verrouillé sur l'état CUDA (`torch==2.11.0+cu128` lors du test) : le charger sur une machine sans CUDA plante aussi, dès `joblib.load()`, avant même `predict_proba()`. **`models/tabicl.joblib` n'est donc pas portable** — utilisable uniquement sur Colab (ou toute machine avec CUDA). Décision d'équipe : accepté, l'usage de TabICL est ponctuel (pas de réentraînement fréquent), pas besoin qu'il tourne ailleurs pour l'instant.

Pour (ré)entraîner TabICL : coller `colab/train_tabicl.py` dans une cellule Colab avec un runtime GPU activé (Modifier > Paramètres du notebook > GPU), après avoir uploadé `data/clean.parquet`, `data/features.json` et `data/split.json`. Voir `docs/soutenance_notes.md` pour le détail du diagnostic (recherche binaire sur les colonnes, comparaison données réelles vs synthétiques, etc.).

## Stability (Rémi)

[Notebook with results](04_stability.executed.ipynb) · [Notebook source](04_stability.ipynb) · [Method and app integration](reports/stability/README.md)

Session-level bootstrap: logit/XGBoost training sensitivity and three-model test uncertainty using frozen TabICL predictions. Reproduce the analysis with `python -m src.stability`; see `requirements-stability.txt` for dependencies.

