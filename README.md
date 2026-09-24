# match_HEC
Dating app for HEC students

## Installation

Créer un venv et installer les dépendances : `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

Le dataset brut (`data/Speed Dating Data.csv` et son dictionnaire `data/Speed Dating Data Key.doc`) n'est pas versionné (voir `.gitignore`) : à télécharger depuis Kaggle (« Speed Dating Experiment ») et à placer manuellement dans `data/`.

## Stability (Rémi)

[Notebook with results](04_stability.executed.ipynb) · [Notebook source](04_stability.ipynb) · [Method and app integration](reports/stability/README.md)

Session-level bootstrap of the v0 logit and XGBoost. TabPFN and the final models still need to be integrated. Reproduce the analysis with `python -m src.stability`; see `requirements-stability.txt` for dependencies.
