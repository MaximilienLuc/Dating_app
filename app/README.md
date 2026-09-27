# App HEC Match (Alex)

Page web autonome (aucun serveur) : ouvrir `app/index.html` dans un navigateur, ou le lien publié.

- **Recommandations** : pour un utilisateur du test, ses candidats classés par le modèle choisi ; les 3 premiers
  sont ceux que l'app montrerait. Explications exactes : SHAP (XGBoost mitigé), coefficient × valeur standardisée
  (logit). Option pour révéler les vraies décisions.
- **Simulateur** : le logit et XGBoost mitigé sont recalculés dans le navigateur (`score.js`), écart < 1e-4 avec
  Python sur les 1 826 couples. TabICL : score figé (GPU uniquement).
- **Comparer les modèles** : tableau des arbitrages (`reports/tradeoffs`, `reports/economics`).

Régénérer après un changement de modèle ou de résultats : `python -m app.build` depuis la racine.
Logit de l'app = logit mitigé réentraîné proprement (cf. `02_interpretability`), AUC 0,578 contre 0,580.
