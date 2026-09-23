*Contexte : projet de groupe HEC, cours « Interpretability, Stability, and Algorithmic Fairness » (Pr. Christophe Pérignon et Dr Sébastien Saurin, MSc DSAIB, septembre 2026).*

*L'énoncé.* Faire une analyse de scoring (cible binaire 0/1) pour un client fictif, en comparant trois modèles : un white box (logit, idéalement PLTR ou AdaLogit, vus en cours), un modèle de ML (XGBoost) et un Tabular Foundation Model (TabPFN). Pour chacun, évaluer quatre dimensions :
- la performance, statistique et *économique* ;
- l'interprétabilité (globale et locale) ;
- la stabilité ;
- la fairness.

Il faut discuter les arbitrages entre ces dimensions et recommander un modèle au client selon une logique d'IA de confiance, pas juste la meilleure AUC.

*Livrables et échéances :*
- Pré-validation du dataset par mail au prof avant *jeudi 24 septembre, 9h40*.
- Envoi par mail avant *lundi 28 septembre, 9h40* : un deck de slides, un notebook complet (de la préparation des données à l'évaluation) et une application ou un site où le client teste les trois modèles.
- Soutenance le lundi 28 : 15 minutes de présentation, 10 minutes de questions. Chaque membre peut être interrogé sur n'importe quelle partie, code compris, et les notes peuvent différer entre membres.
- Notation du projet sur 25 : technique /5, présentation /5, Q&A /5, slides + code + app /10.

*Le sujet choisi : un algorithme d'app de rencontre.*
- *Client fictif* : « HEC Match », une app qui décide quels profils montrer à chaque utilisateur.
- *Accroche* : la slide 11 du cours cite « Love » parmi les « life-changing algorithms ».
- *Dataset* : Speed Dating Experiment (Fisman et Iyengar, Columbia Business School, 2002–2004), sur Kaggle avec son dictionnaire Speed Dating Data Key.doc, ou OpenML 40536. Il compte 8 378 lignes et 195 variables brutes, issues de 21 sessions avec des rendez-vous de quatre minutes. Chaque ligne correspond à un participant sur un rendez-vous, donc chaque rencontre apparaît deux fois.
- *Cible* : dec (A veut revoir B, environ 40 % de oui). Une ligne est un couple (A qui décide, B le candidat). match (réciproque, environ 16,5 %) sert d'indicateur économique.

*Les choix méthodologiques clés :*
1. *Fuite d'information* : n'utiliser que les variables connues *avant* le rendez-vous. On garde le questionnaire d'inscription (suffixe _1) : âge, études, carrière, objectif, fréquence de sorties, centres d'intérêt, préférences attr1_1`…shar1_1`, auto-évaluation attr3_1`…amb3_1`, revenu, int_corr, plus des variables de couple (écart d'âge, adéquation). On exclut toutes les notes données pendant ou après la soirée (attr, sinc, like, prob, les variables *_o, match_es, *_s, *_2, *_3, dec_o, match) et l'ordre des rendez-vous. Conséquence attendue : une AUC modeste, autour de 0,65–0,70, à présenter comme un résultat (« on ne prédit pas l'alchimie à partir d'un profil »). Option : montrer aussi un modèle « après rendez-vous » pour illustrer la fuite.
2. *Découpage train/test par session* (wave), avec GroupShuffleSplit ou GroupKFold et une graine de 42. Découper par participant ne suffit pas, car les paires (A, B) et (B, A) fuient entre train et test.
3. *Préférences des sessions 6 à 9* : notées de 1 à 10 au lieu de 100 points, donc à renormaliser en parts de 100.
4. *Fairness, en termes d'équité d'exposition* : le profil de B est-il moins souvent recommandé à cause de son genre ou de son origine ? Les attributs protégés (cand_female, cand_race, same_race, gender_A, race_A) servent uniquement à l'audit, jamais en entrée. Le débat central : reproduire les préférences des utilisateurs, y compris raciales (documentées par Fisman et Iyengar), ou les corriger, au prix de la précision. Méthodes du cours à appliquer : tests de fairness, FPDP pour trouver les variables proxy, mitigation, test d'équivalence TOST. Deux points restent à trancher : garder ou non imprace, et le fait que le genre de A est exclu mais peut être deviné via les préférences.
5. *P&L construit* : un profil montré qui reçoit un oui rapporte +X €, un profil montré qui reçoit un non coûte −Y €, un match manqué est un revenu perdu. On optimise le seuil sur le P&L et on fait une analyse de sensibilité à X et Y.
6. *Stabilité* : entraîner sur certaines sessions et tester sur d'autres, plus du bootstrap. On mesure les distances entre coefficients et entre vecteurs d'importance, et le pourcentage de décisions qui basculent.
7. *Limites à annoncer* : données anciennes, étudiants de Columbia uniquement, beaucoup de valeurs manquantes (87 % des lignes en ont au moins une, 1,8 % des cases), origine utilisée seulement pour l'audit.

*L'équipe (5 personnes) :*
- *Alex (moi)* : données, variables, découpage, les trois modèles, performance et P&L.
- *Max* : interprétabilité (coefficients, SHAP, LIME, PDP/ICE, permutation importance, XPER). Aujourd'hui, il envoie aussi le mail de pré-validation et code src/metrics.py.
- *Blanquette* : fairness (tests, FPDP, mitigation, TOST) et récit de la soutenance. Aujourd'hui, exploration des taux de oui par genre et par origine.
- *Remi* (absent le premier jour) : stabilité.
- *Oli* (absent le premier jour) : app Streamlit déployée avec un lien public (choisir deux profils, obtenir la probabilité avec les trois modèles, l'explication SHAP, un onglet fairness et un onglet stabilité) et template du deck.

*Le contrat de fichiers, pour travailler en parallèle :*
- data/clean.parquet, data/features.json (cible, variables, attributs protégés, variables limites, règle d'exclusion), data/split.json ;
- models/logit.joblib, xgb.joblib, tabpfn.joblib, tous avec predict_proba ;
- src/metrics.py.
- Un notebook par bloc : 01_data_models (Alex), 02_interpretability (Max), 03_fairness (Blanquette), 04_stability (Remi), plus un dossier app/ (Oli).
- Les noms ne changent plus. La liste des variables est figée samedi matin au plus tard.

*L'état actuel.* J'ai un script v0, 01_data_models_v0.py, *non testé*. Il charge le CSV (encodage ISO-8859-1), construit un profil par participant, renormalise les préférences, fusionne les profils de A et B sur iid`/pid`, crée les variables de couple et les attributs protégés, encode field_cd, career_c et goal en variables indicatrices, découpe par session (25 % en test), sauvegarde le contrat de fichiers et entraîne un logit (imputation, standardisation, régression logistique), XGBoost et TabPFN, avec gestion d'erreur si TabPFN échoue sur CPU.

*Les prochaines étapes pour moi :* faire tourner et déboguer la v0, la pousser et prévenir le groupe, puis valider le tri des variables avec le dictionnaire, remplacer le logit par PLTR ou AdaLogit (package trust-free), régler XGBoost par validation croisée groupée par session, et faire tourner TabPFN sur GPU (Colab) ou via tabpfn-client si besoin.

*Le planning :* jeudi et vendredi, analyses par bloc ; samedi, intégration et branchement des vrais modèles dans l'app ; dimanche, slides, deux répétitions chronométrées et formation croisée pour le Q&A ; lundi 9h40, envoi.

---

Pense à joindre aussi le fichier 01_data_models_v0.py à la nouvelle conversation, pour qu'elle ait le code sous les yeux.