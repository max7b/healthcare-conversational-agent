## Tutoriel faire un assistant conversationnel

Le but de ce repository est de permettre à chacun de se former à l'IA générative sur un cas pratique.

### Explications du cas d'étude

Le cas d'étude est un cas `Santé` pour un hôpital. \
Le dataset avec des données synthétiques de patient.e.s / docteur.e.s peut être retrouvé sur Kaggle : [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)

Leur souhait : fournir un agent conversationnel aux docteur.e.s de l'hôpital pour permettre la recherche.

### Installation du code

Le code fonctionne avec poetry pour la gestion des dépendances. \
Au cas où, un `requirements.txt` est aussi disponible.

Pour installer poetry:
`pip install -U pip setuptools`
`pip install poetry`

Pour ajouter votre `.venv` sur poetry : `poetry env use /full/path/to/.venv/bin/python`. \
(Pour trouver le path, faire un `which python` après avoir activé l'environnement)

### Configuration d'AWS Bedrock

Configurer un profil AWS CLI ayant accès à Bedrock, puis ajouter son nom dans le `.env`.

### Installation des variables d'environnements

Pour faire fonctionner le code, il faut recopier le fichier `.env_template` dans `.env` et y ajouter les variables.

#### Pour faire des modifications dans le code

Il y a des hooks de pre-commit, merci de les utiliser : `pre-commit install` \
Et pour ajouter un package : `poetry add <package_name>`