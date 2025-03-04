# Classification de Chiffres Manuscrits avec CNN Interactive

Ce projet est une application interactive de classification de chiffres manuscrits utilisant un réseau de neurones convolutif (CNN). 

## Fonctionnalités

- Classification des chiffres du dataset MNIST
- Interface de dessin interactive pour tester vos propres chiffres
- Visualisation des activations des couches du CNN
- Interface utilisateur moderne avec Streamlit

## Installation

1. Cloner le repository
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application :
```bash
streamlit run app.py
```

## Structure du Projet

- `app.py` : Application principale Streamlit
- `model.py` : Définition et entraînement du CNN
- `utils.py` : Fonctions utilitaires
- `visualize.py` : Fonctions de visualisation des activations 