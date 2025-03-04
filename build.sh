#!/bin/bash

# Afficher la version de Python
python --version

# Créer le dossier model s'il n'existe pas
mkdir -p model

# Nettoyer les fichiers inutiles
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Vérifier l'installation
pip list 