#!/bin/bash

# Installer les dépendances avec pip en mode minimal
pip install --no-cache-dir -r requirements.txt

# Nettoyer les fichiers inutiles
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type d -name "tests" -exec rm -r {} +
find . -type d -name "test" -exec rm -r {} +
find . -type d -name "docs" -exec rm -r {} +
find . -type d -name "examples" -exec rm -r {} +

# Créer le dossier model s'il n'existe pas
mkdir -p model 