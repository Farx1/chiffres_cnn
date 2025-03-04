#!/bin/bash

# Afficher la version de Python
python --version

# Mettre à jour pip
python -m pip install --upgrade pip

# Installer d'abord les dépendances de build
pip install --no-cache-dir -r requirements-build.txt

# Installer les dépendances principales
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

# Vérifier l'espace disque disponible
df -h

# Lister les paquets installés
pip list 