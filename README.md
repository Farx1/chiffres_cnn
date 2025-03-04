# Classification de Chiffres Manuscrits avec CNN Interactive

Application web interactive de classification de chiffres manuscrits utilisant un réseau de neurones convolutif (CNN).

## 🎯 Fonctionnalités

- ✏️ Interface de dessin interactive pour tester le modèle en temps réel
- 🧠 Visualisation des couches d'activation du réseau neuronal
- 📊 Analyse détaillée des probabilités pour chaque chiffre
- 📈 Suivi des performances d'entraînement du modèle

## 🚀 Installation

1. **Cloner le repository**
```bash
git clone https://github.com/Farx1/chiffres_cnn.git
cd chiffres_cnn
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

1. **Lancer l'application**
```bash
streamlit run app.py
```

2. **Premier lancement**
- Lors du premier lancement, le modèle sera automatiquement entraîné
- Cette étape peut prendre quelques minutes selon votre machine
- Le modèle sera ensuite sauvegardé pour les utilisations futures

3. **Utilisation de l'application**
- Dessinez un chiffre dans la zone de dessin
- Cliquez sur "Analyser" pour obtenir la prédiction
- Observez les visualisations des couches du réseau
- Consultez les probabilités pour chaque chiffre

## 🛠️ Technologies utilisées

- Python 3.11
- TensorFlow 2.15 (CPU)
- Streamlit 1.22.0
- scikit-learn 1.3.0
- OpenCV (Headless)
- Matplotlib 3.7.1

## 📋 Configuration requise

- Python 3.11 ou supérieur
- 4 Go de RAM minimum
- Processeur compatible avec les instructions AVX2 (recommandé)
- Espace disque : environ 500 Mo

## ⚠️ Notes importantes

- L'application utilise TensorFlow en mode CPU pour une meilleure compatibilité
- Lors du premier lancement, l'entraînement du modèle peut prendre plusieurs minutes
- Les performances de prédiction peuvent varier selon votre matériel

## 📝 License

MIT License - Libre d'utilisation et de modification
