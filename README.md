# Classification de Chiffres Manuscrits avec CNN Interactive

Application web interactive de classification de chiffres manuscrits utilisant un réseau de neurones convolutif (CNN).

## 🌟 Démo en ligne

Vous pouvez tester l'application en direct sur Railway : [Lien à venir]

## 🚀 Installation locale

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

4. **Configurer les variables d'environnement**
```bash
cp .env.example .env
# Modifier les valeurs dans .env selon vos besoins
```

5. **Lancer l'application**
```bash
streamlit run app.py
```

## 🚂 Déploiement sur Railway

1. Créez un compte sur [Railway.app](https://railway.app/)
2. Connectez votre compte GitHub
3. Créez un nouveau projet depuis le dépôt GitHub
4. Railway détectera automatiquement la configuration et déploiera l'application

## 🛠️ Technologies utilisées

- Python 3.11
- TensorFlow 2.15
- Streamlit 1.22
- scikit-learn 1.3.0
- OpenCV

## 📊 Fonctionnalités

- ✏️ Interface de dessin interactive
- 🧠 Classification en temps réel
- 📈 Visualisation des activations du réseau
- 📊 Analyse des probabilités de prédiction

## 📝 License

MIT License
