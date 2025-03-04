# Classification de Chiffres Manuscrits avec CNN Interactive

Application web interactive de classification de chiffres manuscrits utilisant un réseau de neurones convolutif (CNN).

## 🚀 Installation

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/chiffres_cnn.git
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

## 🎯 Utilisation

1. **Lancer l'application**
```bash
streamlit run app.py
```

2. **Premier lancement**
- Lors du premier lancement, le modèle sera automatiquement entraîné
- Cette étape peut prendre quelques minutes
- Le modèle sera ensuite sauvegardé pour les utilisations futures

## 📦 Déploiement sur Vercel

1. **Préparer le projet**
- Assurez-vous que tous les fichiers sont commités sur GitHub
- Le modèle sera entraîné automatiquement lors du premier déploiement

2. **Déployer sur Vercel**
- Connectez-vous sur [Vercel](https://vercel.com)
- Importez votre repository GitHub
- Configurez le projet :
  - Framework Preset : Other
  - Build Command : laissez vide
  - Output Directory : laissez vide
  - Install Command : `pip install -r requirements.txt`

## 🔧 Structure du Projet
```
chiffres_cnn/
├── app.py              # Application Streamlit principale
├── model.py           # Définition et entraînement du CNN
├── utils.py           # Fonctions utilitaires
├── visualize.py       # Fonctions de visualisation
├── requirements.txt   # Dépendances Python
└── vercel.json       # Configuration Vercel
```

## 📝 Notes
- Le modèle sera entraîné lors du premier lancement
- Les fichiers du modèle seront générés dans le dossier `model/`
- Temps d'entraînement estimé : 5-10 minutes selon votre machine

## 🔗 Intégration

Pour intégrer l'application dans votre portfolio :

```html
<!-- Option 1: Iframe -->
<iframe 
  src="VOTRE_URL_VERCEL" 
  width="100%" 
  height="800px" 
  frameborder="0"
></iframe>

<!-- Option 2: Lien direct -->
<a href="VOTRE_URL_VERCEL" target="_blank">
  Tester la Classification de Chiffres
</a>
``` 