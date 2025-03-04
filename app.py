import streamlit as st
import numpy as np
from model import create_model, train_model
from utils import preprocess_image, convert_streamlit_sketch_to_image, get_prediction_confidence
from visualize import plot_activation_maps, plot_prediction_confidence, plot_training_history
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os

# Configuration de la page
st.set_page_config(
    page_title="Classification de Chiffres CNN",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger le modèle (avec mise en cache)
@st.cache_resource
def load_model():
    with st.spinner('🔄 Chargement du modèle...'):
        model, history = train_model()
    return model, history

# Chargement du modèle
model, history = load_model()

# Titre principal
st.title("🧠 Classification de Chiffres Manuscrits 🔢✏️")
st.markdown("Un modèle de Deep Learning qui reconnaît vos chiffres manuscrits en temps réel")

# Description du projet
st.header("À propos de cette application")
st.write("""
Cette application utilise un réseau de neurones convolutif (CNN) pour classifier les chiffres manuscrits. 
Le modèle a été entraîné sur le dataset MNIST et peut reconnaître les chiffres de 0 à 9 avec une haute précision.
""")

# Statistiques
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Précision sur MNIST", value="97.5%")
with col2:
    st.metric(label="Paramètres du modèle", value="1.2M")
with col3:
    st.metric(label="Images d'entraînement", value="60K")

# Fonctionnalités
st.subheader("Fonctionnalités")
st.markdown("""
- ✏️ Interface de dessin interactive pour tester le modèle en temps réel
- 🧠 Visualisation des couches d'activation du réseau neuronal
- 📊 Analyse détaillée des probabilités pour chaque chiffre
- 📈 Suivi des performances d'entraînement du modèle
""")

# Tags
st.markdown("**Technologies utilisées:**")
st.markdown("TensorFlow 2.16 • Python 3.9+ • Deep Learning • CNN • MNIST • Computer Vision")

# Création de trois colonnes principales
col_draw, col_pred, col_viz = st.columns([1, 1, 2])

# Zone de dessin
with col_draw:
    st.header("Zone de Dessin")
    st.write("Dessinez un chiffre entre 0 et 9 dans la zone ci-dessous :")
    
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color='#000000',
        background_color='#FFFFFF',
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_btn = st.button("🔍 Analyser", key="analyze_btn")
    with col_btn2:
        clear_btn = st.button("🗑️ Effacer", key="clear_btn")

    if clear_btn:
        st.session_state.canvas_key = st.session_state.get('canvas_key', 0) + 1
        st.rerun()

# Zone des résultats
with col_pred:
    st.header("Résultats")
    st.write("La prédiction et les probabilités apparaîtront ici :")
    
    if canvas_result.image_data is not None and analyze_btn:
        image = convert_streamlit_sketch_to_image(canvas_result.image_data)
        if image is not None:
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                try:
                    with st.spinner('🔄 Analyse en cours...'):
                        predictions = model.predict(processed_image)
                        pred_class, confidence = get_prediction_confidence(predictions)
                        
                        st.success(f"Prédiction : {pred_class}")
                        st.progress(confidence / 100)
                        st.write(f"Confiance : {confidence:.2f}%")
                        
                        st.subheader("Distribution des probabilités")
                        fig = plot_prediction_confidence(predictions)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"⚠️ Erreur lors de la prédiction : {str(e)}")
            else:
                st.error("⚠️ Erreur lors du prétraitement de l'image")
        else:
            st.error("⚠️ Erreur lors de la conversion de l'image")
    else:
        st.info("Dessinez un chiffre et cliquez sur 'Analyser' pour obtenir une prédiction")

# Zone de visualisation
with col_viz:
    st.header("Visualisation")
    if canvas_result.image_data is not None and analyze_btn and processed_image is not None:
        st.subheader("Visualisation des Activations")
        st.write("""
        Observez comment le réseau neuronal traite votre dessin à travers ses différentes couches. 
        Chaque carte montre les caractéristiques détectées par une couche de convolution.
        """)
        fig = plot_activation_maps(processed_image, model)
        st.pyplot(fig)
        
        st.subheader("Métriques d'Entraînement")
        st.write("Évolution de la précision et de la perte pendant l'entraînement sur MNIST.")
        fig = plot_training_history(history)
        st.pyplot(fig)
    else:
        st.info("Les visualisations apparaîtront ici après l'analyse d'un chiffre.")

# Barre latérale avec informations sur le modèle
with st.sidebar:
    st.header("Information sur le Modèle")
    
    st.success("✅ Modèle chargé et prêt")
    
    st.subheader("Architecture du CNN")
    st.table({
        "Type": ["CNN Séquentiel"],
        "Couches totales": ["8"],
        "Paramètres": ["1.2M"],
        "Format d'entrée": ["28x28 px"]
    })
    
    st.subheader("Structure du modèle")
    st.markdown("""
    - 3 blocs de convolution avec :
        - Conv2D
        - BatchNormalization
        - MaxPooling2D
        - Dropout (0.25)
    - Dense (128) avec Dropout (0.5)
    - Sortie Dense (10) avec Softmax
    """)
    
    st.subheader("Entraînement")
    st.write("""
    Le modèle a été entraîné sur MNIST avec :
    - 60 000 images d'entraînement
    - Optimiseur Adam
    - Learning rate adaptatif
    - Early stopping
    """)
    
    st.subheader("Conseils d'utilisation")
    st.info("""
    Pour de meilleurs résultats :
    - Centrez votre dessin
    - Utilisez des traits nets
    - Dessinez un seul chiffre
    - Évitez les dessins trop petits
    """)

    st.subheader("Prochainement:")
    st.write("""
    **Export et Partage**
    - 💾 Téléchargement des dessins au format PNG/JPEG
    - 📊 Export des visualisations et graphiques
    - 📱 Partage des résultats sur les réseaux sociaux
    - 📄 Export des métriques en PDF

    **Interface**
    - 🌓 Mode sombre/clair personnalisable
    - ✏️ Personnalisation du trait (épaisseur, couleur)
    - 🔄 Animations des transitions
    - 📱 Interface responsive pour mobile

    **Fonctionnalités Avancées**
    - 🔢 Reconnaissance de plusieurs chiffres
    - 🤖 Comparaison avec d'autres modèles (SVM, Random Forest)
    - 📚 Mode apprentissage avec suggestions
    - 📈 Historique des prédictions

    **Outils et Développement**
    - 🔍 Mode debug détaillé
    - 🔌 API REST pour intégration
    - 📝 Documentation interactive
    - ⚡ Optimisation des performances
    """)

# Footer
st.markdown("---")
st.markdown("Développé par Farx1 durant une nuit blanche en utilisant Streamlit et TensorFlow")
st.markdown("v1.0.0 • TensorFlow 2.16 • Python 3.9+")