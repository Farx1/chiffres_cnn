import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_activation_maps(input_image, model):
    """Visualise les cartes d'activation des couches de convolution"""
    # Créer la figure avec une taille plus grande
    plt.figure(figsize=(20, 5))
    
    # Afficher l'image d'entrée
    plt.subplot(1, 4, 1)
    # S'assurer que l'image est dans le bon format et correctement normalisée
    display_image = np.squeeze(input_image)  # Enlever les dimensions supplémentaires
    plt.imshow(display_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Image d\'entrée')
    plt.colorbar()  # Ajouter une barre de couleur
    plt.axis('on')  # Afficher les axes
    
    # Obtenir les couches de convolution
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
            if len(conv_layers) == 3:  # On ne prend que les 3 premières couches conv
                break
    
    # Créer des modèles intermédiaires pour chaque couche conv
    for idx, conv_layer in enumerate(conv_layers):
        # Créer un modèle qui va jusqu'à cette couche
        temp_model = tf.keras.Model(inputs=model.input, outputs=conv_layer.output)
        
        # Obtenir l'activation
        activation = temp_model.predict(input_image, verbose=0)
        
        # Afficher la carte d'activation moyenne
        plt.subplot(1, 4, idx + 2)
        feature_map = np.mean(activation[0], axis=-1)
        plt.imshow(feature_map, cmap='viridis')
        plt.title(f'Conv2D {idx + 1}')
        plt.colorbar()  # Ajouter une barre de couleur
        plt.axis('on')  # Afficher les axes
    
    plt.tight_layout(pad=3.0)  # Augmenter l'espace entre les sous-plots
    return plt

def plot_prediction_confidence(predictions):
    """Visualise la confiance des prédictions pour chaque chiffre"""
    # Normaliser les prédictions pour s'assurer que la somme est 1
    predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(10), predictions[0] * 100)  # Convertir en pourcentage
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.title('Confiance de prédiction par chiffre')
    plt.xlabel('Chiffre')
    plt.ylabel('Confiance (%)')
    plt.xticks(range(10))
    plt.ylim(0, 100)  # Limiter l'axe y entre 0 et 100%
    plt.grid(True, alpha=0.3)  # Ajouter une grille légère
    return plt

def plot_training_history(history):
    """Visualise l'historique d'entraînement"""
    plt.figure(figsize=(15, 5))
    
    # Obtenir l'historique (gérer à la fois l'objet History et le dictionnaire chargé)
    hist = history.history if hasattr(history, 'history') else history
    
    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(hist['accuracy'], label='Entraînement', linewidth=2)
    plt.plot(hist['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Précision du modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Entraînement', linewidth=2)
    plt.plot(hist['val_loss'], label='Validation', linewidth=2)
    plt.title('Perte du modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt 