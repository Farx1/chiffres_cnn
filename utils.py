import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile

def save_and_read_image(image_data):
    """Sauvegarde l'image dans un fichier temporaire et la relit"""
    try:
        # Créer un dossier temporaire s'il n'existe pas
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Sauvegarder l'image temporairement
        temp_path = os.path.join(temp_dir, "temp_digit.png")
        cv2.imwrite(temp_path, image_data)
        
        # Relire l'image
        image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        
        # Supprimer le fichier temporaire
        os.remove(temp_path)
        
        return image
    except Exception as e:
        print(f"Erreur lors de la sauvegarde/lecture de l'image : {e}")
        return None

def preprocess_image(image_data):
    """Prétraite l'image dessinée pour la prédiction"""
    try:
        print("Début prétraitement - forme:", image_data.shape)
        
        # Binarisation avec seuil adaptatif (le dessin est noir sur blanc)
        _, image_data = cv2.threshold(image_data, 127, 255, cv2.THRESH_BINARY_INV)
        print("Après binarisation - valeurs uniques:", np.unique(image_data))
        
        # Trouver les contours sur l'image inversée
        contours, _ = cv2.findContours(image_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            print(f"Nombre de contours trouvés: {len(contours)}")
            # Trouver le plus grand contour
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Ajouter une marge autour du chiffre
            margin = 2
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image_data.shape[1] - x, w + 2 * margin)
            h = min(image_data.shape[0] - y, h + 2 * margin)
            
            # Extraire et centrer le chiffre
            digit = image_data[y:y+h, x:x+w]
            
            # Calculer le côté le plus long pour faire un carré
            max_size = max(w, h)
            
            # Créer une image carrée avec le chiffre centré
            square = np.zeros((max_size, max_size), dtype=np.uint8)
            x_offset = (max_size - w) // 2
            y_offset = (max_size - h) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = digit
            
            # Redimensionner à 20x20
            digit = cv2.resize(square, (20, 20))
            
            # Créer l'image finale 28x28 avec une bordure
            final_image = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - 20) // 2
            y_offset = (28 - 20) // 2
            final_image[y_offset:y_offset+20, x_offset:x_offset+20] = digit
            
            print("Dimensions finales:", final_image.shape)
        else:
            print("Aucun contour trouvé - redimensionnement simple")
            final_image = cv2.resize(image_data, (28, 28))
        
        # Normaliser entre 0 et 1 comme dans MNIST
        final_image = final_image.astype('float32') / 255.0
        print("Valeurs après normalisation:", np.unique(final_image))
        
        # Format pour le modèle
        final_image = np.expand_dims(final_image, axis=-1)
        final_image = np.expand_dims(final_image, axis=0)
        
        return final_image
    
    except Exception as e:
        print(f"Erreur de prétraitement: {e}")
        return None

def convert_streamlit_sketch_to_image(sketch_data):
    """Convertit le dessin Streamlit en image numpy"""
    try:
        if sketch_data is not None:
            print("Forme de l'image d'entrée:", sketch_data.shape)
            
            # Pour une image RGBA (4 canaux)
            if sketch_data.shape[2] == 4:
                print("Traitement RGBA - Valeurs RGB:", [np.unique(sketch_data[:,:,i]) for i in range(3)])
                # Convertir en niveaux de gris en utilisant les canaux RGB
                image = cv2.cvtColor(sketch_data[:,:,:3], cv2.COLOR_RGB2GRAY)
            # Pour une image RGB (3 canaux)
            elif sketch_data.shape[2] == 3:
                print("Traitement RGB - Valeurs:", np.unique(sketch_data))
                # Convertir en niveaux de gris
                image = cv2.cvtColor(sketch_data, cv2.COLOR_RGB2GRAY)
            else:
                print(f"Format d'image non supporté: {sketch_data.shape[2]} canaux")
                return None
            
            print("Valeurs après conversion:", np.unique(image))
            return image.astype(np.uint8)
            
        return None
    except Exception as e:
        print(f"Erreur de conversion: {e}")
        return None

def get_prediction_confidence(predictions):
    """Calcule la confiance de la prédiction"""
    try:
        # Normaliser les prédictions
        predictions = predictions / np.sum(predictions)
        
        # Obtenir la classe prédite et sa confiance
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        
        return pred_class, confidence * 100.0
    except Exception as e:
        print(f"Erreur lors du calcul de la confiance : {e}")
        return None, 0.0 