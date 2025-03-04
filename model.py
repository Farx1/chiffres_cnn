import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os
import pickle

MODEL_PATH = 'model/mnist_model.keras'
HISTORY_PATH = 'model/training_history.pkl'

def create_model():
    # Utiliser l'API fonctionnelle
    inputs = Input(shape=(28, 28, 1))
    
    # Augmentation des données
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    
    # Appliquer l'augmentation des données
    x = data_augmentation(inputs)
    
    # Première couche de convolution
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Deuxième couche de convolution
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Troisième couche de convolution
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Couches denses
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Créer le modèle
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def preprocess_data():
    # Charger les données MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normaliser les images
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Ajouter une dimension pour les canaux
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return (x_train, y_train), (x_test, y_test)

def train_model():
    # Vérifier si le modèle et l'historique existent déjà
    if os.path.exists(MODEL_PATH) and os.path.exists(HISTORY_PATH):
        try:
            # Charger le modèle existant
            model = tf.keras.models.load_model(MODEL_PATH)
            # Charger l'historique
            with open(HISTORY_PATH, 'rb') as f:
                history = pickle.load(f)
            print("Modèle chargé depuis le fichier sauvegardé")
            return model, history
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            print("Entraînement d'un nouveau modèle...")

    # Si le modèle n'existe pas ou n'a pas pu être chargé, l'entraîner
    print("Entraînement d'un nouveau modèle...")
    
    # Charger et prétraiter les données
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    
    # Créer et compiler le modèle
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks pour l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Entraînement
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nPrécision sur le jeu de test : {test_acc:.4f}')
    
    # Créer le dossier model s'il n'existe pas
    os.makedirs('model', exist_ok=True)
    
    try:
        # Sauvegarder le modèle
        model.save(MODEL_PATH)
        
        # Sauvegarder l'historique
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)
        
        print("Modèle et historique sauvegardés")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
    
    return model, history

if __name__ == '__main__':
    model, history = train_model()