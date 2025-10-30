# =====================================================================
# 2. CNN MODEL ARCHITECTURE - Face Shape Classification
# =====================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class FaceShapeCNN:
    """
    CNN architecture specifically designed for face shape classification

    Architecture Features:
    1. Convolutional layers for feature extraction
    2. Batch normalization for training stability
    3. Dropout for regularization
    4. Data augmentation for better generalization
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build CNN architecture for face shape classification

        Architecture:
        - Data Augmentation Layer
        - 4 Convolutional Blocks (Conv2D + BatchNorm + MaxPool)
        - Global Average Pooling
        - Dense Classification Head
        """

        model = models.Sequential([
            # Data Augmentation (helps with generalization)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),

            # Block 1: Initial feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 2: Deeper features
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 3: Complex features
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 4: High-level features
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Global pooling instead of flattening (reduces overfitting)
            layers.GlobalAveragePooling2D(),

            # Classification head
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )

        self.model = model

        # Print model summary
        print("CNN Architecture:")
        self.model.summary()

        return self.model

    def create_advanced_model(self):
        """
        Advanced CNN with residual connections and attention
        Use this if basic model doesn't achieve target accuracy
        """

        # Input layer
        inputs = keras.Input(shape=self.input_shape)

        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)

        # Convolutional base
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Residual block 1
        shortcut = x
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Residual block 2
        shortcut = layers.Conv2D(128, 1, padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """
        Train the CNN model with proper callbacks and monitoring
        """

        if self.model is None:
            self.build_model()

        # Prepare validation data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        # Define callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.get_model_save_path(), 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train the model
        print(f"Training CNN on {len(X_train)} images...")
        print(f"Validation set: {len(X_val)} images")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )

        print("Training completed!")
        return self.history

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""

        if self.model is None:
            print("No model to evaluate. Train first!")
            return None

        # Evaluate
        test_loss, test_accuracy, test_top2 = self.model.evaluate(X_test, y_test, verbose=0)

        # Predictions for detailed analysis
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Print results
        print(f"   Test Results:")
        print(f"   Accuracy: {test_accuracy:.3f}")
        print(f"   Top-2 Accuracy: {test_top2:.3f}")
        print(f"   Loss: {test_loss:.3f}")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred_classes)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['round', 'oval', 'square', 'heart', 'oblong'],
                   yticklabels=['round', 'oval', 'square', 'heart', 'oblong'])
        plt.title('Face Shape Classification - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Classification report
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_classes,
                                  target_names=['round', 'oval', 'square', 'heart', 'oblong']))

        return {
            'accuracy': test_accuracy,
            'top2_accuracy': test_top2,
            'loss': test_loss,
            'predictions': y_pred_classes,
            'probabilities': y_pred
        }

    def plot_training_history(self):
        """Plot training history"""

        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def get_model_save_path(self):
        """Get path to save model"""
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_model(self, filename="face_shape_cnn_final.h5"):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return

        save_path = os.path.join(self.get_model_save_path(), filename)
        self.model.save(save_path)
        print(f"Model saved: {save_path}")
        return save_path