import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import zipfile
import json
from pathlib import Path
import time
from tqdm import tqdm
import seaborn as sns

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# =====================================================================
# 1. SMART DATASET MANAGER - CNN Optimized
# =====================================================================

class CNNDatasetManager:
    """
    CNN-optimized dataset manager that:
    1. Extracts images in batches
    2. Preprocesses for CNN input (224x224x3)
    3. Handles labels for classification
    4. Creates TensorFlow-compatible datasets
    """

    def __init__(self, project_path):
        self.project_path = project_path
        self.data_raw_path = os.path.join(project_path, "data", "raw")
        self.data_processed_path = os.path.join(project_path, "data", "processed")
        self.image_size = (224, 224)
        self.batch_size = 32

        # Face shape mapping for CNN
        self.face_shapes = ['round', 'oval', 'square', 'heart', 'oblong']
        self.num_classes = len(self.face_shapes)

        self.setup_directories()

    def setup_directories(self):
        """Create CNN-specific directory structure"""
        os.makedirs(self.data_processed_path, exist_ok=True)

        # Create organized folders for each face shape
        for shape in self.face_shapes:
            shape_dir = os.path.join(self.data_processed_path, "organized", shape)
            os.makedirs(shape_dir, exist_ok=True)

        # Create train/test split directories
        for split in ['train', 'test']:
            for shape in self.face_shapes:
                split_dir = os.path.join(self.data_processed_path, split, shape)
                os.makedirs(split_dir, exist_ok=True)

    def extract_and_preprocess_batch(self, zip_path, sample_size=2000, batch_size=500):
        """
        Extract images in batches and preprocess for CNN training
        This solves large dataset problem!

        """
        print(f"CNN Dataset Extraction: {sample_size} images in batches of {batch_size}")

        all_images = []
        all_labels = []

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            # Filter for image files
            image_files = [file for file in file_list if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} total images in dataset")

            # Process in batches to avoid memory issues
            for batch_start in range(0, min(sample_size, len(image_files)), batch_size):
                batch_end = min(batch_start + batch_size, min(sample_size, len(image_files)))
                batch_files = image_files[batch_start:batch_end]

                print(f"\nProcessing batch {batch_start//batch_size + 1}: images {batch_start}-{batch_end}")

                batch_images = []
                batch_labels = []

                for file_name in tqdm(batch_files, desc="Processing"):
                    try:
                        # Extract and read image
                        zip_ref.extract(file_name, "/tmp")
                        img_path = os.path.join("/tmp", file_name)

                        # Preprocess for CNN
                        processed_img = self.preprocess_for_cnn(img_path)

                        if processed_img is not None:
                            batch_images.append(processed_img)

                            # Generate label
                            label = self.generate_label_for_image(file_name, img_path)
                            batch_labels.append(label)

                        # Clean up temp file
                        os.remove(img_path)

                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")

                # Add batch to main arrays
                all_images.extend(batch_images)
                all_labels.extend(batch_labels)

                print(f"Batch complete: {len(batch_images)} images processed")

        # Convert to numpy arrays
        X = np.array(all_images)
        y = np.array(all_labels)

        print(f"Final dataset: {X.shape} images, {len(y)} labels")

        # Save processed dataset
        self.save_processed_dataset(X, y)

        return X, y

    def preprocess_for_cnn(self, img_path):
        """
        CNN-specific preprocessing pipeline:
        1. Load image
        2. Resize to 224x224 (standard CNN input)
        3. Normalize to [0,1]
        4. Handle RGB conversion

        """
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to CNN input size (224x224)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)

            # Optional: Light denoising
            img = cv2.bilateralFilter(img, 9, 75, 75)

            # Normalize to [0, 1] for CNN training
            img = img.astype(np.float32) / 255.0

            return img

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def generate_label_for_image(self, filename, img_path):
        """
        Generate face shape labels for CNN training.

        For now, this is a placeholder - you'll need to implement:
        1. Manual annotation
        2. Existing label files
        3. Semi-automatic labeling

        """

        # PLACEHOLDER: Random labeling for development
        # Replace with actual labeling strategy

        # For development - random assignment
        return np.random.choice(self.face_shapes)

    def save_processed_dataset(self, X, y):
        """Save processed dataset for CNN training"""

        # Convert labels to categorical
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Save as compressed numpy arrays
        dataset_path = os.path.join(self.data_processed_path, "cnn_dataset.npz")

        np.savez_compressed(
            dataset_path,
            images=X,
            labels=y_encoded,
            label_names=y,
            face_shapes=self.face_shapes
        )

        print(f"CNN dataset saved: {dataset_path}")

        # Save label encoder
        import joblib
        encoder_path = os.path.join(self.data_processed_path, "label_encoder.pkl")
        joblib.dump(label_encoder, encoder_path)

        return dataset_path

