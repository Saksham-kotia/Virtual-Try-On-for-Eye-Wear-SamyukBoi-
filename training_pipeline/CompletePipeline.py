# =====================================================================
# 4. COMPLETE CNN PIPELINE - Putting It All Together
# =====================================================================

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from training_pipeline.DataLoadingPreprocesing import CNNDatasetManager
from training_pipeline.FaceShapeClassifier import FaceShapeCNN
from core_logic.landmark_extractor import LandmarkExtractor

class CompleteCNNPipeline:
    """
    Complete pipeline: Dataset → CNN Training → Landmark Integration → Ready for Overlay
    """

    def __init__(self, project_path):
        self.project_path = project_path
        self.dataset_manager = CNNDatasetManager(project_path)
        self.cnn_model = FaceShapeCNN()
        self.landmark_extractor = LandmarkExtractor()

    def run_complete_pipeline(self, zip_path, sample_size=2000):
        """
        Run the complete CNN-based pipeline
        """
        print("CNN-Based Face Shape Classification Pipeline")
        print("=" * 60)

        # Step 1: Dataset preparation
        print("\nStep 1: Dataset Extraction & Preprocessing")
        X, y = self.dataset_manager.extract_and_preprocess_batch(
            zip_path,
            sample_size=sample_size,
            batch_size=500
        )

        # Step 2: Train/test split
        print("\nStep 2: Train/Test Split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")

        # Step 3: CNN Training
        print("\nStep 3: CNN Model Training")
        self.cnn_model.build_model()
        history = self.cnn_model.train(X_train, y_train, epochs=30)

        # Step 4: Model evaluation
        print("\nStep 4: Model Evaluation")
        results = self.cnn_model.evaluate_model(X_test, y_test)

        # Step 5: Save model
        print("\nStep 5: Save Trained Model")
        model_path = self.cnn_model.save_model()

        # Step 6: Test landmark integration
        print("\nStep 6: Test Landmark Integration")
        self.test_landmark_integration(X_test[:5])

        print("\nComplete CNN Pipeline Finished!")
        print(f"Final Accuracy: {results['accuracy']:.3f}")
        print(f"Model saved at: {model_path}")

        return self.cnn_model, results

    def test_landmark_integration(self, test_images):
        """Test landmark extraction on sample images"""

        print("Testing landmark extraction for overlay engine...")

        for i, img in enumerate(test_images):
            # Convert back to uint8 for landmark extraction
            img_uint8 = (img * 255).astype(np.uint8)

            # Extract landmarks
            landmarks = self.landmark_extractor.extract_glasses_landmarks(img_uint8)

            if landmarks:
                # Calculate glasses transformation
                transform = self.landmark_extractor.calculate_glasses_transform(landmarks)

                print(f"Image {i+1}: Landmarks extracted successfully")
                if transform:
                    print(f"   Glasses center: {transform['center']}")
                    print(f"   Glasses width: {transform['width']}")
            else:
                print(f"Image {i+1}: No landmarks detected")