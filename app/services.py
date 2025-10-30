# app/services.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os

# =====================================================================
# LANDMARK SERVICE
# =====================================================================
class LandmarkService:
    """Facial landmark extraction using MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.key_landmarks = {
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'nose_bridge': 6,
            'nose_tip': 1,
            'left_temple': 234,
            'right_temple': 454
        }
    
    def extract_glasses_landmarks(self, image):
        """Extract landmarks for glasses positioning"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            glasses_points = {}
            for name, idx in self.key_landmarks.items():
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    glasses_points[name] = (x, y)
            
            return glasses_points
        
        return None
    
    def calculate_glasses_transform(self, landmarks):
        """Calculate transformation for glasses overlay"""
        if not landmarks:
            return None
        
        left_eye = landmarks.get('left_eye_outer')
        right_eye = landmarks.get('right_eye_outer')
        
        if left_eye and right_eye:
            glasses_width = abs(right_eye[0] - left_eye[0])
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            
            angle = np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            )
            
            return {
                'center': (center_x, center_y),
                'width': glasses_width,
                'angle': float(angle),
                'scale': glasses_width / 150
            }
        
        return None


# =====================================================================
# FACE DETECTION SERVICE
# =====================================================================
class FaceDetectionService:
    """Face detection using MediaPipe"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
    
    def detect_face(self, image):
        """Detect face and return bounding box"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(rgb_image)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = image.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            return {'x': x, 'y': y, 'width': width, 'height': height}
        
        return None


# =====================================================================
# CLASSIFICATION SERVICE - YOUR CNN MODEL
# =====================================================================
class ClassificationService:
    """Face shape classification using YOUR trained CNN"""
    
    def __init__(self, model_path="models/best_model.h5", 
                 encoder_path="models/label_encoder.pkl"):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.label_encoder = None
        self.image_size = (224, 224)
        
        self.load_model()
    
    def load_model(self):
        """Load YOUR trained model and encoder"""
        try:
            # Load model
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load label encoder
            self.label_encoder = joblib.load(self.encoder_path)
            print(f"✅ Label encoder loaded from {self.encoder_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image_path):
        """
        Preprocess image - SAME as your training pipeline
        From your DataLoadingPreprocessing.py
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Light denoising (bilateral filter)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return np.expand_dims(img, axis=0)
    
    def predict(self, image_path):
        """Predict face shape"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess
        processed_img = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(processed_img, verbose=0)[0]
        
        # Get face shape
        predicted_idx = np.argmax(predictions)
        face_shape = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(predictions[predicted_idx])
        
        # All probabilities
        probabilities = {
            self.label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(predictions)
        }
        
        return {
            'face_shape': face_shape,
            'confidence': confidence,
            'probabilities': probabilities
        }


# =====================================================================
# OVERLAY SERVICE
# =====================================================================
class OverlayService:
    """Overlay glasses on face"""
    
    def overlay_glasses(self, user_image, glasses_path, transform, 
                       scale=1.0, rotation=0.0):
        """Overlay glasses using landmarks"""
        glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        if glasses is None:
            raise ValueError("Could not load glasses image")
        
        center = transform['center']
        width = int(transform['width'] * scale)
        angle = transform['angle'] + rotation
        
        # Resize
        aspect_ratio = glasses.shape[0] / glasses.shape[1]
        new_height = int(width * aspect_ratio)
        glasses_resized = cv2.resize(glasses, (width, new_height))
        
        # Rotate
        angle_deg = np.degrees(angle)
        M = cv2.getRotationMatrix2D((width // 2, new_height // 2), angle_deg, 1.0)
        glasses_rotated = cv2.warpAffine(glasses_resized, M, (width, new_height))
        
        # Position
        x = center[0] - width // 2
        y = center[1] - new_height // 2
        
        # Overlay with alpha blending
        result = user_image.copy()
        
        for c in range(3):
            if glasses_rotated.shape[2] == 4:  # Has alpha
                alpha = glasses_rotated[:, :, 3] / 255.0
                
                y1, y2 = max(0, y), min(result.shape[0], y + new_height)
                x1, x2 = max(0, x), min(result.shape[1], x + width)
                
                gy1, gy2 = max(0, -y), min(new_height, result.shape[0] - y)
                gx1, gx2 = max(0, -x), min(width, result.shape[1] - x)
                
                if y2 > y1 and x2 > x1:
                    result[y1:y2, x1:x2, c] = (
                        alpha[gy1:gy2, gx1:gx2] * glasses_rotated[gy1:gy2, gx1:gx2, c] +
                        (1 - alpha[gy1:gy2, gx1:gx2]) * result[y1:y2, x1:x2, c]
                    )
        
        return result


# =====================================================================
# RECOMMENDATION SERVICE
# =====================================================================
class RecommendationService:
    """Glasses recommendation based on face shape"""
    
    def __init__(self, metadata_path="accessories/metadata.json"):
        self.metadata_path = metadata_path
        self.glasses_data = self.load_metadata()
    
    def load_metadata(self):
        """Load glasses metadata"""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_recommendations(self, face_shape, top_k=5):
        """Get top-k recommendations for face shape"""
        recommendations = []
        
        for glasses_id, glasses_info in self.glasses_data.items():
            suitable_shapes = glasses_info.get('suitable_for', [])
            
            if face_shape.lower() in [s.lower() for s in suitable_shapes]:
                score = glasses_info.get('score', 0.5)
                
                recommendations.append({
                    'id': glasses_id,
                    'name': glasses_info.get('name', glasses_id),
                    'category': glasses_info.get('category', 'classic'),
                    'score': score,
                    'path': glasses_info.get('path', f"accessories/glasses/{glasses_id}"),
                    'description': glasses_info.get('description', ''),
                    'price': glasses_info.get('price', 0)
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]