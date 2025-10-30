# app/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict

class ImageUploadResponse(BaseModel):
    success: bool
    message: str
    image_id: str
    image_path: str

class LandmarksResponse(BaseModel):
    success: bool
    landmarks: Optional[Dict]
    glasses_transform: Optional[Dict]
    visualization_path: Optional[str]

class FaceShapeResponse(BaseModel):
    success: bool
    face_shape: str
    confidence: float
    probabilities: Dict[str, float]

class RecommendationResponse(BaseModel):
    success: bool
    face_shape: str
    recommended_glasses: List[Dict]

class OverlayResponse(BaseModel):
    success: bool
    overlaid_image_path: str
    overlay_params: Dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str

class CompleteTryOnResponse(BaseModel):
    success: bool
    face_shape: str
    confidence: float
    recommendations: List[Dict]
    overlays: List[Dict]