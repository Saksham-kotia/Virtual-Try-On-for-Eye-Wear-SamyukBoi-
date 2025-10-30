# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Import from YOUR structure
from app.services import (
    LandmarkService,
    FaceDetectionService, 
    ClassificationService,
    OverlayService,
    RecommendationService
)
from app.schemas import (
    ImageUploadResponse,
    LandmarksResponse,
    FaceShapeResponse,
    RecommendationResponse,
    OverlayResponse,
    HealthResponse
)

# Initialize FastAPI
app = FastAPI(
    title="Virtual Try-On API",
    description="AI-powered glasses virtual try-on",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/accessories", StaticFiles(directory="accessories"), name="accessories")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Initialize services on startup
landmark_service = None
face_detection_service = None
classification_service = None
overlay_service = None
recommendation_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize all services"""
    global landmark_service, face_detection_service, classification_service
    global overlay_service, recommendation_service
    
    print("Initializing services...")
    
    landmark_service = LandmarkService()
    face_detection_service = FaceDetectionService()
    classification_service = ClassificationService(
        model_path="models/best_model.h5",  # YOUR model
        encoder_path="models/label_encoder.pkl"  # YOUR encoder
    )
    overlay_service = OverlayService()
    recommendation_service = RecommendationService()
    
    print("All services initialized!")

# Health check
@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "healthy",
        "models_loaded": classification_service is not None,
        "version": "1.0.0"
    }

# Upload endpoint
@app.post("/api/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload user image"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        import uuid
        image_id = str(uuid.uuid4())
        
        os.makedirs("outputs/user_uploads", exist_ok=True)
        file_extension = file.filename.split('.')[-1]
        file_path = f"outputs/user_uploads/{image_id}.{file_extension}"
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        return {
            "success": True,
            "message": "Image uploaded successfully",
            "image_id": image_id,
            "image_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Classify face shape
@app.post("/api/classify-face-shape", response_model=FaceShapeResponse)
async def classify_face_shape(image_id: str):
    """Classify face shape using CNN"""
    try:
        # Find image
        image_path = None
        for ext in ['jpg', 'jpeg', 'png']:
            path = f"outputs/user_uploads/{image_id}.{ext}"
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Classify
        result = classification_service.predict(image_path)
        
        return {
            "success": True,
            "face_shape": result['face_shape'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Glass recommendation
@app.post("/api/recommend-glasses", response_model=RecommendationResponse)
async def recommend_glasses(face_shape: str):
    try:
        recs = recommendation_service.get_recommendations(face_shape)
        return {
            "success": True,
            "face_shape": face_shape,
            "recommended_glasses": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Overlay Glass
@app.post("/api/overlay", response_model=OverlayResponse)
async def overlay_glasses(image_id: str, glasses_id: str):
    try:
        # Locate user image
        for ext in ['jpg', 'jpeg', 'png']:
            path = f"outputs/user_uploads/{image_id}.{ext}"
            if os.path.exists(path):
                user_image_path = path
                break
        else:
            raise HTTPException(status_code=404, detail="User image not found")

        # Read user image
        user_image = cv2.imread(user_image_path)
        if user_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Extract landmarks and transformation
        landmarks = landmark_service.extract_glasses_landmarks(user_image)
        transform = landmark_service.calculate_glasses_transform(landmarks)
        if not transform:
            raise HTTPException(status_code=400, detail="Unable to detect landmarks")

        # Get glasses path from metadata
        metadata = recommendation_service.load_metadata()
        if glasses_id not in metadata:
            raise HTTPException(status_code=404, detail="Glasses not found")
        glasses_path = metadata[glasses_id]["path"]

        # Perform overlay
        result = overlay_service.overlay_glasses(user_image, glasses_path, transform)

        # Save result
        overlaid_path = f"outputs/demo_images/{image_id}_overlay.png"
        cv2.imwrite(overlaid_path, result)

        return {
            "success": True,
            "overlaid_image_path": overlaid_path,
            "overlay_params": transform
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Add other endpoints from the artifact...
# /api/landmarks
# /api/detect-face
# /api/recommend-glasses
# /api/overlay
# /api/complete-tryon

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)