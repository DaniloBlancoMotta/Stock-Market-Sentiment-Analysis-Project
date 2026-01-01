import logging
import os
import pickle
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "model.bin"
PORT = 9696
HOST = "0.0.0.0"

class SentimentRequest(BaseModel):
    """Schema for incoming sentiment analysis requests."""
    text: str = Field(..., min_length=1, description="Financial text to analyze")

class PredictionResponse(BaseModel):
    """Schema for sentiment prediction responses."""
    label: str = Field(..., description="Sentiment label (Positive or Negative)")
    confidence: float = Field(..., description="Prediction confidence score")
    text: str = Field(..., description="Original input text")

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Sentiment API",
    description="A high-performance API for classifying financial sentiment.",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
model: Any = None

@app.on_event("startup")
def load_model() -> None:
    """Loads the trained model pipeline on startup."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model successfully loaded from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": str(model is not None)}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: SentimentRequest) -> PredictionResponse:
    """Classifies the sentiment of the provided financial text.

    Args:
        request: The SentimentRequest containing the text.

    Returns:
        PredictionResponse with the label and confidence.

    Raises:
        HTTPException: If the model is not loaded or prediction fails.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        # Prediction
        prediction = int(model.predict([request.text])[0])
        # Mapping: 1 -> Positive, 0 -> Negative
        label = "Positive" if prediction == 1 else "Negative"
        
        logger.info(f"Prediction made: {label} for text: {request.text[:50]}...")
        
        return PredictionResponse(
            label=label,
            confidence=1.0,  # Placeholder for LinearSVC
            text=request.text
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
