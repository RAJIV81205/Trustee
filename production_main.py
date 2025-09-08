from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import UserPreferences, MatchingResponse, InternshipMatch
from internship_matcher import InternshipMatchingModel
import uvicorn
import logging
import json
from pydantic import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Internship Matching API",
    description="Production ML-powered internship matching system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML model (loads automatically if trained)
model = InternshipMatchingModel()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        if not model.load_model():
            logger.info("No trained model found. Training new model...")
            model.train_model()
        else:
            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")

@app.get("/")
async def root():
    return {
        "message": "AI Internship Matching API",
        "version": "1.0.0",
        "status": "running",
        "model_trained": model.is_trained
    }

@app.post("/match-internships")
async def match_internships(user_preferences: UserPreferences):
    """
    Find all matching internships based on user preferences using ML model
    
    Required fields:
    - skills: List of strings (e.g., ["Python", "Machine Learning"])
    - experience: Integer (years of experience)
    - mode: String ("Hybrid", "On-site", or "Remote")
    - location: String (state name, e.g., "Karnataka")
    - sector: String ("tech" or "non-tech")
    - stipend: Integer (expected stipend amount)
    - duration: Integer (duration in months)
    """
    try:
        # Convert Pydantic model to dict
        preferences_dict = {
            'skills': user_preferences.skills,
            'experience': user_preferences.experience,
            'mode': user_preferences.mode.value,
            'location': user_preferences.location,
            'sector': user_preferences.sector.value,
            'stipend': user_preferences.stipend,
            'duration': user_preferences.duration
        }
        
        # Get all matches using ML model
        matches = model.predict_matches(preferences_dict)
        
        # Convert to response model
        internship_matches = [
            InternshipMatch(**match) for match in matches
        ]
        
        response = MatchingResponse(
            matches=internship_matches,
            total_matches=len(internship_matches)
        )
        
        return JSONResponse(
            status_code=200,
            content=response.dict()
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "message": "Please check your request format",
                "details": str(e),
                "example": {
                    "skills": ["Python", "Machine Learning"],
                    "experience": 1,
                    "mode": "Hybrid",
                    "location": "Karnataka",
                    "sector": "tech",
                    "stipend": 25000,
                    "duration": 6
                }
            }
        )
    except Exception as e:
        logger.error(f"Error finding matches: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": f"Error finding matches: {str(e)}"
            }
        )

@app.get("/internships")
async def get_all_internships():
    """
    Get all available internships
    """
    try:
        internships = model.df.to_dict('records')
        return {
            "success": True,
            "data": {
                "internships": internships,
                "total": len(internships)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching internships: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching internships: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_trained": model.is_trained,
        "total_internships": len(model.df)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)