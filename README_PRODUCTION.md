# Production Internship Matching API

A production-ready ML-powered internship matching system that trains once and serves predictions efficiently.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (One Time Only)

```bash
python train_model.py
```

### 3. Start Production Server

```bash
python production_main.py
```

### 4. Test the API

```bash
python simple_test.py
```

## 📡 API Endpoints

### POST /match-internships

Get all internships ranked by ML-predicted matching scores.

**Request:**

```json
{
  "skills": ["Python", "Machine Learning"],
  "experience": 1,
  "mode": "Hybrid",
  "location": "Karnataka",
  "sector": "tech",
  "stipend": 25000,
  "duration": 6
}
```

**Response:**

```json
{
  "matches": [
    {
      "name": "Data Science Intern",
      "location": "Karnataka",
      "stipend": 30000,
      "requirements": "Python,SQL,Machine Learning",
      "mode": "Hybrid",
      "sector": "tech",
      "duration": 6,
      "matching_score": 9.22
    }
  ],
  "total_matches": 20
}
```

### GET /internships

Get all available internships.

**Response:**

```json
{
  "success": true,
  "data": {
    "internships": [...],
    "total": 20
  }
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_trained": true,
  "total_internships": 20
}
```

## 🏗️ Production Features

- **Auto Model Loading**: Loads trained model on startup
- **Error Handling**: Proper HTTP error responses
- **Logging**: Production logging for monitoring
- **CORS Support**: Ready for frontend integration
- **Health Checks**: Monitor API status
- **Clean JSON**: Proper JSON structure for all responses

## 📁 Production Files

```
├── production_main.py      # Production API server
├── internship_matcher.py   # ML model (production optimized)
├── models.py              # API models
├── train_model.py         # One-time training script
├── simple_test.py         # Production testing
├── sample_internships.csv # Data (state-based)
├── requirements.txt       # Dependencies
└── [Model Files - Auto Generated]
    ├── internship_model.joblib
    ├── tfidf_vectorizer.joblib
    └── scaler.joblib
```

## 🔧 Model Training

The model trains **once** and saves to disk:

- **Random Forest Regressor** for score prediction
- **TF-IDF Vectorization** for skill matching
- **Feature Engineering** with 7 key features
- **Automatic Loading** on API startup

## 🌐 Integration Example

```python
import requests

# API endpoint
url = "http://localhost:8000/match-internships"

# User preferences
data = {
    "skills": ["JavaScript", "React"],
    "experience": 0,
    "mode": "Remote",
    "location": "Maharashtra",
    "sector": "tech",
    "stipend": 20000,
    "duration": 3
}

# Get matches
response = requests.post(url, json=data)
matches = response.json()

# Process results
for match in matches['matches']:
    print(f"{match['name']}: {match['matching_score']}/10")
```

## 🚀 Deployment Ready

- Clean error handling
- Proper HTTP status codes
- Structured JSON responses
- Production logging
- Health monitoring
- CORS configured
- Auto model initialization
