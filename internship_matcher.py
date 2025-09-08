import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import joblib
import os
import re

class InternshipMatchingModel:
    def __init__(self, csv_path: str = "sample_internships.csv"):
        self.df = pd.read_csv(csv_path)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.model_path = "internship_model.joblib"
        self.vectorizer_path = "tfidf_vectorizer.joblib"
        self.encoders_path = "label_encoders.joblib"
        self.scaler_path = "scaler.joblib"
        
        # State mapping for better location matching
        self.state_mapping = {
            'maharashtra': ['mumbai', 'pune', 'nagpur', 'nashik'],
            'karnataka': ['bangalore', 'bengaluru', 'mysore', 'hubli'],
            'delhi': ['new delhi', 'delhi ncr'],
            'telangana': ['hyderabad', 'secunderabad'],
            'tamil nadu': ['chennai', 'coimbatore', 'madurai'],
            'gujarat': ['ahmedabad', 'surat', 'vadodara'],
            'rajasthan': ['jaipur', 'udaipur', 'jodhpur'],
            'haryana': ['gurgaon', 'faridabad', 'noida']
        }
        
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and clean the internship data"""
        # Clean and standardize the data
        self.df['requirements_clean'] = self.df['requirements'].str.lower()
        self.df['location_clean'] = self.df['location'].str.lower()
        self.df['mode_clean'] = self.df['mode'].str.lower()
        self.df['sector_clean'] = self.df['sector'].str.lower()
        
        # Fit TF-IDF vectorizer on requirements
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['requirements_clean'])
    
    def _calculate_skill_similarity(self, user_skills: List[str], internship_requirements: str) -> float:
        """Calculate skill matching score using TF-IDF and cosine similarity"""
        user_skills_text = ' '.join([skill.lower() for skill in user_skills])
        user_vector = self.tfidf_vectorizer.transform([user_skills_text])
        
        # Transform internship requirements
        internship_vector = self.tfidf_vectorizer.transform([internship_requirements.lower()])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(user_vector, internship_vector)[0][0]
        return similarity
    
    def _calculate_location_match(self, user_state: str, internship_state: str) -> float:
        """Calculate location matching score based on state names"""
        user_state_clean = user_state.lower().strip()
        internship_state_clean = internship_state.lower().strip()
        
        # Direct state match
        if user_state_clean == internship_state_clean:
            return 1.0
        
        # Check if user state matches any city in the internship state
        for state, cities in self.state_mapping.items():
            if state == internship_state_clean and user_state_clean in cities:
                return 1.0
            if state == user_state_clean and internship_state_clean in cities:
                return 1.0
        
        return 0.0
    
    def _create_training_data(self):
        """Create training data with features and synthetic scores"""
        features = []
        scores = []
        
        # Create synthetic user preferences for training
        sample_users = [
            {'skills': ['python', 'machine learning'], 'experience': 1, 'mode': 'hybrid', 'location': 'karnataka', 'sector': 'tech', 'stipend': 25000, 'duration': 6},
            {'skills': ['javascript', 'react'], 'experience': 0, 'mode': 'remote', 'location': 'maharashtra', 'sector': 'tech', 'stipend': 20000, 'duration': 3},
            {'skills': ['marketing', 'social media'], 'experience': 2, 'mode': 'on-site', 'location': 'delhi', 'sector': 'non-tech', 'stipend': 15000, 'duration': 2},
            {'skills': ['java', 'spring'], 'experience': 1, 'mode': 'on-site', 'location': 'telangana', 'sector': 'tech', 'stipend': 28000, 'duration': 6},
            {'skills': ['design', 'photoshop'], 'experience': 0, 'mode': 'remote', 'location': 'delhi', 'sector': 'non-tech', 'stipend': 16000, 'duration': 3}
        ]
        
        for user in sample_users:
            for idx, internship in self.df.iterrows():
                feature_vector = self._extract_features(user, internship)
                synthetic_score = self._calculate_synthetic_score(user, internship)
                features.append(feature_vector)
                scores.append(synthetic_score)
        
        return np.array(features), np.array(scores)
    
    def _extract_features(self, user_prefs: Dict, internship: pd.Series) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Skill similarity
        skill_sim = self._calculate_skill_similarity(user_prefs['skills'], internship['requirements'])
        features.append(skill_sim)
        
        # Experience match (normalized)
        exp_diff = abs(user_prefs['experience'] - 1) / 5.0  # Normalize assuming max 5 years
        features.append(1 - exp_diff)
        
        # Mode match (binary)
        mode_match = 1.0 if user_prefs['mode'].lower() == internship['mode'].lower() else 0.0
        features.append(mode_match)
        
        # Location match
        location_match = self._calculate_location_match(user_prefs['location'], internship['location'])
        features.append(location_match)
        
        # Sector match (binary)
        sector_match = 1.0 if user_prefs['sector'].lower() == internship['sector'].lower() else 0.0
        features.append(sector_match)
        
        # Stipend ratio
        stipend_ratio = min(internship['stipend'] / max(user_prefs['stipend'], 1), 2.0) / 2.0
        features.append(stipend_ratio)
        
        # Duration match
        duration_diff = abs(user_prefs['duration'] - internship['duration']) / 6.0
        duration_match = 1 - duration_diff
        features.append(max(duration_match, 0))
        
        return features
    
    def _calculate_synthetic_score(self, user_prefs: Dict, internship: pd.Series) -> float:
        """Calculate synthetic score for training (using the original weighted approach)"""
        weights = {
            'skills': 0.40,
            'experience': 0.10,
            'mode': 0.10,
            'location': 0.10,
            'sector': 0.05,
            'stipend': 0.15,
            'duration': 0.10
        }
        
        # Calculate individual scores
        skill_score = self._calculate_skill_similarity(user_prefs['skills'], internship['requirements']) * 10
        
        experience_score = 8.0 if user_prefs['experience'] <= 1 else (9.0 if user_prefs['experience'] <= 3 else 7.0)
        
        mode_score = 10.0 if user_prefs['mode'].lower() == internship['mode'].lower() else (7.0 if user_prefs['mode'].lower() == "hybrid" else 3.0)
        
        location_score = 10.0 if self._calculate_location_match(user_prefs['location'], internship['location']) == 1.0 else 2.0
        
        sector_score = 10.0 if user_prefs['sector'].lower() == internship['sector'].lower() else 1.0
        
        stipend_score = 10.0 if internship['stipend'] >= user_prefs['stipend'] else min((internship['stipend'] / user_prefs['stipend']) * 10, 10.0)
        
        duration_diff = abs(user_prefs['duration'] - internship['duration'])
        duration_score = max(10.0 - (duration_diff * 2), 1.0)
        
        # Calculate weighted total score
        total_score = (
            skill_score * weights['skills'] +
            experience_score * weights['experience'] +
            mode_score * weights['mode'] +
            location_score * weights['location'] +
            sector_score * weights['sector'] +
            stipend_score * weights['stipend'] +
            duration_score * weights['duration']
        )
        
        return total_score
    
    def train_model(self, verbose=True):
        """Train the ML model"""
        if verbose:
            print("Training internship matching model...")
        
        # Create training data
        X, y = self._create_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        if verbose:
            print(f"Training R² Score: {train_score:.4f}")
            print(f"Testing R² Score: {test_score:.4f}")
        
        self.is_trained = True
        self.save_model()
        if verbose:
            print("Model training completed and saved!")
    
    def save_model(self, verbose=True):
        """Save the trained model and preprocessors"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.tfidf_vectorizer, self.vectorizer_path)
        joblib.dump(self.scaler, self.scaler_path)
        if verbose:
            print("Model saved successfully!")
    
    def load_model(self, verbose=False):
        """Load the trained model and preprocessors"""
        if all(os.path.exists(path) for path in [self.model_path, self.vectorizer_path, self.scaler_path]):
            self.model = joblib.load(self.model_path)
            self.tfidf_vectorizer = joblib.load(self.vectorizer_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            if verbose:
                print("Model loaded successfully!")
            return True
        return False
    
    def predict_matches(self, user_preferences: Dict) -> List[Dict]:
        """Predict matching scores for all internships"""
        if not self.is_trained:
            if not self.load_model():
                self.train_model(verbose=False)
        
        matches = []
        
        for idx, internship in self.df.iterrows():
            # Extract features
            features = self._extract_features(user_preferences, internship)
            features_scaled = self.scaler.transform([features])
            
            # Predict score
            predicted_score = self.model.predict(features_scaled)[0]
            predicted_score = max(0, min(10, predicted_score))  # Clamp between 0-10
            
            match_data = {
                'name': internship['name'],
                'location': internship['location'],
                'stipend': int(internship['stipend']),
                'requirements': internship['requirements'],
                'mode': internship['mode'],
                'sector': internship['sector'],
                'duration': int(internship['duration']),
                'matching_score': round(predicted_score, 2)
            }
            
            matches.append(match_data)
        
        # Sort by score (descending) and return all matches
        matches.sort(key=lambda x: x['matching_score'], reverse=True)
        return matches