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
            {'skills': ['python', 'machine learning'], 'experience': 1, 'mode': 'hybrid', 'location': 'karnataka', 'sector': 'tech', 'stipend': {'min_stipend': 20000, 'max_stipend': 30000}, 'duration': 6},
            {'skills': ['javascript', 'react'], 'experience': 0, 'mode': 'remote', 'location': 'maharashtra', 'sector': 'tech', 'stipend': {'min_stipend': 15000, 'max_stipend': 25000}, 'duration': 3},
            {'skills': ['marketing', 'social media'], 'experience': 2, 'mode': 'on-site', 'location': 'delhi', 'sector': 'non-tech', 'stipend': {'min_stipend': 12000, 'max_stipend': 18000}, 'duration': 2},
            {'skills': ['java', 'spring'], 'experience': 1, 'mode': 'on-site', 'location': 'telangana', 'sector': 'tech', 'stipend': {'min_stipend': 25000, 'max_stipend': 35000}, 'duration': 6},
            {'skills': ['design', 'photoshop'], 'experience': 0, 'mode': 'remote', 'location': 'delhi', 'sector': 'non-tech', 'stipend': {'min_stipend': 14000, 'max_stipend': 20000}, 'duration': 3}
        ]
        
        for user in sample_users:
            for idx, internship in self.df.iterrows():
                feature_vector = self._extract_features(user, internship)
                synthetic_score = self._calculate_synthetic_score(user, internship)
                features.append(feature_vector)
                scores.append(synthetic_score)
        
        return np.array(features), np.array(scores)
    
    def _calculate_stipend_match(self, stipend_range: Dict, internship_stipend: int) -> float:
        """Calculate stipend matching score based on range"""
        min_stipend = stipend_range['min_stipend']
        max_stipend = stipend_range['max_stipend']
        
        # If internship stipend is within range, perfect match
        if min_stipend <= internship_stipend <= max_stipend:
            return 1.0
        
        # Calculate how far outside the range
        if internship_stipend < min_stipend:
            # Below minimum - penalize based on how far below
            diff = min_stipend - internship_stipend
            max_penalty = min_stipend * 0.5  # 50% below minimum gets 0 score
            penalty = min(diff / max_penalty, 1.0)
            return 1.0 - penalty
        else:
            # Above maximum - slight bonus for higher stipend, but cap at 1.0
            return 1.0
    
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
        
        # Stipend match using range
        stipend_match = self._calculate_stipend_match(user_prefs['stipend'], internship['stipend'])
        features.append(stipend_match)
        
        # Duration match
        duration_diff = abs(user_prefs['duration'] - internship['duration']) / 6.0
        duration_match = 1 - duration_diff
        features.append(max(duration_match, 0))
        
        return features
    
    def _calculate_refined_skill_score(self, user_skills: List[str], internship_requirements: str) -> float:
        """Calculate refined skill matching score with better granularity"""
        # Basic TF-IDF similarity
        base_similarity = self._calculate_skill_similarity(user_skills, internship_requirements)
        
        # Parse internship requirements
        internship_skills = [skill.strip().lower() for skill in internship_requirements.split(',')]
        user_skills_lower = [skill.strip().lower() for skill in user_skills]
        
        # Exact matches count
        exact_matches = len(set(user_skills_lower) & set(internship_skills))
        total_required = len(internship_skills)
        
        # Partial matches (substring matching for related skills)
        partial_matches = 0
        for user_skill in user_skills_lower:
            for req_skill in internship_skills:
                if user_skill in req_skill or req_skill in user_skill:
                    if user_skill not in req_skill and req_skill not in user_skill:  # Avoid double counting exact matches
                        partial_matches += 0.5
        
        # Calculate refined score
        if total_required == 0:
            return 5.0  # Neutral score if no requirements specified
        
        exact_score = (exact_matches / total_required) * 10
        partial_score = min(partial_matches, total_required - exact_matches) / total_required * 3
        tfidf_bonus = base_similarity * 2  # TF-IDF provides additional context
        
        final_score = min(exact_score + partial_score + tfidf_bonus, 10.0)
        return max(final_score, 1.0)  # Minimum score of 1.0
    
    def _calculate_refined_experience_score(self, user_experience: int, internship_requirements: str) -> float:
        """Calculate refined experience matching score"""
        # Extract experience requirements from internship description (if any)
        # For now, assume most internships are entry-level friendly
        
        if user_experience == 0:
            # Fresh graduates - good for most internships
            return 8.5
        elif user_experience == 1:
            # 1 year experience - excellent for internships
            return 9.5
        elif user_experience == 2:
            # 2 years - still good but might be overqualified for some
            return 8.0
        elif user_experience >= 3:
            # 3+ years - might be overqualified for internships
            return 6.5
        else:
            return 7.0
    
    def _calculate_refined_mode_score(self, user_mode: str, internship_mode: str) -> float:
        """Calculate refined work mode matching score"""
        user_mode_clean = user_mode.lower().strip()
        internship_mode_clean = internship_mode.lower().strip()
        
        if user_mode_clean == internship_mode_clean:
            return 10.0  # Perfect match
        
        # Compatibility matrix
        compatibility = {
            'remote': {'hybrid': 7.5, 'on-site': 3.0},
            'hybrid': {'remote': 8.0, 'on-site': 8.0},
            'on-site': {'hybrid': 6.5, 'remote': 2.0}
        }
        
        return compatibility.get(user_mode_clean, {}).get(internship_mode_clean, 4.0)
    
    def _calculate_refined_location_score(self, user_location: str, internship_location: str) -> float:
        """Calculate refined location matching score with partial matches"""
        user_state_clean = user_location.lower().strip()
        internship_state_clean = internship_location.lower().strip()
        
        # Direct state match
        if user_state_clean == internship_state_clean:
            return 10.0
        
        # Check state-city relationships
        for state, cities in self.state_mapping.items():
            if state == internship_state_clean and user_state_clean in cities:
                return 10.0  # User city matches internship state
            if state == user_state_clean and internship_state_clean in cities:
                return 10.0  # User state matches internship city
        
        # Check for neighboring states or similar regions (basic implementation)
        neighboring_regions = {
            'maharashtra': ['gujarat', 'karnataka', 'goa'],
            'karnataka': ['maharashtra', 'tamil nadu', 'telangana', 'kerala'],
            'delhi': ['haryana', 'uttar pradesh', 'punjab'],
            'telangana': ['karnataka', 'maharashtra', 'andhra pradesh'],
            'tamil nadu': ['karnataka', 'kerala', 'andhra pradesh'],
            'gujarat': ['maharashtra', 'rajasthan', 'madhya pradesh'],
            'rajasthan': ['gujarat', 'haryana', 'delhi', 'madhya pradesh'],
            'haryana': ['delhi', 'punjab', 'rajasthan', 'uttar pradesh']
        }
        
        if user_state_clean in neighboring_regions.get(internship_state_clean, []):
            return 6.0  # Neighboring state bonus
        
        return 2.0  # Different regions
    
    def _calculate_refined_stipend_score(self, stipend_range: Dict, internship_stipend: int) -> float:
        """Calculate refined stipend matching score with better granularity"""
        min_stipend = stipend_range['min_stipend']
        max_stipend = stipend_range['max_stipend']
        
        # Perfect range match
        if min_stipend <= internship_stipend <= max_stipend:
            # Score based on position within range (higher is better)
            range_size = max_stipend - min_stipend
            if range_size > 0:
                position_in_range = (internship_stipend - min_stipend) / range_size
                return 9.0 + (position_in_range * 1.0)  # 9.0 to 10.0
            return 10.0
        
        # Below minimum
        if internship_stipend < min_stipend:
            diff_percentage = (min_stipend - internship_stipend) / min_stipend
            if diff_percentage <= 0.1:  # Within 10% below
                return 7.5
            elif diff_percentage <= 0.2:  # Within 20% below
                return 6.0
            elif diff_percentage <= 0.3:  # Within 30% below
                return 4.0
            else:
                return 2.0
        
        # Above maximum (bonus for higher stipend)
        else:
            diff_percentage = (internship_stipend - max_stipend) / max_stipend
            if diff_percentage <= 0.2:  # Up to 20% above
                return 10.0
            elif diff_percentage <= 0.5:  # Up to 50% above
                return 9.5
            else:
                return 9.0  # Very high stipend, still good
    
    def _calculate_refined_duration_score(self, user_duration: int, internship_duration: int) -> float:
        """Calculate refined duration matching score"""
        duration_diff = abs(user_duration - internship_duration)
        
        if duration_diff == 0:
            return 10.0  # Perfect match
        elif duration_diff == 1:
            return 8.5  # 1 month difference
        elif duration_diff == 2:
            return 7.0  # 2 months difference
        elif duration_diff == 3:
            return 5.5  # 3 months difference
        elif duration_diff <= 6:
            return 4.0  # Up to 6 months difference
        else:
            return 2.0  # More than 6 months difference
    
    def _generate_score_explanations(self, user_prefs: Dict, internship: pd.Series, scores: Dict) -> Dict[str, str]:
        """Generate explanations for each score"""
        explanations = {}
        
        # Skills explanation
        user_skills_lower = [skill.strip().lower() for skill in user_prefs['skills']]
        internship_skills = [skill.strip().lower() for skill in internship['requirements'].split(',')]
        matched_skills = set(user_skills_lower) & set(internship_skills)
        
        if scores['skills_matching'] >= 8.0:
            explanations['skills_explanation'] = f"Excellent match! You have {len(matched_skills)} matching skills: {', '.join(matched_skills)}"
        elif scores['skills_matching'] >= 6.0:
            explanations['skills_explanation'] = f"Good match with {len(matched_skills)} relevant skills. Some skill gaps to bridge."
        else:
            explanations['skills_explanation'] = f"Limited skill overlap. Consider developing: {', '.join(set(internship_skills) - set(user_skills_lower))}"
        
        # Experience explanation
        if scores['experience_matching'] >= 9.0:
            explanations['experience_explanation'] = "Perfect experience level for this internship"
        elif scores['experience_matching'] >= 7.0:
            explanations['experience_explanation'] = "Good experience level, well-suited for this role"
        else:
            explanations['experience_explanation'] = "Experience level may not be optimal for this internship"
        
        # Mode explanation
        if scores['mode_matching'] == 10.0:
            explanations['mode_explanation'] = "Perfect work mode match"
        elif scores['mode_matching'] >= 7.0:
            explanations['mode_explanation'] = "Compatible work modes with some flexibility"
        else:
            explanations['mode_explanation'] = f"Work mode mismatch: You prefer {user_prefs['mode']}, this is {internship['mode']}"
        
        # Location explanation
        if scores['location_matching'] == 10.0:
            explanations['location_explanation'] = "Perfect location match"
        elif scores['location_matching'] >= 6.0:
            explanations['location_explanation'] = "Good location compatibility (neighboring regions)"
        else:
            explanations['location_explanation'] = f"Location mismatch: You're in {user_prefs['location']}, internship is in {internship['location']}"
        
        # Sector explanation
        if scores['sector_matching'] == 10.0:
            explanations['sector_explanation'] = "Perfect sector alignment"
        else:
            explanations['sector_explanation'] = f"Cross-sector opportunity: You prefer {user_prefs['sector']}, this is {internship['sector']}"
        
        # Stipend explanation
        min_stipend = user_prefs['stipend']['min_stipend']
        max_stipend = user_prefs['stipend']['max_stipend']
        internship_stipend = internship['stipend']
        
        if min_stipend <= internship_stipend <= max_stipend:
            explanations['stipend_explanation'] = f"Stipend ₹{internship_stipend} is within your expected range"
        elif internship_stipend > max_stipend:
            explanations['stipend_explanation'] = f"Great! Stipend ₹{internship_stipend} exceeds your expectations"
        else:
            diff = min_stipend - internship_stipend
            explanations['stipend_explanation'] = f"Stipend ₹{internship_stipend} is ₹{diff} below your minimum expectation"
        
        # Duration explanation
        duration_diff = abs(user_prefs['duration'] - internship['duration'])
        if duration_diff == 0:
            explanations['duration_explanation'] = "Perfect duration match"
        else:
            explanations['duration_explanation'] = f"Duration differs by {duration_diff} months from your preference"
        
        return explanations
    
    def _extract_individual_scores(self, user_prefs: Dict, internship: pd.Series) -> Dict[str, float]:
        """Extract refined individual matching scores for detailed breakdown"""
        scores = {}
        
        # Refined skill matching
        scores['skills_matching'] = round(self._calculate_refined_skill_score(user_prefs['skills'], internship['requirements']), 1)
        
        # Refined experience matching
        scores['experience_matching'] = round(self._calculate_refined_experience_score(user_prefs['experience'], internship['requirements']), 1)
        
        # Refined mode matching
        scores['mode_matching'] = round(self._calculate_refined_mode_score(user_prefs['mode'], internship['mode']), 1)
        
        # Refined location matching
        scores['location_matching'] = round(self._calculate_refined_location_score(user_prefs['location'], internship['location']), 1)
        
        # Sector match (binary but refined)
        sector_match = 1.0 if user_prefs['sector'].lower() == internship['sector'].lower() else 0.0
        scores['sector_matching'] = 10.0 if sector_match == 1.0 else 3.0  # Some cross-sector opportunities
        
        # Refined stipend matching
        scores['stipend_matching'] = round(self._calculate_refined_stipend_score(user_prefs['stipend'], internship['stipend']), 1)
        
        # Refined duration matching
        scores['duration_matching'] = round(self._calculate_refined_duration_score(user_prefs['duration'], internship['duration']), 1)
        
        return scores
    
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
        
        stipend_score = self._calculate_stipend_match(user_prefs['stipend'], internship['stipend']) * 10
        
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
        
        all_matches = []
        
        for idx, internship in self.df.iterrows():
            # Extract features
            features = self._extract_features(user_preferences, internship)
            features_scaled = self.scaler.transform([features])
            
            # Predict score
            predicted_score = self.model.predict(features_scaled)[0]
            predicted_score = max(0, min(10, predicted_score))  # Clamp between 0-10
            
            # Get individual scores
            individual_scores = self._extract_individual_scores(user_preferences, internship)
            
            match_data = {
                'id': int(internship['id']),
                'name': internship['name'],
                'company_name': internship['company_name'],
                'company_description': internship['company_description'],
                'location': internship['location'],
                'stipend': int(internship['stipend']),
                'requirements': internship['requirements'],
                'mode': internship['mode'],
                'sector': internship['sector'],
                'duration': int(internship['duration']),
                'matching_score': round(predicted_score, 2),
                'individual_scores': individual_scores
            }
            
            all_matches.append(match_data)
        
        # Sort by score (descending)
        all_matches.sort(key=lambda x: x['matching_score'], reverse=True)
        
        # Filter matches with score > 7
        high_score_matches = [match for match in all_matches if match['matching_score'] > 7.0]
        
        # If no matches above 7, return all matches (fallback)
        if not high_score_matches:
            return all_matches
        
        return high_score_matches