from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class ModeEnum(str, Enum):
    HYBRID = "Hybrid"
    ONSITE = "On-site"
    REMOTE = "Remote"

class SectorEnum(str, Enum):
    TECH = "tech"
    NON_TECH = "non-tech"

class StipendRange(BaseModel):
    min_stipend: int
    max_stipend: int

class UserPreferences(BaseModel):
    skills: List[str]
    experience: int
    mode: ModeEnum
    location: str
    sector: SectorEnum
    stipend: StipendRange
    duration: int

class ScoreExplanation(BaseModel):
    skills_explanation: str
    experience_explanation: str
    mode_explanation: str
    location_explanation: str
    sector_explanation: str
    stipend_explanation: str
    duration_explanation: str

class IndividualScores(BaseModel):
    skills_matching: float  # 0-10: How well user skills match internship requirements
    experience_matching: float  # 0-10: Experience level compatibility
    mode_matching: float  # 0-10: Work mode preference alignment (Remote/Hybrid/On-site)
    location_matching: float  # 0-10: Geographic location compatibility
    sector_matching: float  # 0-10: Industry sector alignment (tech/non-tech)
    stipend_matching: float  # 0-10: Stipend expectation vs offered amount
    duration_matching: float  # 0-10: Internship duration preference match
    explanations: Optional[ScoreExplanation] = None

class InternshipMatch(BaseModel):
    id: int
    name: str
    location: str
    stipend: int
    requirements: str
    mode: str
    sector: str
    duration: int
    matching_score: float
    individual_scores: IndividualScores

class MatchingResponse(BaseModel):
    matches: List[InternshipMatch]
    total_matches: int