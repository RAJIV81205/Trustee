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

class UserPreferences(BaseModel):
    skills: List[str]
    experience: int
    mode: ModeEnum
    location: str
    sector: SectorEnum
    stipend: int
    duration: int

class InternshipMatch(BaseModel):
    name: str
    location: str
    stipend: int
    requirements: str
    mode: str
    sector: str
    duration: int
    matching_score: float

class MatchingResponse(BaseModel):
    matches: List[InternshipMatch]
    total_matches: int