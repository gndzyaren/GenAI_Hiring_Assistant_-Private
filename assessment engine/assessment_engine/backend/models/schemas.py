from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionFormat(str, Enum):
    MCQ = "mcq"
    NUMERICAL = "numerical"
    CODING = "coding"


class QuestionSection(str, Enum):
    MENTAL_ABILITY = "mental_ability"
    TECHNICAL = "technical"
    CODING = "coding"


class MCQOption(BaseModel):
    label: str  # A, B, C, D
    text: str


class Question(BaseModel):
    id: str
    section: QuestionSection
    format: QuestionFormat
    difficulty: DifficultyLevel
    topic: str
    question_text: str
    options: Optional[List[MCQOption]] = None       # MCQ only
    correct_answer: str
    explanation: str
    hints: Optional[List[str]] = None
    test_cases: Optional[List[Dict[str, Any]]] = None  # coding only
    hidden_test_cases: Optional[List[Dict[str, Any]]] = None
    time_limit_seconds: Optional[int] = None
    irt_difficulty: float = 0.0       # IRT b-parameter
    irt_discrimination: float = 1.0   # IRT a-parameter
    embedding: Optional[List[float]] = None


class JDInput(BaseModel):
    job_title: str
    job_description: str
    required_skills: Optional[List[str]] = None
    programming_languages: Optional[List[str]] = Field(default=["Python"])
    experience_level: Optional[str] = "mid"


class AssessmentConfig(BaseModel):
    jd: JDInput
    mental_ability_count: int = 10
    technical_count: int = 10
    coding_count: int = 3
    start_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM


class CandidateAnswer(BaseModel):
    question_id: str
    answer: str
    time_taken_seconds: int


class CandidateResponse(BaseModel):
    session_id: str
    question_id: str
    answer: str
    time_taken_seconds: int


class SessionState(BaseModel):
    session_id: str
    assessment_id: str
    current_question_index: int = 0
    theta: float = 0.0          # IRT ability estimate
    responses: List[CandidateAnswer] = []
    current_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    questions: List[Question] = []
    is_complete: bool = False


class QuestionResult(BaseModel):
    question_id: str
    question_text: str
    section: str
    topic: str
    candidate_answer: str
    correct_answer: str
    is_correct: bool
    difficulty: str
    explanation: str
    time_taken_seconds: int


class SectionReport(BaseModel):
    section: str
    total: int
    correct: int
    accuracy: float
    avg_time_seconds: float
    difficulty_distribution: Dict[str, int]


class AssessmentReport(BaseModel):
    session_id: str
    candidate_theta: float
    ability_level: str
    total_questions: int
    total_correct: int
    overall_accuracy: float
    total_time_minutes: float
    section_reports: List[SectionReport]
    question_results: List[QuestionResult]
    strengths: List[str]
    improvement_areas: List[str]
    recommendations: str
