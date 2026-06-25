"""
Assessment API Routes
"""

from fastapi import APIRouter, HTTPException
from backend.models.schemas import (
    AssessmentConfig, CandidateResponse, AssessmentReport
)
from backend.services.orchestrator import create_assessment
from backend.services.cat_engine import session_manager, build_report
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/assessment", tags=["Assessment"])


@router.post("/create")
async def create_new_assessment(config: AssessmentConfig):
    """
    Create a new adaptive assessment from a Job Description.
    Returns session_id and metadata.
    """
    try:
        result = create_assessment(config)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Assessment creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/next-question")
async def get_next_question(session_id: str):
    """
    Get the next adaptive question for a session.
    Returns None if assessment is complete.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    question = session.get_next_question()
    if not question:
        return {"status": "complete", "question": None}

    # Return question without correct answer and hidden test cases
    q_dict = question.dict()
    q_dict.pop("correct_answer", None)
    q_dict.pop("hidden_test_cases", None)
    q_dict.pop("embedding", None)

    return {
        "status": "active",
        "question": q_dict,
        "progress": {
            "answered": len(session.responses),
            "total": len(session.all_questions),
            "current_theta": round(session.theta, 3),
            "current_difficulty": session.current_difficulty.value,
        }
    }


@router.post("/{session_id}/submit-answer")
async def submit_answer(session_id: str, response: CandidateResponse):
    """
    Submit an answer for the current question.
    Returns correctness and updated ability estimate.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the question
    question = next(
        (q for q in session.all_questions if q.id == response.question_id), None
    )
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    is_correct = session.record_response(question, response.answer, response.time_taken_seconds)

    is_complete = session.is_complete(len(session.all_questions))

    return {
        "is_correct": is_correct,
        "correct_answer": question.correct_answer,
        "explanation": question.explanation,
        "updated_theta": round(session.theta, 3),
        "next_difficulty": session.current_difficulty.value,
        "is_assessment_complete": is_complete,
    }


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    """Generate and return the final assessment report."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    report = build_report(session, session.all_questions)
    return {"status": "success", "report": report.dict()}


@router.get("/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current session status."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "questions_answered": len(session.responses),
        "total_questions": len(session.all_questions),
        "theta": round(session.theta, 3),
        "current_difficulty": session.current_difficulty.value,
        "is_complete": session.is_complete(len(session.all_questions)),
    }
