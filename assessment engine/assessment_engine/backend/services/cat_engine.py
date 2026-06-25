"""
Computer Adaptive Testing (CAT) Engine
Implements Item Response Theory (IRT) 2-Parameter Logistic Model for ability estimation.

The 2PL IRT Model:
  P(correct | θ, a, b) = 1 / (1 + exp(-a * (θ - b)))

  θ (theta) = candidate ability estimate
  a = item discrimination (how well item differentiates ability levels)
  b = item difficulty (ability level at which P(correct) = 0.5)

Theta update: Maximum Likelihood Estimation (MLE) via Newton-Raphson
"""

import math
import logging
import uuid
from typing import List, Tuple, Dict, Optional
from backend.models.schemas import (
    Question, DifficultyLevel, SessionState, CandidateAnswer, AssessmentReport,
    QuestionResult, SectionReport
)

logger = logging.getLogger(__name__)

# Theta bounds
THETA_MIN = -4.0
THETA_MAX = 4.0
THETA_STEP = 0.3  # Coarse adjustment when MLE is unstable


class IRTEngine:
    """2-Parameter Logistic IRT model."""

    def probability_correct(self, theta: float, a: float, b: float) -> float:
        """P(correct | theta, a, b) using 2PL model."""
        try:
            return 1.0 / (1.0 + math.exp(-a * (theta - b)))
        except OverflowError:
            return 0.0 if a * (theta - b) < 0 else 1.0

    def log_likelihood(self, theta: float, responses: List[Tuple[float, float, int]]) -> float:
        """
        Compute log-likelihood of theta given responses.
        responses: list of (a, b, u) where u=1 correct, u=0 incorrect
        """
        ll = 0.0
        for a, b, u in responses:
            p = self.probability_correct(theta, a, b)
            p = max(p, 1e-10)
            p = min(p, 1 - 1e-10)
            ll += u * math.log(p) + (1 - u) * math.log(1 - p)
        return ll

    def estimate_theta(self, responses: List[Tuple[float, float, int]],
                       current_theta: float = 0.0) -> float:
        """
        Newton-Raphson MLE for theta estimation.
        Falls back to EAP (expected value) if all correct or all incorrect.
        """
        if not responses:
            return current_theta

        # All correct or all incorrect → theta can't be estimated reliably
        scores = [u for _, _, u in responses]
        if all(s == 1 for s in scores):
            return min(current_theta + THETA_STEP * len(responses), THETA_MAX)
        if all(s == 0 for s in scores):
            return max(current_theta + THETA_STEP * len(responses) * -1, THETA_MIN)

        # Newton-Raphson optimization
        theta = current_theta
        for _ in range(50):  # max iterations
            d1 = 0.0  # first derivative
            d2 = 0.0  # second derivative (Fisher info)
            for a, b, u in responses:
                p = self.probability_correct(theta, a, b)
                q = 1 - p
                d1 += a * (u - p)
                d2 -= a * a * p * q
            if abs(d2) < 1e-10:
                break
            delta = d1 / d2
            theta -= delta
            theta = max(THETA_MIN, min(THETA_MAX, theta))
            if abs(delta) < 0.001:
                break

        return round(theta, 4)

    def fisher_information(self, theta: float, a: float, b: float) -> float:
        """Fisher information for a single item."""
        p = self.probability_correct(theta, a, b)
        return a * a * p * (1 - p)

    def select_next_difficulty(self, theta: float, last_correct: bool,
                               question_count: int) -> DifficultyLevel:
        """
        Select next question difficulty based on theta and recent performance.
        Uses theta thresholds with smoothing.
        """
        # Early questions: use simple correct/incorrect rule
        if question_count < 3:
            if last_correct:
                return DifficultyLevel.HARD if question_count > 0 else DifficultyLevel.MEDIUM
            else:
                return DifficultyLevel.EASY
        # IRT-guided selection
        if theta >= 1.0:
            return DifficultyLevel.HARD
        elif theta >= -0.5:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY


irt_engine = IRTEngine()


class CATSession:
    """Manages a single candidate's adaptive test session."""

    def __init__(self, session_id: str, questions: List[Question]):
        self.session_id = session_id
        self.all_questions = questions
        self.asked_ids: set = set()
        self.responses: List[CandidateAnswer] = []
        self.theta: float = 0.0
        self.current_difficulty = DifficultyLevel.MEDIUM
        self.question_index = 0

    def get_next_question(self) -> Optional[Question]:
        """Get next question matching current difficulty."""
        available = [
            q for q in self.all_questions
            if q.id not in self.asked_ids and q.difficulty == self.current_difficulty
        ]
        if not available:
            # Fallback to any unasked question
            available = [q for q in self.all_questions if q.id not in self.asked_ids]
        if not available:
            return None
        # Pick question with maximum Fisher information at current theta
        best = max(available, key=lambda q: irt_engine.fisher_information(
            self.theta, q.irt_discrimination, q.irt_difficulty))
        self.asked_ids.add(best.id)
        self.question_index += 1
        return best

    def record_response(self, question: Question, answer: str,
                        time_taken: int) -> bool:
        """Record answer, update theta, set next difficulty. Returns is_correct."""
        is_correct = self._check_answer(question, answer)
        self.responses.append(CandidateAnswer(
            question_id=question.id,
            answer=answer,
            time_taken_seconds=time_taken
        ))

        # Build IRT response tuples
        irt_responses = []
        for resp in self.responses:
            q = next((q for q in self.all_questions if q.id == resp.question_id), None)
            if q:
                u = 1 if self._check_answer(q, resp.answer) else 0
                irt_responses.append((q.irt_discrimination, q.irt_difficulty, u))

        self.theta = irt_engine.estimate_theta(irt_responses, self.theta)
        self.current_difficulty = irt_engine.select_next_difficulty(
            self.theta, is_correct, len(self.responses)
        )
        logger.debug(f"Response recorded. Theta={self.theta:.3f}, "
                     f"Next difficulty={self.current_difficulty.value}")
        return is_correct

    def _check_answer(self, question: Question, answer: str) -> bool:
        """Check if answer is correct (case-insensitive, stripped)."""
        correct = question.correct_answer.strip().upper()
        given = answer.strip().upper()
        return correct == given or given in correct or correct in given

    def is_complete(self, total_questions: int) -> bool:
        return len(self.asked_ids) >= total_questions


class SessionManager:
    """In-memory session store."""

    def __init__(self):
        self._sessions: Dict[str, CATSession] = {}

    def create_session(self, questions: List[Question]) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = CATSession(session_id, questions)
        logger.info(f"Created CAT session {session_id} with {len(questions)} questions.")
        return session_id

    def get_session(self, session_id: str) -> Optional[CATSession]:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str):
        self._sessions.pop(session_id, None)


session_manager = SessionManager()


def build_report(session: CATSession, all_questions: List[Question]) -> AssessmentReport:
    """Generate comprehensive assessment report from a completed session."""
    question_map = {q.id: q for q in all_questions}
    results = []
    section_data: Dict[str, dict] = {}

    for resp in session.responses:
        q = question_map.get(resp.question_id)
        if not q:
            continue
        is_correct = session._check_answer(q, resp.answer)
        result = QuestionResult(
            question_id=q.id,
            question_text=q.question_text,
            section=q.section.value,
            topic=q.topic,
            candidate_answer=resp.answer,
            correct_answer=q.correct_answer,
            is_correct=is_correct,
            difficulty=q.difficulty.value,
            explanation=q.explanation,
            time_taken_seconds=resp.time_taken_seconds
        )
        results.append(result)

        sec = q.section.value
        if sec not in section_data:
            section_data[sec] = {"total": 0, "correct": 0, "times": [], "difficulties": {}}
        section_data[sec]["total"] += 1
        section_data[sec]["correct"] += int(is_correct)
        section_data[sec]["times"].append(resp.time_taken_seconds)
        d = q.difficulty.value
        section_data[sec]["difficulties"][d] = section_data[sec]["difficulties"].get(d, 0) + 1

    section_reports = []
    for sec, data in section_data.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        section_reports.append(SectionReport(
            section=sec,
            total=data["total"],
            correct=data["correct"],
            accuracy=round(acc, 3),
            avg_time_seconds=round(sum(data["times"]) / len(data["times"]), 1) if data["times"] else 0,
            difficulty_distribution=data["difficulties"]
        ))

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    overall_acc = correct / total if total > 0 else 0
    total_time = sum(r.time_taken_seconds for r in results) / 60

    theta = session.theta
    ability_level = (
        "Advanced" if theta >= 1.5 else
        "Proficient" if theta >= 0.5 else
        "Developing" if theta >= -0.5 else
        "Beginner"
    )

    # Generate strengths and improvement areas
    strengths = []
    improvements = []
    for sr in section_reports:
        if sr.accuracy >= 0.7:
            strengths.append(f"{sr.section.replace('_', ' ').title()} ({sr.accuracy*100:.0f}% accuracy)")
        else:
            improvements.append(f"{sr.section.replace('_', ' ').title()} ({sr.accuracy*100:.0f}% accuracy)")

    recommendations = _generate_recommendations(ability_level, section_reports, theta)

    return AssessmentReport(
        session_id=session.session_id,
        candidate_theta=round(theta, 3),
        ability_level=ability_level,
        total_questions=total,
        total_correct=correct,
        overall_accuracy=round(overall_acc, 3),
        total_time_minutes=round(total_time, 1),
        section_reports=section_reports,
        question_results=results,
        strengths=strengths,
        improvement_areas=improvements,
        recommendations=recommendations
    )


def _generate_recommendations(ability: str, sections: List[SectionReport], theta: float) -> str:
    parts = []
    if theta < -0.5:
        parts.append("Focus on foundational concepts before attempting advanced topics.")
    elif theta < 0.5:
        parts.append("Good foundation. Work on applied problem-solving and timed practice.")
    else:
        parts.append("Strong performance. Challenge yourself with system design and optimization problems.")

    for sr in sections:
        if sr.accuracy < 0.5:
            parts.append(f"Prioritize {sr.section.replace('_', ' ')} — review core concepts and practice more problems.")

    return " ".join(parts)
