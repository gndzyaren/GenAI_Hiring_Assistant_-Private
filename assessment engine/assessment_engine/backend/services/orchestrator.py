"""
Assessment Orchestrator
Coordinates JD analysis, question generation, deduplication, and session creation.
"""

import logging
import random
from typing import List, Dict, Optional
from backend.models.schemas import (
    AssessmentConfig, Question, QuestionSection,
    QuestionFormat, DifficultyLevel
)
from backend.services.jd_analyzer import analyze_jd
from backend.services.question_generator import question_generator
from backend.services.deduplication import dedup_service
from backend.services.cat_engine import session_manager, CATSession

logger = logging.getLogger(__name__)

DIFFICULTIES = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]


def _difficulty_distribution(count: int) -> List[DifficultyLevel]:
    """Return a list of difficulty levels: 30% easy, 40% medium, 30% hard."""
    easy = max(1, round(count * 0.3))
    hard = max(1, round(count * 0.3))
    medium = count - easy - hard
    return ([DifficultyLevel.EASY] * easy +
            [DifficultyLevel.MEDIUM] * medium +
            [DifficultyLevel.HARD] * hard)


def _generate_section_questions(
    topics: List[str],
    section: QuestionSection,
    count: int,
    context: str = "",
    prefer_format: Optional[QuestionFormat] = None,
) -> List[Question]:
    """
    Generate `count` questions for a section across given topics.
    Mixes MCQ and Numerical formats for non-coding sections.
    """
    questions = []
    difficulties = _difficulty_distribution(count)
    random.shuffle(difficulties)
    topic_cycle = topics * (count // len(topics) + 1)

    for i, difficulty in enumerate(difficulties):
        topic = topic_cycle[i % len(topic_cycle)]

        # Decide format
        if prefer_format:
            fmt = prefer_format
        elif section == QuestionSection.MENTAL_ABILITY:
            fmt = random.choice([QuestionFormat.MCQ, QuestionFormat.NUMERICAL])
        else:
            fmt = random.choice([QuestionFormat.MCQ, QuestionFormat.NUMERICAL])

        q = None
        # Try model generation
        try:
            if fmt == QuestionFormat.MCQ:
                q = question_generator.generate_mcq(topic, section, difficulty, context)
            else:
                q = question_generator.generate_numerical(topic, section, difficulty, context)
        except Exception as e:
            logger.warning(f"Model generation failed: {e}. Using fallback.")

        # Fallback to template
        if q is None:
            q = question_generator.get_fallback_question(section, difficulty)

        questions.append(q)

    return questions


def _generate_coding_questions(
    topics: List[str],
    count: int,
    languages: List[str]
) -> List[Question]:
    """Generate coding problems."""
    questions = []
    difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD][:count]

    for i in range(count):
        topic = topics[i % len(topics)]
        difficulty = difficulties[i % len(difficulties)]
        lang = languages[i % len(languages)]

        q = None
        try:
            q = question_generator.generate_coding(topic, difficulty, lang)
        except Exception as e:
            logger.warning(f"Coding generation failed: {e}. Using fallback.")

        if q is None:
            q = question_generator.get_fallback_question(QuestionSection.CODING, difficulty)

        questions.append(q)

    return questions


def create_assessment(config: AssessmentConfig) -> Dict:
    """
    Full pipeline:
    1. Analyze JD
    2. Generate questions per section
    3. Deduplicate
    4. Create CAT session
    Returns assessment metadata and session_id.
    """
    logger.info(f"Creating assessment for: {config.jd.job_title}")

    # Step 1: Analyze JD
    jd_analysis = analyze_jd(config.jd)
    logger.info(f"Detected domains: {jd_analysis['detected_domains']}")

    context = f"Job: {config.jd.job_title}. Experience: {jd_analysis['experience_level']}."

    all_questions: List[Question] = []

    # Step 2a: Mental Ability questions
    logger.info("Generating Mental Ability questions...")
    mental_qs = _generate_section_questions(
        topics=jd_analysis["mental_ability_topics"],
        section=QuestionSection.MENTAL_ABILITY,
        count=config.mental_ability_count,
        context=context
    )
    all_questions.extend(mental_qs)
    logger.info(f"Generated {len(mental_qs)} mental ability questions.")

    # Step 2b: Technical questions
    logger.info("Generating Technical questions...")
    tech_qs = _generate_section_questions(
        topics=jd_analysis["technical_topics"],
        section=QuestionSection.TECHNICAL,
        count=config.technical_count,
        context=context
    )
    all_questions.extend(tech_qs)
    logger.info(f"Generated {len(tech_qs)} technical questions.")

    # Step 2c: Coding problems
    logger.info("Generating Coding problems...")
    coding_topics = jd_analysis["technical_topics"][:5] + ["Arrays", "Recursion", "Strings"]
    coding_qs = _generate_coding_questions(
        topics=coding_topics,
        count=config.coding_count,
        languages=jd_analysis["programming_languages"]
    )
    all_questions.extend(coding_qs)
    logger.info(f"Generated {len(coding_qs)} coding problems.")

    # Step 3: Deduplicate
    logger.info("Running deduplication...")
    unique_questions = dedup_service.filter_unique(all_questions)
    logger.info(f"Unique questions after dedup: {len(unique_questions)}")

    # Ensure minimum questions per section
    if len(unique_questions) < 5:
        logger.warning("Too many duplicates removed. Adding fallback questions.")
        for section in [QuestionSection.MENTAL_ABILITY, QuestionSection.TECHNICAL, QuestionSection.CODING]:
            unique_questions.append(
                question_generator.get_fallback_question(section, DifficultyLevel.MEDIUM)
            )

    # Step 4: Create CAT session
    session_id = session_manager.create_session(unique_questions)

    return {
        "session_id": session_id,
        "total_questions": len(unique_questions),
        "jd_analysis": jd_analysis,
        "sections": {
            "mental_ability": len([q for q in unique_questions if q.section == QuestionSection.MENTAL_ABILITY]),
            "technical": len([q for q in unique_questions if q.section == QuestionSection.TECHNICAL]),
            "coding": len([q for q in unique_questions if q.section == QuestionSection.CODING]),
        },
        "dedup_stats": dedup_service.get_stats()
    }
