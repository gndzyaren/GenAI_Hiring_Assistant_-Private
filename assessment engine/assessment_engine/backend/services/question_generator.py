"""
Question Generation Service
Uses Mistral-7B-Instruct locally to generate MCQ, Numerical, and Coding questions.
Falls back to a template bank if model is not loaded yet.
"""

import json
import uuid
import re
import logging
from typing import List, Optional, Dict
from backend.models.schemas import (
    Question, QuestionSection, QuestionFormat,
    DifficultyLevel, MCQOption
)
from backend.services.model_loader import model_loader

logger = logging.getLogger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────────────

MCQ_PROMPT = """<s>[INST] You are an expert technical assessment designer. Generate exactly 1 multiple-choice question.

Topic: {topic}
Section: {section}
Difficulty: {difficulty}
Context: {context}

Respond ONLY with valid JSON in this exact format:
{{
  "question_text": "...",
  "options": [
    {{"label": "A", "text": "..."}},
    {{"label": "B", "text": "..."}},
    {{"label": "C", "text": "..."}},
    {{"label": "D", "text": "..."}}
  ],
  "correct_answer": "A",
  "explanation": "Step-by-step explanation here...",
  "hints": ["hint 1", "hint 2"]
}}
[/INST]"""

NUMERICAL_PROMPT = """<s>[INST] You are an expert assessment designer. Generate exactly 1 numerical/calculation question.

Topic: {topic}
Difficulty: {difficulty}
Context: {context}

Respond ONLY with valid JSON:
{{
  "question_text": "Full question with numbers to calculate...",
  "correct_answer": "42 (numeric value or expression)",
  "explanation": "Step 1: ... Step 2: ... Final answer: ...",
  "hints": ["hint 1"]
}}
[/INST]"""

CODING_PROMPT = """<s>[INST] You are an expert software engineer. Generate a coding problem.

Topic: {topic}
Difficulty: {difficulty}
Language: {language}

Respond ONLY with valid JSON:
{{
  "question_text": "Problem statement with input/output format...",
  "correct_answer": "Complete working {language} solution code here",
  "explanation": "Algorithm explanation and time/space complexity",
  "test_cases": [
    {{"input": "example input", "output": "example output", "visible": true}},
    {{"input": "edge case input", "output": "edge output", "visible": false}}
  ],
  "hints": ["Think about data structure", "Consider edge cases"]
}}
[/INST]"""


# ─── Fallback Template Bank ───────────────────────────────────────────────────
# Used during model warm-up or if generation fails

TEMPLATE_QUESTIONS = {
    QuestionSection.MENTAL_ABILITY: [
        {
            "question_text": "A train travels 120 km in 2 hours. What is its average speed?",
            "format": QuestionFormat.NUMERICAL,
            "options": None,
            "correct_answer": "60 km/h",
            "explanation": "Speed = Distance / Time = 120 / 2 = 60 km/h",
            "topic": "Time Speed Distance",
            "hints": ["Use Speed = Distance / Time"]
        },
        {
            "question_text": "If A is the brother of B, B is the sister of C, what is C's relationship to A?",
            "format": QuestionFormat.MCQ,
            "options": [
                {"label": "A", "text": "Brother"},
                {"label": "B", "text": "Sister"},
                {"label": "C", "text": "Cannot be determined"},
                {"label": "D", "text": "Cousin"}
            ],
            "correct_answer": "C",
            "explanation": "We know A and C are siblings (through B) but C's gender is not stated, so we cannot determine if C is brother or sister.",
            "topic": "Blood Relations",
            "hints": ["Draw a family tree"]
        },
        {
            "question_text": "2, 6, 12, 20, 30, ?",
            "format": QuestionFormat.MCQ,
            "options": [
                {"label": "A", "text": "38"},
                {"label": "B", "text": "40"},
                {"label": "C", "text": "42"},
                {"label": "D", "text": "44"}
            ],
            "correct_answer": "C",
            "explanation": "Differences are 4, 6, 8, 10, 12. Next term = 30 + 12 = 42.",
            "topic": "Number Series",
            "hints": ["Find the pattern in differences"]
        },
    ],
    QuestionSection.TECHNICAL: [
        {
            "question_text": "What is the time complexity of binary search?",
            "format": QuestionFormat.MCQ,
            "options": [
                {"label": "A", "text": "O(n)"},
                {"label": "B", "text": "O(log n)"},
                {"label": "C", "text": "O(n²)"},
                {"label": "D", "text": "O(1)"}
            ],
            "correct_answer": "B",
            "explanation": "Binary search halves the search space each step, giving O(log n) time complexity.",
            "topic": "Searching",
            "hints": ["How many times can you halve n before reaching 1?"]
        },
        {
            "question_text": "In a circuit, two resistors of 4Ω and 6Ω are connected in parallel. What is the equivalent resistance?",
            "format": QuestionFormat.NUMERICAL,
            "options": None,
            "correct_answer": "2.4 Ω",
            "explanation": "1/R = 1/4 + 1/6 = 3/12 + 2/12 = 5/12. R = 12/5 = 2.4 Ω",
            "topic": "Network Analysis",
            "hints": ["1/R_total = 1/R1 + 1/R2 for parallel"]
        },
    ],
    QuestionSection.CODING: [
        {
            "question_text": "Write a function to find the maximum element in an array without using built-in max().\n\nInput: A list of integers\nOutput: The maximum integer\n\nExample:\nInput: [3, 1, 4, 1, 5, 9, 2, 6]\nOutput: 9",
            "format": QuestionFormat.CODING,
            "options": None,
            "correct_answer": """def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for num in arr[1:]:
        if num > max_val:
            max_val = num
    return max_val""",
            "explanation": "Iterate through all elements maintaining a running maximum. Time: O(n), Space: O(1).",
            "topic": "Arrays",
            "test_cases": [
                {"input": "[3,1,4,1,5,9,2,6]", "output": "9", "visible": True},
                {"input": "[-1,-5,-3]", "output": "-1", "visible": True},
                {"input": "[42]", "output": "42", "visible": False},
                {"input": "[0,0,0]", "output": "0", "visible": False}
            ],
            "hints": ["Start with first element as max", "Update max when you find larger element"]
        }
    ]
}


# ─── Core Generator ───────────────────────────────────────────────────────────

class QuestionGenerator:

    def __init__(self):
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = model_loader.get_text_pipeline()
        return self._pipeline

    def _parse_json_from_output(self, raw: str) -> Optional[Dict]:
        """Extract JSON from model output, handling common formatting issues."""
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return None
        json_str = json_match.group(0)
        # Remove trailing commas (common LLM mistake)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {json_str[:200]}")
            return None

    def _build_question(
        self,
        parsed: Dict,
        section: QuestionSection,
        fmt: QuestionFormat,
        topic: str,
        difficulty: DifficultyLevel,
        irt_difficulty: float
    ) -> Question:
        """Convert parsed LLM output to Question object."""
        options = None
        if fmt == QuestionFormat.MCQ and parsed.get("options"):
            options = [MCQOption(**o) for o in parsed["options"]]

        test_cases = None
        hidden_test_cases = None
        if fmt == QuestionFormat.CODING and parsed.get("test_cases"):
            all_tc = parsed["test_cases"]
            test_cases = [tc for tc in all_tc if tc.get("visible", True)]
            hidden_test_cases = [tc for tc in all_tc if not tc.get("visible", True)]

        return Question(
            id=str(uuid.uuid4()),
            section=section,
            format=fmt,
            difficulty=difficulty,
            topic=topic,
            question_text=parsed["question_text"],
            options=options,
            correct_answer=parsed["correct_answer"],
            explanation=parsed.get("explanation", ""),
            hints=parsed.get("hints", []),
            test_cases=test_cases,
            hidden_test_cases=hidden_test_cases,
            time_limit_seconds=self._get_time_limit(fmt, difficulty),
            irt_difficulty=irt_difficulty,
            irt_discrimination=1.0 + (0.5 if difficulty == DifficultyLevel.HARD else 0),
        )

    def _get_time_limit(self, fmt: QuestionFormat, diff: DifficultyLevel) -> int:
        base = {QuestionFormat.MCQ: 90, QuestionFormat.NUMERICAL: 180, QuestionFormat.CODING: 1800}
        multiplier = {DifficultyLevel.EASY: 0.8, DifficultyLevel.MEDIUM: 1.0, DifficultyLevel.HARD: 1.5}
        return int(base[fmt] * multiplier[diff])

    def _irt_difficulty(self, difficulty: DifficultyLevel) -> float:
        return {DifficultyLevel.EASY: -1.0, DifficultyLevel.MEDIUM: 0.0, DifficultyLevel.HARD: 1.5}[difficulty]

    def generate_mcq(self, topic: str, section: QuestionSection,
                     difficulty: DifficultyLevel, context: str = "") -> Optional[Question]:
        try:
            pipe = self._get_pipeline()
            prompt = MCQ_PROMPT.format(
                topic=topic, section=section.value,
                difficulty=difficulty.value, context=context
            )
            output = pipe(prompt)[0]["generated_text"]
            # Strip the prompt from output
            response = output[len(prompt):]
            parsed = self._parse_json_from_output(response)
            if parsed:
                return self._build_question(parsed, section, QuestionFormat.MCQ, topic, difficulty,
                                            self._irt_difficulty(difficulty))
        except Exception as e:
            logger.error(f"MCQ generation failed for {topic}: {e}")
        return None

    def generate_numerical(self, topic: str, section: QuestionSection,
                           difficulty: DifficultyLevel, context: str = "") -> Optional[Question]:
        try:
            pipe = self._get_pipeline()
            prompt = NUMERICAL_PROMPT.format(topic=topic, difficulty=difficulty.value, context=context)
            output = pipe(prompt)[0]["generated_text"]
            response = output[len(prompt):]
            parsed = self._parse_json_from_output(response)
            if parsed:
                return self._build_question(parsed, section, QuestionFormat.NUMERICAL, topic, difficulty,
                                            self._irt_difficulty(difficulty))
        except Exception as e:
            logger.error(f"Numerical generation failed for {topic}: {e}")
        return None

    def generate_coding(self, topic: str, difficulty: DifficultyLevel,
                        language: str = "Python") -> Optional[Question]:
        try:
            pipe = self._get_pipeline()
            prompt = CODING_PROMPT.format(topic=topic, difficulty=difficulty.value, language=language)
            output = pipe(prompt)[0]["generated_text"]
            response = output[len(prompt):]
            parsed = self._parse_json_from_output(response)
            if parsed:
                return self._build_question(parsed, QuestionSection.CODING, QuestionFormat.CODING,
                                            topic, difficulty, self._irt_difficulty(difficulty))
        except Exception as e:
            logger.error(f"Coding generation failed for {topic}: {e}")
        return None

    def get_fallback_question(self, section: QuestionSection,
                              difficulty: DifficultyLevel) -> Question:
        """Return a template question when model is unavailable."""
        import random
        templates = TEMPLATE_QUESTIONS.get(section, TEMPLATE_QUESTIONS[QuestionSection.MENTAL_ABILITY])
        t = random.choice(templates)
        options = None
        if t.get("options"):
            options = [MCQOption(**o) for o in t["options"]]
        return Question(
            id=str(uuid.uuid4()),
            section=section,
            format=t["format"],
            difficulty=difficulty,
            topic=t["topic"],
            question_text=t["question_text"],
            options=options,
            correct_answer=t["correct_answer"],
            explanation=t["explanation"],
            hints=t.get("hints", []),
            test_cases=t.get("test_cases"),
            time_limit_seconds=self._get_time_limit(t["format"], difficulty),
            irt_difficulty=self._irt_difficulty(difficulty),
            irt_discrimination=1.0,
        )


question_generator = QuestionGenerator()
