"""
JD Analyzer Service
Parses Job Description to extract relevant technical domains and topics.
Uses keyword/NLP matching — no external API needed.
"""

import re
import logging
from typing import List, Dict, Tuple
from backend.models.schemas import JDInput

logger = logging.getLogger(__name__)

# Domain → topics map
DOMAIN_TOPIC_MAP: Dict[str, List[str]] = {
    # Electrical / Electronics
    "power_electronics": [
        "Power Converters", "DC-DC Converters", "Inverters", "Rectifiers",
        "MOSFET", "IGBT", "PWM Control", "Buck Boost Converter"
    ],
    "electrical_machines": [
        "DC Motors", "AC Motors", "Induction Motors", "Transformers",
        "Synchronous Machines", "Generator", "Motor Control"
    ],
    "basic_electrical": [
        "Ohm's Law", "Kirchhoff's Laws", "RC Circuits", "RL Circuits",
        "Thevenin Norton", "Network Analysis", "AC/DC Fundamentals"
    ],
    "vlsi_embedded": [
        "CMOS Design", "Digital Logic", "Microcontrollers", "FPGA",
        "Embedded C", "RTOS", "ARM Architecture", "Interrupts"
    ],
    "signal_processing": [
        "Fourier Transform", "Filters", "Sampling Theorem", "Z-Transform",
        "Convolution", "FFT", "Modulation", "ADC/DAC"
    ],

    # Mechanical
    "thermodynamics": [
        "Laws of Thermodynamics", "Carnot Cycle", "Heat Transfer",
        "Entropy", "Rankine Cycle", "Refrigeration", "Psychrometrics"
    ],
    "strength_of_materials": [
        "Stress and Strain", "Bending Moment", "Shear Force",
        "Deflection", "Columns and Struts", "Torsion", "Fatigue"
    ],
    "fluid_mechanics": [
        "Bernoulli's Equation", "Continuity Equation", "Reynolds Number",
        "Pipe Flow", "Turbomachinery", "Viscosity", "Drag and Lift"
    ],
    "manufacturing": [
        "CNC Machining", "Welding", "Casting", "Tolerances",
        "GD&T", "Lean Manufacturing", "Quality Control", "CAD/CAM"
    ],

    # Civil
    "structural": [
        "Structural Analysis", "Beams", "Trusses", "Frames",
        "RCC Design", "Steel Design", "Concrete", "Load Calculations"
    ],
    "geotechnical": [
        "Soil Mechanics", "Foundation Design", "Bearing Capacity",
        "Settlement", "Consolidation", "Shear Strength", "Retaining Walls"
    ],

    # Software / CS
    "dsa": [
        "Arrays", "Linked Lists", "Trees", "Graphs",
        "Sorting Algorithms", "Searching", "Dynamic Programming", "Recursion"
    ],
    "databases": [
        "SQL", "Normalization", "Indexing", "Transactions",
        "NoSQL", "Query Optimization", "ER Diagrams", "ACID Properties"
    ],
    "os_networking": [
        "Process Management", "Memory Management", "File Systems",
        "TCP/IP", "OSI Model", "Socket Programming", "Deadlocks", "Scheduling"
    ],
    "system_design": [
        "Microservices", "Load Balancing", "Caching", "Message Queues",
        "CAP Theorem", "Distributed Systems", "API Design", "Scalability"
    ],
    "ml_ai": [
        "Machine Learning", "Neural Networks", "Deep Learning",
        "NLP", "Computer Vision", "Model Training", "Evaluation Metrics", "Feature Engineering"
    ],
}

# Keywords to domain mapping
KEYWORD_TO_DOMAIN: Dict[str, str] = {
    # Electrical
    "power": "power_electronics", "converter": "power_electronics",
    "inverter": "power_electronics", "pwm": "power_electronics",
    "mosfet": "power_electronics", "igbt": "power_electronics",
    "motor": "electrical_machines", "transformer": "electrical_machines",
    "generator": "electrical_machines", "induction": "electrical_machines",
    "circuit": "basic_electrical", "electrical": "basic_electrical",
    "ohm": "basic_electrical", "kirchhoff": "basic_electrical",
    "vlsi": "vlsi_embedded", "fpga": "vlsi_embedded", "embedded": "vlsi_embedded",
    "microcontroller": "vlsi_embedded", "rtos": "vlsi_embedded",
    "signal": "signal_processing", "dsp": "signal_processing",
    "filter": "signal_processing", "fourier": "signal_processing",
    # Mechanical
    "thermodynamics": "thermodynamics", "heat": "thermodynamics",
    "stress": "strength_of_materials", "strain": "strength_of_materials",
    "fluid": "fluid_mechanics", "flow": "fluid_mechanics",
    "manufacturing": "manufacturing", "cnc": "manufacturing", "welding": "manufacturing",
    # Civil
    "structural": "structural", "concrete": "structural", "rcc": "structural",
    "soil": "geotechnical", "foundation": "geotechnical", "geotechnical": "geotechnical",
    # Software
    "algorithm": "dsa", "data structure": "dsa", "array": "dsa",
    "sql": "databases", "database": "databases", "nosql": "databases",
    "operating system": "os_networking", "network": "os_networking", "tcp": "os_networking",
    "system design": "system_design", "microservice": "system_design",
    "machine learning": "ml_ai", "deep learning": "ml_ai", "nlp": "ml_ai", "ai": "ml_ai",
}

# Mental ability topics (always included)
MENTAL_ABILITY_TOPICS = [
    "Number Series", "Logical Reasoning", "Analogies",
    "Data Interpretation", "Percentage and Ratio",
    "Time Speed Distance", "Profit and Loss",
    "Probability", "Coding-Decoding", "Blood Relations"
]


def analyze_jd(jd: JDInput) -> Dict:
    """
    Extract technical domains and topics from the JD.
    Returns structured topic plan for question generation.
    """
    text = f"{jd.job_title} {jd.job_description} {' '.join(jd.required_skills or [])}".lower()

    detected_domains = set()
    for keyword, domain in KEYWORD_TO_DOMAIN.items():
        if keyword in text:
            detected_domains.add(domain)

    # Default fallback — always include DSA if programming roles
    if not detected_domains or any(lang in text for lang in ["python", "java", "c++", "code", "software", "developer"]):
        detected_domains.add("dsa")
        detected_domains.add("os_networking")

    # Cap at 3 technical domains to keep assessment focused
    detected_domains = list(detected_domains)[:3]

    technical_topics = []
    for domain in detected_domains:
        topics = DOMAIN_TOPIC_MAP.get(domain, [])
        technical_topics.extend(topics[:4])  # 4 topics per domain

    # Detect programming languages
    languages = jd.programming_languages or []
    if not languages:
        for lang in ["python", "java", "c++", "javascript", "c#"]:
            if lang in text:
                languages.append(lang.capitalize())
        if not languages:
            languages = ["Python"]

    return {
        "detected_domains": detected_domains,
        "technical_topics": technical_topics,
        "mental_ability_topics": MENTAL_ABILITY_TOPICS,
        "programming_languages": languages,
        "job_title": jd.job_title,
        "experience_level": jd.experience_level or "mid",
    }
