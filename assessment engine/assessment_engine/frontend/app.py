"""
GenAI Assessment Engine — Streamlit Frontend
Multi-page adaptive assessment experience.
"""

import streamlit as st
import httpx
import time
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="GenAI Assessment Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styles ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Global */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Page background */
  .stApp { background: #0f1117; color: #e2e8f0; }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1040 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
  }
  .hero-banner h1 { font-size: 2.4rem; font-weight: 700; color: #f8fafc; margin: 0; }
  .hero-banner p  { color: #94a3b8; font-size: 1.05rem; margin-top: 0.5rem; }
  .hero-tag {
    display: inline-block;
    background: #312e81;
    color: #c7d2fe;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
  }

  /* Cards */
  .card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .card-accent {
    border-left: 4px solid #6366f1;
  }

  /* Question card */
  .question-box {
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.8rem;
    margin: 1rem 0;
    font-size: 1.05rem;
    line-height: 1.7;
    color: #e2e8f0;
  }

  /* Difficulty badges */
  .badge-easy   { background: #14532d; color: #86efac; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
  .badge-medium { background: #713f12; color: #fde68a; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
  .badge-hard   { background: #7f1d1d; color: #fca5a5; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
  .badge-section { background: #1e3a5f; color: #93c5fd; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }

  /* Metric tiles */
  .metric-tile {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-tile .value { font-size: 2rem; font-weight: 700; color: #818cf8; }
  .metric-tile .label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; }

  /* Code block */
  .code-area {
    font-family: 'JetBrains Mono', monospace;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c9d1d9;
    white-space: pre;
    overflow-x: auto;
  }

  /* Progress bar override */
  .stProgress > div > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

  /* Correct / wrong feedback */
  .feedback-correct { background: #14532d; border: 1px solid #16a34a; border-radius: 8px; padding: 0.8rem 1.2rem; color: #86efac; }
  .feedback-wrong   { background: #450a0a; border: 1px solid #b91c1c; border-radius: 8px; padding: 0.8rem 1.2rem; color: #fca5a5; }

  /* Sidebar styling */
  .css-1d391kg { background: #0f1117; }

  /* Ability level pill */
  .ability-advanced   { background: #312e81; color: #c7d2fe; border-radius: 20px; padding: 4px 14px; font-weight: 600; }
  .ability-proficient { background: #064e3b; color: #6ee7b7; border-radius: 20px; padding: 4px 14px; font-weight: 600; }
  .ability-developing { background: #78350f; color: #fde68a; border-radius: 20px; padding: 4px 14px; font-weight: 600; }
  .ability-beginner   { background: #450a0a; color: #fca5a5; border-radius: 20px; padding: 4px 14px; font-weight: 600; }

  button[kind="primary"] { background: #6366f1 !important; border: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── State Helpers ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "page": "home",          # home | setup | assessment | report
        "session_id": None,
        "current_question": None,
        "progress": {},
        "question_start_time": None,
        "feedback": None,        # {"correct": bool, "explanation": str, "correct_answer": str}
        "report": None,
        "assessment_meta": None,
        "theta_history": [],
        "difficulty_history": [],
        "answer_submitted": False,
        "selected_option": None,
        "code_answer": "",
        "numerical_answer": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── API Helpers ──────────────────────────────────────────────────────────────

def api_call(method: str, path: str, **kwargs):
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = getattr(client, method)(f"{BACKEND_URL}{path}", **kwargs)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to backend. Make sure `run_backend.py` is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def check_backend():
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{BACKEND_URL}/health")
            return resp.status_code == 200
    except:
        return False


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 Assessment Engine")
        st.markdown("---")

        backend_ok = check_backend()
        if backend_ok:
            st.markdown("🟢 **Backend:** Online")
        else:
            st.markdown("🔴 **Backend:** Offline")
            st.caption("Run `python run_backend.py` to start")

        st.markdown("---")

        if st.session_state.page == "assessment" and st.session_state.progress:
            p = st.session_state.progress
            answered = p.get("answered", 0)
            total = p.get("total", 1)
            theta = p.get("current_theta", 0.0)
            diff = p.get("current_difficulty", "medium")

            st.markdown("**📊 Live Stats**")
            pct = answered / total
            st.progress(int(pct) * 100)
            st.caption(f"{answered + 1}/{total} questions answered")

            st.markdown(f"**θ (Ability):** `{theta:.3f}`")

            diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(diff, "⚪")
            st.markdown(f"**Next Difficulty:** {diff_color} {diff.capitalize()}")

            if st.session_state.theta_history:
                theta_df = pd.DataFrame({
                    "Q#": range(1, len(st.session_state.theta_history) + 1),
                    "θ": st.session_state.theta_history
                })
                fig = px.line(theta_df, x="Q#", y="θ", height=160,
                              color_discrete_sequence=["#6366f1"])
                fig.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="#94a3b8", margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    showlegend=False
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#475569")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        if st.session_state.page != "home":
            if st.button("🏠 Home", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                init_state()
                st.rerun()

        st.markdown("---")
        st.caption("Powered by HuggingFace Transformers\nMistral-7B · MiniLM · CodeGen")


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_home():
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-tag">GENAI · ADAPTIVE · ASSESSMENT</div>
      <h1>🧠 Intelligent Assessment Engine</h1>
      <p>AI-powered adaptive testing — personalized for every candidate, every role</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="card card-accent">
        <h4>🎯 JD-Driven Generation</h4>
        <p style="color:#94a3b8;font-size:0.9rem">Paste any Job Description and the engine auto-extracts domains, topics, and difficulty profiles.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card card-accent">
        <h4>🔄 Computer Adaptive Testing</h4>
        <p style="color:#94a3b8;font-size:0.9rem">2-Parameter IRT model continuously estimates candidate ability and adjusts question difficulty in real-time.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="card card-accent">
        <h4>🚫 Zero Repetition</h4>
        <p style="color:#94a3b8;font-size:0.9rem">FAISS + MiniLM semantic embeddings detect and filter duplicate questions across all test iterations.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Covers Every Assessment Layer")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card">
        <h5>🧩 Mental Ability</h5>
        <ul style="color:#94a3b8;font-size:0.88rem">
        <li>Logical Reasoning</li><li>Quantitative Aptitude</li>
        <li>Number Series</li><li>Data Interpretation</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
        <h5>⚡ Technical Aptitude</h5>
        <ul style="color:#94a3b8;font-size:0.88rem">
        <li>Domain-specific (Electrical, Mechanical, Civil, CS)</li>
        <li>Auto-detected from JD</li>
        <li>MCQ + Numerical formats</li>
        </ul></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card">
        <h5>💻 Coding Problems</h5>
        <ul style="color:#94a3b8;font-size:0.88rem">
        <li>Python / Java / C++</li>
        <li>Visible + Hidden Test Cases</li>
        <li>Step-by-step solutions</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🚀 Create Assessment", type="primary", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()


def page_setup():
    st.markdown("## 📋 Configure Assessment")
    st.markdown("Paste the Job Description below — the engine will auto-detect technical domains.")

    with st.form("assessment_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            job_title = st.text_input("Job Title *", placeholder="e.g. Power Electronics Engineer")
            job_description = st.text_area(
                "Job Description *", height=220,
                placeholder="Paste the full JD here — required skills, responsibilities, domain keywords..."
            )
            required_skills = st.text_input(
                "Key Skills (optional, comma-separated)",
                placeholder="e.g. Power Electronics, MATLAB, Embedded C, Python"
            )

        with col2:
            st.markdown("**Assessment Settings**")
            experience = st.selectbox("Experience Level", ["fresher", "junior", "mid", "senior"], index=2)
            prog_langs = st.multiselect(
                "Coding Languages",
                ["Python", "Java", "C++", "C", "JavaScript"],
                default=["Python"]
            )
            st.markdown("---")
            mental_count = st.slider("Mental Ability Questions", 5, 20, 10)
            tech_count = st.slider("Technical Questions", 5, 20, 10)
            coding_count = st.slider("Coding Problems", 1, 5, 3)

        submitted = st.form_submit_button("⚡ Generate Assessment", type="primary", use_container_width=True)

    if submitted:
        if not job_title or not job_description:
            st.error("Job Title and Job Description are required.")
            return

        skills_list = [s.strip() for s in required_skills.split(",") if s.strip()] if required_skills else []

        payload = {
            "jd": {
                "job_title": job_title,
                "job_description": job_description,
                "required_skills": skills_list,
                "programming_languages": prog_langs or ["Python"],
                "experience_level": experience
            },
            "mental_ability_count": mental_count,
            "technical_count": tech_count,
            "coding_count": coding_count,
            "start_difficulty": "medium"
        }

        with st.spinner("🤖 Generating questions using Mistral-7B... This may take a few minutes on first run (model download + generation)."):
            result = api_call("post", "/api/assessment/create", json=payload)

        if result and result.get("status") == "success":
            data = result["data"]
            st.session_state.session_id = data["session_id"]
            st.session_state.assessment_meta = data
            st.success(f"✅ Assessment created! {data['total_questions']} unique questions generated.")

            # Show detected domains
            domains = data["jd_analysis"].get("detected_domains", [])
            if domains:
                st.info(f"📌 Detected domains: **{', '.join(d.replace('_', ' ').title() for d in domains)}**")

            time.sleep(1.5)
            st.session_state.page = "assessment"
            st.rerun()


def page_assessment():
    if not st.session_state.session_id:
        st.error("No active session. Please create an assessment first.")
        st.session_state.page = "home"
        st.rerun()
        return

    # Load next question if not loaded
    if st.session_state.current_question is None and not st.session_state.answer_submitted:
        result = api_call("get", f"/api/assessment/{st.session_state.session_id}/next-question")
        if result:
            if result["status"] == "complete":
                st.session_state.page = "report"
                st.rerun()
                return
            st.session_state.current_question = result["question"]
            st.session_state.progress = result["progress"]
            st.session_state.question_start_time = time.time()
            st.session_state.feedback = None
            st.session_state.selected_option = None
            st.session_state.code_answer = ""
            st.session_state.numerical_answer = ""

    q = st.session_state.current_question
    if not q:
        return

    progress = st.session_state.progress
    answered = progress.get("answered", 0)
    total = progress.get("total", 1)

    # ── Header row
    col_prog, col_diff = st.columns([3, 1])
    with col_prog:
        st.progress(answered / total, text=f"Question {answered + 1} of {total}")
    with col_diff:
        diff = q.get("difficulty", "medium")
        section = q.get("section", "").replace("_", " ").title()
        st.markdown(
            f'<span class="badge-section">{section}</span> '
            f'<span class="badge-{diff}">{diff.capitalize()}</span>',
            unsafe_allow_html=True
        )

    st.markdown(f"**Topic:** `{q.get('topic', '')}`")

    # ── Question text
    st.markdown(f'<div class="question-box">{q["question_text"]}</div>', unsafe_allow_html=True)

    # ── Answer input
    fmt = q.get("format", "mcq")

    if not st.session_state.answer_submitted:
        if fmt == "mcq" and q.get("options"):
            options = q["options"]
            labels = [f"{o['label']}. {o['text']}" for o in options]
            selected = st.radio("Select your answer:", labels, key=f"radio_{q['id']}")
            st.session_state.selected_option = selected[0] if selected else None

        elif fmt == "numerical":
            val = st.text_input("Your answer (numeric):", key=f"num_{q['id']}")
            st.session_state.numerical_answer = val

        elif fmt == "coding":
            if q.get("test_cases"):
                with st.expander("📋 Example Test Cases"):
                    for i, tc in enumerate(q["test_cases"]):
                        st.markdown(f"**Input:** `{tc['input']}`  →  **Output:** `{tc['output']}`")

            hints = q.get("hints", [])
            if hints:
                with st.expander("💡 Hints"):
                    for h in hints:
                        st.markdown(f"- {h}")

            code = st.text_area("Write your solution:", height=250,
                                placeholder="# Write your code here...",
                                key=f"code_{q['id']}")
            st.session_state.code_answer = code

        col_sub, col_skip = st.columns([2, 1])
        with col_sub:
            if st.button("✅ Submit Answer", type="primary", use_container_width=True):
                _submit_answer(q, fmt)
        with col_skip:
            if st.button("⏭️ Skip", use_container_width=True):
                _submit_answer(q, fmt, skip=True)

    # ── Feedback panel
    if st.session_state.answer_submitted and st.session_state.feedback:
        fb = st.session_state.feedback
        if fb["correct"]:
            st.markdown('<div class="feedback-correct">✅ <strong>Correct!</strong></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="feedback-wrong">❌ <strong>Incorrect.</strong> '
                f'Correct answer: <code>{fb["correct_answer"]}</code></div>',
                unsafe_allow_html=True)

        with st.expander("📖 Explanation", expanded=True):
            st.markdown(fb["explanation"])

        if fmt == "coding":
            st.markdown("**✅ Reference Solution:**")
            st.markdown(f'<div class="code-area">{fb["correct_answer"]}</div>',
                        unsafe_allow_html=True)

        st.markdown(f"🧭 **Updated ability estimate (θ):** `{fb['updated_theta']:.3f}`")

        if st.button("➡️ Next Question", type="primary"):
            st.session_state.current_question = None
            st.session_state.answer_submitted = False
            st.session_state.feedback = None
            st.rerun()


def _submit_answer(q, fmt: str, skip: bool = False):
    if skip:
        answer = "SKIP"
    elif fmt == "mcq":
        answer = st.session_state.selected_option or "A"
    elif fmt == "numerical":
        answer = st.session_state.numerical_answer or "0"
    else:
        answer = st.session_state.code_answer or ""

    elapsed = int(time.time() - (st.session_state.question_start_time or time.time()))

    payload = {
        "session_id": st.session_state.session_id,
        "question_id": q["id"],
        "answer": answer,
        "time_taken_seconds": elapsed
    }

    result = api_call("post", f"/api/assessment/{st.session_state.session_id}/submit-answer",
                      json=payload)

    if result:
        st.session_state.feedback = {
            "correct": result["is_correct"],
            "correct_answer": result["correct_answer"],
            "explanation": result["explanation"],
            "updated_theta": result["updated_theta"],
        }
        st.session_state.theta_history.append(result["updated_theta"])
        st.session_state.difficulty_history.append(result["next_difficulty"])
        st.session_state.answer_submitted = True

        if result.get("is_assessment_complete"):
            _finalize_report()

        st.rerun()


def _finalize_report():
    result = api_call("get", f"/api/assessment/{st.session_state.session_id}/report")
    if result:
        st.session_state.report = result["report"]


def page_report():
    # Fetch report if not loaded
    if not st.session_state.report:
        with st.spinner("Generating report..."):
            _finalize_report()
        if not st.session_state.report:
            st.error("Could not load report.")
            return

    r = st.session_state.report

    # ── Header
    ability = r["ability_level"]
    ability_class = f"ability-{ability.lower()}"
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-tag">ASSESSMENT COMPLETE</div>
      <h1>📊 Your Results</h1>
      <p>Ability Level: <span class="{ability_class}">{ability}</span></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics
    cols = st.columns(5)
    metrics = [
        ("θ Score", f"{r['candidate_theta']:.3f}", "IRT Ability"),
        ("Accuracy", f"{r['overall_accuracy']*100:.1f}%", "Overall"),
        ("Correct", f"{r['total_correct']}/{r['total_questions']}", "Questions"),
        ("Time", f"{r['total_time_minutes']:.1f} min", "Total"),
        ("Level", ability, "Assessment"),
    ]
    for col, (val, sub, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""<div class="metric-tile">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Ability Progression (θ)")
        if st.session_state.theta_history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.theta_history,
                x=list(range(1, len(st.session_state.theta_history) + 1)),
                mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=6),
                name="Theta"
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#475569",
                          annotation_text="Baseline", annotation_font_color="#64748b")
            fig.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#94a3b8", height=280,
                xaxis_title="Question #", yaxis_title="θ",
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#334155"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.markdown("#### Section Performance")
        sec_reports = r["section_reports"]
        if sec_reports:
            fig = go.Figure()
            sections = [s["section"].replace("_", " ").title() for s in sec_reports]
            accuracies = [s["accuracy"] * 100 for s in sec_reports]
            colors = ["#6366f1" if a >= 70 else "#f59e0b" if a >= 50 else "#ef4444" for a in accuracies]
            fig.add_trace(go.Bar(x=sections, y=accuracies, marker_color=colors,
                                 text=[f"{a:.1f}%" for a in accuracies], textposition="outside"))
            fig.add_hline(y=70, line_dash="dot", line_color="#22c55e",
                          annotation_text="70% target", annotation_font_color="#22c55e")
            fig.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#94a3b8", height=280,
                yaxis=dict(range=[0, 110], gridcolor="#334155"),
                xaxis=dict(showgrid=False),
                margin=dict(l=10, r=10, t=10, b=30),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Strengths & Improvements
    col_s, col_i = st.columns(2)
    with col_s:
        st.markdown("#### ✅ Strengths")
        for s in r.get("strengths", []) or ["Keep it up!"]:
            st.markdown(f"- {s}")
    with col_i:
        st.markdown("#### 📈 Areas to Improve")
        for s in r.get("improvement_areas", []) or ["Review your weak sections."]:
            st.markdown(f"- {s}")

    # ── Recommendations
    st.markdown(f"""<div class="card card-accent">
    <h4>🎯 Recommendations</h4>
    <p style="color:#e2e8f0">{r['recommendations']}</p>
    </div>""", unsafe_allow_html=True)

    # ── Difficulty distribution
    st.markdown("#### Difficulty Distribution (Adaptive Path)")
    if st.session_state.difficulty_history:
        diff_counts = {"easy": 0, "medium": 0, "hard": 0}
        for d in st.session_state.difficulty_history:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        fig = go.Figure(go.Pie(
            labels=list(diff_counts.keys()),
            values=list(diff_counts.values()),
            marker=dict(colors=["#22c55e", "#f59e0b", "#ef4444"]),
            hole=0.4
        ))
        fig.update_layout(
            paper_bgcolor="#1e293b", font_color="#94a3b8",
            height=250, margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(bgcolor="#1e293b", font=dict(color="#94a3b8"))
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Detailed Q&A review
    with st.expander("📝 Detailed Question Review"):
        for i, qr in enumerate(r.get("question_results", []), 1):
            icon = "✅" if qr["is_correct"] else "❌"
            diff_badge = f'<span class="badge-{qr["difficulty"]}">{qr["difficulty"].capitalize()}</span>'
            st.markdown(
                f"**{i}. {icon} {qr['question_text'][:120]}...** {diff_badge}",
                unsafe_allow_html=True
            )
            if not qr["is_correct"]:
                st.caption(f"Your answer: `{qr['candidate_answer']}` | Correct: `{qr['correct_answer']}`")
            st.caption(f"Section: {qr['section']} | Topic: {qr['topic']} | Time: {qr['time_taken_seconds']}s")
            st.markdown("---")

    # ── Actions
    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        if st.button("🔁 New Assessment", type="primary", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_state()
            st.session_state.page = "setup"
            st.rerun()
    with col_b:
        report_json = json.dumps(r, indent=2, default=str)
        st.download_button(
            "⬇️ Download Report",
            data=report_json,
            file_name=f"assessment_report_{r['session_id'][:8]}.json",
            mime="application/json",
            use_container_width=True
        )


# ─── Router ───────────────────────────────────────────────────────────────────

render_sidebar()

page = st.session_state.page

if page == "home":
    page_home()
elif page == "setup":
    page_setup()
elif page == "assessment":
    page_assessment()
elif page == "report":
    page_report()
