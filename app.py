import json
import re
from io import BytesIO
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import plotly.express as px

import litellm
from crewai import Agent, Task, Crew, Process, LLM

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LITELLM SAFETY PATCH
# ============================================================

_original_completion = litellm.completion
_original_acompletion = getattr(litellm, "acompletion", None)


def clean_litellm_parameters(value):
    if isinstance(value, dict):
        cleaned = {}

        for key, item in value.items():
            if key in {"cache_breakpoint", "cache_control"}:
                continue

            cleaned[key] = clean_litellm_parameters(item)

        return cleaned

    if isinstance(value, list):
        return [clean_litellm_parameters(item) for item in value]

    return value


def safe_completion(*args, **kwargs):
    kwargs = clean_litellm_parameters(kwargs)

    model = str(kwargs.get("model", ""))

    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
        kwargs["include_reasoning"] = False

        # Keep requests small enough for the current Groq limit.
        kwargs["max_tokens"] = 2200

    return _original_completion(*args, **kwargs)


litellm.completion = safe_completion


if _original_acompletion is not None:

    async def safe_acompletion(*args, **kwargs):
        kwargs = clean_litellm_parameters(kwargs)

        model = str(kwargs.get("model", ""))

        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"
            kwargs["include_reasoning"] = False
            kwargs["max_tokens"] = 2200

        return await _original_acompletion(*args, **kwargs)

    litellm.acompletion = safe_acompletion


# ============================================================
# CSS
# IMPORTANT:
# We use CSS only for styling.
# NO visible HTML components are used for the interface.
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background: #0b1220;
    }

    .main {
        background: #0b1220;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }


    /* =========================
       NORMAL TEXT
       ========================= */

    p,
    label,
    .stMarkdown,
    .stText,
    .stCaption {
        color: #e5e7eb;
    }


    /* =========================
       TITLE
       ========================= */

    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    .main-subtitle {
        color: #aebbd0;
        font-size: 1.08rem;
        line-height: 1.7;
        margin-bottom: 1.8rem;
    }


    /* =========================
       SECTION LABELS
       ========================= */

    .section-label {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 1.8rem;
        margin-bottom: 0.7rem;
    }


    /* =========================
       INPUT
       ========================= */

    textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    textarea::placeholder {
        color: #94a3b8 !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    button[kind="primary"] {
        background: #2563eb !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    button[kind="primary"]:hover {
        background: #1d4ed8 !important;
    }

    .stButton button {
        border-radius: 8px !important;
        min-height: 42px;
    }


    /* =========================
       EXAMPLE BUTTONS
       ========================= */

    div[data-testid="stHorizontalBlock"] .stButton button {
        white-space: normal !important;
        height: auto !important;
        min-height: 58px !important;
        line-height: 1.35 !important;
    }


    /* =========================
       REPORT PAPER
       ========================= */

    .report-paper {
        background: #ffffff !important;
        color: #111111 !important;
        border-radius: 6px;
        padding: 45px 55px;
        margin-top: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 35px rgba(0, 0, 0, 0.35);
        border: 1px solid #d7dce3;
    }

    .report-paper h1,
    .report-paper h2,
    .report-paper h3,
    .report-paper h4 {
        color: #111111 !important;
    }

    .report-paper p,
    .report-paper li,
    .report-paper td,
    .report-paper th {
        color: #111111 !important;
    }


    /* =========================
       REPORT TITLE
       ========================= */

    .report-title {
        color: #111111 !important;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 0.5rem;
    }

    .report-date {
        color: #555555 !important;
        font-size: 0.9rem;
        margin-bottom: 1.8rem;
    }


    /* =========================
       REPORT SECTIONS
       ========================= */

    .report-heading {
        color: #111111 !important;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.6rem;
        margin-bottom: 0.65rem;
    }

    .report-paragraph {
        color: #111111 !important;
        font-size: 1rem;
        line-height: 1.8;
        margin-bottom: 0.8rem;
    }


    /* =========================
       SOURCE BOX
       ========================= */

    .source-box {
        background: #f6f7f9 !important;
        border-left: 4px solid #2563eb;
        padding: 14px 17px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .source-box strong {
        color: #111111 !important;
    }

    .source-box span {
        color: #333333 !important;
    }


    /* =========================
       REPORT TABLE
       ========================= */

    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        margin-bottom: 20px;
    }

    .report-table th {
        background: #e9edf3 !important;
        color: #111111 !important;
        border: 1px solid #b8c0cc;
        padding: 9px;
        text-align: left;
    }

    .report-table td {
        background: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #cbd1da;
        padding: 9px;
        vertical-align: top;
    }


    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 9px;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* =========================
       CODE / RAW TEXT PROTECTION
       ========================= */

    code {
        color: #111111;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "research_question" not in st.session_state:
    st.session_state.research_question = ""

if "report" not in st.session_state:
    st.session_state.report = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "research_mode" not in st.session_state:
    st.session_state.research_mode = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔎 AI Research Agent")

    st.caption(
        "Evidence-based web research with one CrewAI agent."
    )

    st.divider()

    st.subheader("Research Pipeline")

    st.write("1. Understand the question")
    st.write("2. Select research strategy")
    st.write("3. Search the live web")
    st.write("4. Read original sources")
    st.write("5. Cross-check evidence")
    st.write("6. Build the report")

    st.divider()

    st.subheader("Technology")

    st.write("🤖 CrewAI")
    st.write("🧠 Groq GPT-OSS 20B")
    st.write("🌐 DuckDuckGo Search")
    st.write("📄 BeautifulSoup")
    st.write("📊 Plotly")
    st.write("📑 ReportLab")

    st.divider()

    st.caption(
        "The agent uses the question to decide whether the "
        "report needs findings, reasons, comparisons, "
        "timelines, tables, or charts."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 AI Research Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-subtitle">
    Ask a research question and receive a complete,
    evidence-based report built from current web sources.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="section-label">Try a research question</div>',
    unsafe_allow_html=True,
)

examples = [
    "How is artificial intelligence dangerous for us? Explain the reasons.",
    "Tell me about the Prime Ministers of Pakistan from 1947 until now.",
    "Compare renewable energy and fossil fuels.",
]

example_cols = st.columns(3)

for i, example in enumerate(examples):

    with example_cols[i]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.research_question = example
            st.rerun()


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    '<div class="section-label">Research Question</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Research Question",
    key="research_question",
    height=125,
    placeholder=(
        "Example: How is artificial intelligence dangerous "
        "for us? Explain the reasons."
    ),
    label_visibility="collapsed",
)


start = st.button(
    "🔎 Start Deep Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text))

    return text.strip()


def get_domain(url):

    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def detect_research_mode(question):

    q = question.lower()

    if any(
        word in q
        for word in [
            "danger",
            "dangerous",
            "risk",
            "risks",
            "harm",
            "harms",
            "harmful",
            "threat",
            "threats",
            "negative impact",
            "problem",
            "problems",
        ]
    ):
        return "risks"

    if any(
        word in q
        for word in [
            "compare",
            "comparison",
            "versus",
            " vs ",
            "difference between",
            "differences between",
        ]
    ):
        return "comparison"

    if any(
        word in q
        for word in [
            "from 19",
            "from 20",
            "from 21",
            "history",
            "historical",
            "timeline",
            "throughout history",
            "until now",
            "from beginning",
        ]
    ):
        return "historical"

    if any(
        word in q
        for word in [
            "percentage",
            "statistics",
            "statistic",
            "data",
            "growth",
            "increase",
            "decrease",
            "trend",
            "rate",
            "over time",
        ]
    ):
        return "data"

    if any(
        word in q
        for word in [
            "effects",
            "impact",
            "impacts",
            "effect on",
            "influence",
            "consequences",
        ]
    ):
        return "impacts"

    return "general"


# ============================================================
# TARGETED SEARCH QUERIES
# ============================================================

def build_search_queries(question, mode):

    if mode == "risks":

        return [
            f"{question} risks evidence",
            f"{question} harms impacts research",
            f"{question} safety risks official",
            f"{question} academic study",
            f"{question} policy report",
        ]

    if mode == "historical":

        return [
            f"{question} official history",
            f"{question} timeline official",
            f"{question} historical record",
            f"{question} encyclopedia history",
            f"{question} archive",
        ]

    if mode == "comparison":

        return [
            f"{question} comparison evidence",
            f"{question} statistics comparison",
            f"{question} official data",
            f"{question} research study",
            f"{question} expert analysis",
        ]

    if mode == "data":

        return [
            f"{question} official statistics",
            f"{question} data 2026",
            f"{question} statistics dataset",
            f"{question} research data",
            f"{question} trend report",
        ]

    if mode == "impacts":

        return [
            f"{question} evidence research",
            f"{question} effects study",
            f"{question} impacts official report",
            f"{question} academic research",
            f"{question} statistics evidence",
        ]

    return [
        f"{question} official information",
        f"{question} research evidence",
        f"{question} academic research",
        f"{question} expert analysis",
        f"{question} report",
    ]


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(question, mode):

    queries = build_search_queries(question, mode)

    results = []
    seen = set()

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:

                    found = ddgs.text(
                        query,
                        max_results=4,
                    )

                    for item in found:

                        url = item.get("href") or item.get("url")

                        if not url:
                            continue

                        if url in seen:
                            continue

                        seen.add(url)

                        results.append(
                            {
                                "title": clean_text(
                                    item.get("title", "Untitled")
                                ),
                                "url": url,
                                "domain": get_domain(url),
                                "snippet": clean_text(
                                    item.get("body", "")
                                ),
                            }
                        )

                except Exception:
                    continue

    except Exception as e:

        raise RuntimeError(
            f"Web search failed: {e}"
        )

    return results


# ============================================================
# SOURCE SCORING
# ============================================================

def score_source(source, question):

    domain = source["domain"].lower()

    score = 0

    # Government
    if domain.endswith(".gov"):
        score += 30

    # Education
    if domain.endswith(".edu"):
        score += 20

    # International organizations
    if any(
        name in domain
        for name in [
            "un.org",
            "who.int",
            "oecd.org",
            "worldbank.org",
            "unesco.org",
        ]
    ):
        score += 30

    # Research institutions
    if any(
        name in domain
        for name in [
            "nih.gov",
            "nasa.gov",
            "nature.com",
            "science.org",
        ]
    ):
        score += 25

    # Major news / reference sources
    if any(
        name in domain
        for name in [
            "reuters.com",
            "bbc.com",
            "apnews.com",
        ]
    ):
        score += 15

    # Search snippet quality
    if len(source["snippet"]) >= 150:
        score += 5

    # Question keywords appearing in title/snippet
    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{5,}\b",
            question.lower(),
        )
    )

    source_words = set(
        re.findall(
            r"\b[a-zA-Z]{5,}\b",
            (
                source["title"]
                + " "
                + source["snippet"]
            ).lower(),
        )
    )

    score += min(
        len(question_words.intersection(source_words)) * 2,
        15,
    )

    return score


def choose_sources(results, question):

    ranked = sorted(
        results,
        key=lambda x: score_source(x, question),
        reverse=True,
    )

    return ranked[:6]


# ============================================================
# READ WEB PAGE
# ============================================================

def read_webpage(source):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            source["url"],
            headers=headers,
            timeout=12,
        )

        if response.status_code != 200:
            return source["snippet"]

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "svg",
                "iframe",
                "noscript",
            ]
        ):
            tag.decompose()

        text = clean_text(
            soup.get_text(" ")
        )

        return text[:2200]

    except Exception:

        return source["snippet"]


# ============================================================
# BUILD COMPACT EVIDENCE
# ============================================================

def build_evidence(sources):

    evidence = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        page_text = read_webpage(source)

        evidence.append(
            f"""
SOURCE {index}

TITLE:
{source['title']}

DOMAIN:
{source['domain']}

URL:
{source['url']}

SEARCH SNIPPET:
{source['snippet'][:450]}

PAGE EVIDENCE:
{page_text[:2200]}
""".strip()
        )

    return "\n\n".join(evidence)


# ============================================================
# CREATE CREWAI AGENT
# ============================================================

def create_agent():

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=api_key,
        api_base="https://api.groq.com/openai/v1",
        temperature=0.3,
    )

    agent = Agent(

        role="Senior Evidence-Based Research Analyst",

        goal=(
            "Research the user's question using only the supplied "
            "web evidence and produce a useful, accurate, structured "
            "research report. Select the appropriate report format "
            "based on the question."
        ),

        backstory=(
            "You are a professional research analyst. You carefully "
            "evaluate evidence, prefer authoritative sources, distinguish "
            "facts from interpretations, identify conflicting evidence, "
            "and never invent facts, dates, statistics, quotations, "
            "sources, or URLs."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,

        max_iter=1,
    )

    return agent


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = str(text).strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```",
        "",
        text,
    ).strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[start : end + 1]

        try:
            return json.loads(candidate)

        except Exception:
            pass

    raise ValueError(
        "The research agent did not return valid structured data."
    )


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    question,
    mode,
    evidence,
    sources,
):

    agent = create_agent()

    source_list = "\n".join(
        [
            f"{i + 1}. {s['title']} — {s['url']}"
            for i, s in enumerate(sources)
        ]
    )

    task_prompt = f"""
USER RESEARCH QUESTION:
{question}

RESEARCH MODE:
{mode}

CURRENT DATE:
2026-09-21

AVAILABLE SOURCES:
{source_list}

WEB EVIDENCE:
{evidence}

YOUR JOB:

Create a complete professional research report answering the user's
question.

IMPORTANT:

1. Use ONLY information supported by the supplied evidence.

2. Never invent:
   - facts
   - dates
   - statistics
   - quotations
   - people
   - organizations
   - URLs
   - source claims

3. If sources disagree, clearly explain the disagreement.

4. Prefer primary and authoritative sources.

5. Do not simply summarize the websites.

6. Synthesize the evidence into a useful explanation.

7. Write in clear professional English.

8. The report must be useful to a student or researcher.

REPORT FORMAT:

For a RISKS question:

- Explain the main risks.
- Give a separate section for each major risk.
- Explain WHY each risk matters.
- Explain the mechanism or reason behind the risk.
- Include evidence where available.
- Include practical implications.
- Do not exaggerate.
- Distinguish documented risks from uncertain/speculative risks.

For a HISTORICAL question:

- Give chronological coverage.
- Do not skip major people/events supported by evidence.
- Include dates or periods.
- Explain the significance/context of each major period.
- Create a chronological table.
- Create timeline data when dates are available.

For a COMPARISON question:

- Identify the comparison criteria.
- Explain each side.
- Create a clear comparison table.
- Do not declare a winner unless the evidence itself establishes a factual outcome.

For a DATA question:

- Explain the trend.
- Include the available numbers.
- Create a data table.
- Create chart data when enough numerical data exists.

For an IMPACT question:

- Explain the major effects.
- Separate positive and negative effects if appropriate.
- Explain evidence and limitations.

For GENERAL research:

- Give an executive summary.
- Give organized findings.
- Use headings and paragraphs.
- Add a table only if it genuinely improves understanding.

RETURN ONLY VALID JSON.

Use exactly this structure:

{{
  "title": "Professional report title",

  "executive_summary": "A concise but informative summary.",

  "sections": [
    {{
      "heading": "Section heading",
      "paragraphs": [
        "Paragraph one.",
        "Paragraph two."
      ]
    }}
  ],

  "table": {{
    "title": "Table title",
    "columns": ["Column 1", "Column 2"],
    "rows": [
      ["Value", "Value"]
    ]
  }},

  "timeline": [
    {{
      "name": "Person or event",
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "era": "Era or period",
      "description": "Short evidence-based description."
    }}
  ],

  "chart": {{
    "type": "none",
    "title": "",
    "x_label": "",
    "y_label": "",
    "data": []
  }},

  "sources": [
    {{
      "title": "Exact source title",
      "url": "Exact URL supplied above",
      "why_relevant": "Why this source supports the report."
    }}
  ]
}}

CHART RULES:

Use:

"type": "timeline"

for historical date-based questions.

Use:

"type": "line"

for numerical time-series data.

Use:

"type": "bar"

for categorical numerical comparisons.

Otherwise use:

"type": "none"

Do NOT create a chart merely for decoration.

TABLE RULES:

If a table is not useful, use:

{{
  "title": "",
  "columns": [],
  "rows": []
}}

TIMELINE RULES:

If a timeline is not appropriate, use:

[]

SOURCE RULE:

Only use URLs that appear in the supplied source list.

Do not create new URLs.
"""

    task = Task(
        description=task_prompt,
        expected_output=(
            "Valid JSON containing a professional research report."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    return extract_json(result)


# ============================================================
# REPORT MARKDOWN
# ============================================================

def build_copyable_report(report):

    lines = []

    lines.append(
        report.get(
            "title",
            "AI Research Report",
        )
    )

    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append("")

    lines.append(
        report.get(
            "executive_summary",
            "",
        )
    )

    for section in report.get(
        "sections",
        [],
    ):

        heading = section.get(
            "heading",
            "",
        )

        lines.append("")
        lines.append(heading.upper())
        lines.append("")

        for paragraph in section.get(
            "paragraphs",
            [],
        ):

            lines.append(paragraph)
            lines.append("")

    table = report.get(
        "table",
        {},
    )

    columns = table.get(
        "columns",
        [],
    )

    rows = table.get(
        "rows",
        [],
    )

    if columns and rows:

        lines.append("")
        lines.append(
            table.get(
                "title",
                "Summary Table",
            ).upper()
        )
        lines.append("")

        lines.append(
            " | ".join(
                str(x)
                for x in columns
            )
        )

        lines.append(
            " | ".join(
                ["---"] * len(columns)
            )
        )

        for row in rows:

            lines.append(
                " | ".join(
                    str(x)
                    for x in row
                )
            )

    return "\n".join(lines)


# ============================================================
# PDF
# ============================================================

def create_pdf(report):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.black,
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        textColor=colors.black,
        spaceAfter=8,
    )

    story = []

    story.append(
        Paragraph(
            report.get(
                "title",
                "AI Research Report",
            ),
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Executive Summary",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            report.get(
                "executive_summary",
                "",
            ),
            body_style,
        )
    )

    for section in report.get(
        "sections",
        [],
    ):

        story.append(
            Paragraph(
                section.get(
                    "heading",
                    "",
                ),
                heading_style,
            )
        )

        for paragraph in section.get(
            "paragraphs",
            [],
        ):

            story.append(
                Paragraph(
                    paragraph,
                    body_style,
                )
            )

    table_data = report.get(
        "table",
        {},
    )

    columns = table_data.get(
        "columns",
        [],
    )

    rows = table_data.get(
        "rows",
        [],
    )

    if columns and rows:

        story.append(
            Paragraph(
                table_data.get(
                    "title",
                    "Summary Table",
                ),
                heading_style,
            )
        )

        safe_rows = [
            [
                str(cell)
                for cell in row
            ]
            for row in rows
        ]

        pdf_table = Table(
            [columns] + safe_rows,
            repeatRows=1,
        )

        pdf_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, -1),
                        colors.black,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (0, 1),
                        (-1, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(pdf_table)

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# DISPLAY REPORT
# ============================================================

def display_report(report):

    st.markdown(
        '<div class="section-label">Research Report</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PAPER CONTAINER
    # --------------------------------------------------------

    st.markdown(
        '<div class="report-paper">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="report-title">'
        f'{report.get("title", "AI Research Report")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-date">'
        'Research generated from current web evidence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-heading">'
        'Executive Summary'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        report.get(
            "executive_summary",
            "",
        )
    )

    # --------------------------------------------------------
    # SECTIONS
    # --------------------------------------------------------

    for section in report.get(
        "sections",
        [],
    ):

        st.markdown(
            f'<div class="report-heading">'
            f'{section.get("heading", "")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        for paragraph in section.get(
            "paragraphs",
            [],
        ):

            st.markdown(
                paragraph
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    table = report.get(
        "table",
        {},
    )

    columns = table.get(
        "columns",
        [],
    )

    rows = table.get(
        "rows",
        [],
    )

    if columns and rows:

        st.markdown(
            '<div class="section-label">Data Table</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(
            rows,
            columns=columns,
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    timeline = report.get(
        "timeline",
        [],
    )

    if timeline:

        valid_timeline = []

        for item in timeline:

            try:

                valid_timeline.append(
                    {
                        "Name": item.get(
                            "name",
                            "",
                        ),
                        "Start": pd.to_datetime(
                            item.get("start")
                        ),
                        "End": pd.to_datetime(
                            item.get("end")
                        ),
                        "Era": item.get(
                            "era",
                            "",
                        ),
                    }
                )

            except Exception:
                continue

        if valid_timeline:

            timeline_df = pd.DataFrame(
                valid_timeline
            )

            st.markdown(
                '<div class="section-label">'
                'Timeline'
                '</div>',
                unsafe_allow_html=True,
            )

            fig = px.timeline(
                timeline_df,
                x_start="Start",
                x_end="End",
                y="Name",
                color="Era",
                hover_name="Name",
                title="Chronological Timeline",
            )

            fig.update_layout(
                template="plotly_white",
                height=max(
                    500,
                    len(valid_timeline) * 30,
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=30,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    chart = report.get(
        "chart",
        {},
    )

    chart_type = chart.get(
        "type",
        "none",
    )

    chart_data = chart.get(
        "data",
        [],
    )

    if (
        chart_type != "none"
        and chart_data
    ):

        st.markdown(
            '<div class="section-label">'
            'Chart'
            '</div>',
            unsafe_allow_html=True,
        )

        chart_df = pd.DataFrame(
            chart_data
        )

        try:

            if chart_type == "line":

                x_column = chart_df.columns[0]
                y_column = chart_df.columns[1]

                fig = px.line(
                    chart_df,
                    x=x_column,
                    y=y_column,
                    title=chart.get(
                        "title",
                        "",
                    ),
                    markers=True,
                )

            elif chart_type == "bar":

                x_column = chart_df.columns[0]
                y_column = chart_df.columns[1]

                fig = px.bar(
                    chart_df,
                    x=x_column,
                    y=y_column,
                    title=chart.get(
                        "title",
                        "",
                    ),
                )

            else:

                fig = None

            if fig:

                fig.update_layout(
                    template="plotly_white",
                    height=500,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    report_sources = report.get(
        "sources",
        [],
    )

    st.markdown(
        '<div class="section-label">'
        'Sources'
        '</div>',
        unsafe_allow_html=True,
    )

    if report_sources:

        for source in report_sources:

            title = source.get(
                "title",
                "Source",
            )

            url = source.get(
                "url",
                "",
            )

            why = source.get(
                "why_relevant",
                "",
            )

            st.markdown(
                f"**{title}**"
            )

            if why:
                st.caption(why)

            if url:
                st.markdown(
                    f"[Open original source]({url})"
                )

            st.divider()

    # --------------------------------------------------------
    # COPY / DOWNLOAD
    # --------------------------------------------------------

    copyable = build_copyable_report(
        report
    )

    pdf_bytes = create_pdf(
        report
    )

    st.markdown(
        '<div class="section-label">'
        'Export Report'
        '</div>',
        unsafe_allow_html=True,
    )

    export_cols = st.columns(3)

    with export_cols[0]:

        st.download_button(
            "📄 Download TXT",
            data=copyable,
            file_name="ai_research_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with export_cols[1]:

        st.download_button(
            "📑 Download PDF",
            data=pdf_bytes,
            file_name="ai_research_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with export_cols[2]:

        st.download_button(
            "📋 Copyable Report",
            data=copyable,
            file_name="copyable_research_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ============================================================
# START RESEARCH
# ============================================================

if start:

    if not question.strip():

        st.warning(
            "Please enter a research question first."
        )

        st.stop()

    st.session_state.report = None
    st.session_state.sources = []
    st.session_state.topic = question

    try:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        with st.status(
            "Researching your question...",
            expanded=True,
        ) as status:

            st.write(
                "Understanding the research question..."
            )

            mode = detect_research_mode(
                question
            )

            st.session_state.research_mode = mode

            st.write(
                f"Research strategy: **{mode.title()}**"
            )

            # ------------------------------------------------
            # STEP 2
            # ------------------------------------------------

            st.write(
                "Searching the live web..."
            )

            results = search_web(
                question,
                mode,
            )

            if not results:

                raise RuntimeError(
                    "No web search results were found."
                )

            st.write(
                f"Found {len(results)} candidate sources."
            )

            # ------------------------------------------------
            # STEP 3
            # ------------------------------------------------

            st.write(
                "Selecting relevant and authoritative sources..."
            )

            sources = choose_sources(
                results,
                question,
            )

            st.session_state.sources = sources

            st.write(
                f"Selected {len(sources)} sources."
            )

            # ------------------------------------------------
            # STEP 4
            # ------------------------------------------------

            st.write(
                "Reading original source pages..."
            )

            evidence = build_evidence(
                sources
            )

            # ------------------------------------------------
            # STEP 5
            # ------------------------------------------------

            st.write(
                "One CrewAI research agent is analyzing the evidence..."
            )

            report = run_agent(
                question,
                mode,
                evidence,
                sources,
            )

            st.session_state.report = report

            status.update(
                label="Research complete",
                state="complete",
                expanded=False,
            )

    except Exception as e:

        st.error(
            f"Research failed: {e}"
        )

        st.info(
            "Open Manage app → Logs in Streamlit Cloud "
            "if you need the detailed error."
        )

        st.stop()


# ============================================================
# SHOW RESULT
# ============================================================

if st.session_state.report:

    display_report(
        st.session_state.report
    )
