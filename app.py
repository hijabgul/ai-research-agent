
import json
import re
from io import BytesIO
from urllib.parse import urlparse
from html import escape

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
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# ============================================================
# PAGE CONFIG
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

            if key in {
                "cache_breakpoint",
                "cache_control",
            }:
                continue

            cleaned[key] = clean_litellm_parameters(item)

        return cleaned

    if isinstance(value, list):

        return [
            clean_litellm_parameters(item)
            for item in value
        ]

    return value


def safe_completion(*args, **kwargs):

    kwargs = clean_litellm_parameters(kwargs)

    model = str(
        kwargs.get("model", "")
    )

    if "gpt-oss" in model:

        kwargs["reasoning_effort"] = "low"
        kwargs["include_reasoning"] = False

        # Keep request below current Groq TPM pressure.
        kwargs["max_tokens"] = 1600

    return _original_completion(
        *args,
        **kwargs,
    )


litellm.completion = safe_completion


if _original_acompletion is not None:

    async def safe_acompletion(*args, **kwargs):

        kwargs = clean_litellm_parameters(
            kwargs
        )

        model = str(
            kwargs.get("model", "")
        )

        if "gpt-oss" in model:

            kwargs["reasoning_effort"] = "low"
            kwargs["include_reasoning"] = False
            kwargs["max_tokens"] = 1600

        return await _original_acompletion(
            *args,
            **kwargs,
        )

    litellm.acompletion = safe_acompletion


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

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

    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    p,
    label,
    .stMarkdown,
    .stCaption {
        color: #e5e7eb;
    }

    textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    textarea::placeholder {
        color: #94a3b8 !important;
    }

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

    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .main-subtitle {
        color: #aebbd0;
        font-size: 1.08rem;
        line-height: 1.7;
        margin-bottom: 1.8rem;
    }

    .section-label {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 1.8rem;
        margin-bottom: 0.7rem;
    }

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
        "The agent selects an appropriate research format "
        "such as findings, comparisons, timelines, tables, "
        "or charts."
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
# EXAMPLES
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
            width="stretch",
        ):

            st.session_state.research_question = example
            st.rerun()


# ============================================================
# QUESTION
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
    width="stretch",
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        str(text),
    )

    return text.strip()


def get_domain(url):

    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


# ============================================================
# RESEARCH MODE
# ============================================================

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
# SEARCH QUERIES
# ============================================================

def build_search_queries(
    question,
    mode,
):

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

def search_web(
    question,
    mode,
):

    queries = build_search_queries(
        question,
        mode,
    )

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

                        url = (
                            item.get("href")
                            or item.get("url")
                        )

                        if not url:
                            continue

                        if url in seen:
                            continue

                        seen.add(url)

                        results.append(
                            {
                                "title": clean_text(
                                    item.get(
                                        "title",
                                        "Untitled",
                                    )
                                ),
                                "url": url,
                                "domain": get_domain(
                                    url
                                ),
                                "snippet": clean_text(
                                    item.get(
                                        "body",
                                        "",
                                    )
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

def score_source(
    source,
    question,
):

    domain = source["domain"].lower()

    score = 0

    if domain.endswith(".gov"):
        score += 30

    if domain.endswith(".edu"):
        score += 20

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

    if any(
        name in domain
        for name in [
            "reuters.com",
            "bbc.com",
            "apnews.com",
        ]
    ):
        score += 15

    if len(
        source["snippet"]
    ) >= 150:

        score += 5

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
        len(
            question_words.intersection(
                source_words
            )
        ) * 2,
        15,
    )

    return score


def choose_sources(
    results,
    question,
):

    ranked = sorted(
        results,
        key=lambda x: score_source(
            x,
            question,
        ),
        reverse=True,
    )

    # IMPORTANT:
    # Keep only four sources to reduce
    # the Groq request size.
    return ranked[:4]


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

        # IMPORTANT:
        # Reduced from 2200 to 1200 characters.
        return text[:1200]

    except Exception:

        return source["snippet"]


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence(
    sources,
):

    evidence = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        page_text = read_webpage(
            source
        )

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
{source['snippet'][:350]}

PAGE EVIDENCE:
{page_text[:1200]}
""".strip()
        )

    return "\n\n".join(
        evidence
    )


# ============================================================
# CREATE CREWAI AGENT
# ============================================================

def create_agent():

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is missing from "
            "Streamlit Secrets."
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
            "Research the user's question using the supplied "
            "web evidence and produce an accurate, useful, "
            "structured research report."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You evaluate evidence carefully, prefer "
            "authoritative sources, distinguish facts "
            "from interpretations, identify conflicts, "
            "and never invent facts, dates, statistics, "
            "quotes, sources, or URLs."
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

    text = str(
        text
    ).strip()

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

        return json.loads(
            text
        )

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except Exception:
            pass

    raise ValueError(
        "The research agent did not "
        "return valid structured JSON."
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
            f"{i + 1}. "
            f"{s['title']} — "
            f"{s['url']}"
            for i, s in enumerate(
                sources
            )
        ]
    )

    task_prompt = f"""
QUESTION:
{question}

RESEARCH MODE:
{mode}

CURRENT DATE:
2026-09-21

SOURCES:
{source_list}

EVIDENCE:
{evidence}

Create a professional research report.

RULES:

1. Use ONLY the supplied evidence.

2. Never invent facts, dates,
statistics, quotations, people,
organizations, URLs, or claims.

3. If evidence conflicts, explain
the conflict.

4. Prefer authoritative sources.

5. Do not simply summarize sources.
Synthesize the evidence.

6. Use clear professional English.

7. Do not make unsupported claims.

REPORT REQUIREMENTS:

RISKS:
Explain the major risks, why they
matter, how they occur, evidence,
and practical implications.

HISTORICAL:
Provide chronological coverage,
dates or periods, important people
or events, context, and a table.
Use timeline data when supported.

COMPARISON:
Identify criteria, explain both
sides, and create a comparison table.

DATA:
Explain trends, include available
numbers, and create a data table
and chart data when appropriate.

IMPACTS:
Explain major effects and separate
positive and negative effects when
appropriate.

GENERAL:
Provide an executive summary and
organized findings.

RETURN ONLY VALID JSON.

Use exactly:

{{
"title": "Report title",

"executive_summary": "Summary",

"sections": [
    {{
        "heading": "Heading",
        "paragraphs": [
            "Paragraph"
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

"timeline": [],

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
        "url": "Exact URL from supplied sources",
        "why_relevant": "Reason"
    }}
]
}}

CHART TYPES:

Historical date-based:
"type": "timeline"

Numerical time series:
"type": "line"

Categorical numerical:
"type": "bar"

Otherwise:
"type": "none"

Only use source URLs supplied above.
"""

    task = Task(
        description=task_prompt,

        expected_output=(
            "Valid JSON containing a "
            "professional research report."
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

    return extract_json(
        result
    )


# ============================================================
# COPYABLE REPORT
# ============================================================

def build_copyable_report(
    report,
):

    lines = []

    lines.append(
        report.get(
            "title",
            "AI Research Report",
        )
    )

    lines.append("")

    lines.append(
        "EXECUTIVE SUMMARY"
    )

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

        lines.append("")

        lines.append(
            section.get(
                "heading",
                "",
            ).upper()
        )

        lines.append("")

        for paragraph in section.get(
            "paragraphs",
            [],
        ):

            lines.append(
                paragraph
            )

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

    return "\n".join(
        lines
    )


# ============================================================
# PDF
# ============================================================

def create_pdf(
    report,
):

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
            escape(
                report.get(
                    "title",
                    "AI Research Report",
                )
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
            escape(
                report.get(
                    "executive_summary",
                    "",
                )
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
                escape(
                    section.get(
                        "heading",
                        "",
                    )
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
                    escape(
                        paragraph
                    ),
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
                escape(
                    table_data.get(
                        "title",
                        "Summary Table",
                    )
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
            [
                columns
            ]
            + safe_rows,
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
                ]
            )
        )

        story.append(
            pdf_table
        )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# DISPLAY REPORT
# ============================================================

def display_report(
    report,
):

    st.markdown(
        '<div class="section-label">'
        'Research Report'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-paper">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="report-title">'
        f'{escape(str(report.get("title", "AI Research Report")))}'
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

    for section in report.get(
        "sections",
        [],
    ):

        st.markdown(
            '<div class="report-heading">'
            + escape(
                str(
                    section.get(
                        "heading",
                        "",
                    )
                )
            )
            + '</div>',
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
            '<div class="section-label">'
            'Data Table'
            '</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(
            rows,
            columns=columns,
        )

        st.dataframe(
            df,
            width="stretch",
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
                            item.get(
                                "start"
                            )
                        ),
                        "End": pd.to_datetime(
                            item.get(
                                "end"
                            )
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
            )

            st.plotly_chart(
                fig,
                width="stretch",
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
                    width="stretch",
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
                st.caption(
                    why
                )

            if url:
                st.markdown(
                    f"[Open original source]({url})"
                )

            st.divider()

    # --------------------------------------------------------
    # EXPORT
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
            width="stretch",
        )

    with export_cols[1]:

        st.download_button(
            "📑 Download PDF",
            data=pdf_bytes,
            file_name="ai_research_report.pdf",
            mime="application/pdf",
            width="stretch",
        )

    with export_cols[2]:

        st.download_button(
            "📋 Copyable Report",
            data=copyable,
            file_name="copyable_research_report.txt",
            mime="text/plain",
            width="stretch",
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

        with st.status(
            "Researching your question...",
            expanded=True,
        ) as status:

            # STEP 1

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

            # STEP 2

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

            # STEP 3

            st.write(
                "Selecting relevant and authoritative sources..."
            )

            sources = choose_sources(
                results,
                question,
            )

            if not sources:

                raise RuntimeError(
                    "No suitable sources were selected."
                )

            st.session_state.sources = sources

            st.write(
                f"Selected {len(sources)} sources."
            )

            # STEP 4

            st.write(
                "Reading original source pages..."
            )

            evidence = build_evidence(
                sources
            )

            if not evidence.strip():

                raise RuntimeError(
                    "No readable evidence was extracted."
                )

            # STEP 5

            st.write(
                "One CrewAI research agent is analyzing the evidence..."
            )

            report = run_agent(
                question,
                mode,
                evidence,
                sources,
            )

            if not report:

                raise RuntimeError(
                    "The research agent returned an empty report."
                )

            st.session_state.report = report

            status.update(
                label="Research complete",
                state="complete",
                expanded=False,
            )

    except Exception as e:

        # ====================================================
        # IMPORTANT:
        # Show the COMPLETE error directly in Streamlit.
        # ====================================================

        import traceback

        st.error(
            "❌ Research failed"
        )

        st.code(
            traceback.format_exc(),
            language="text",
        )

        st.warning(
            "The full technical error above is shown "
            "so the exact failing component can be identified."
        )

        st.stop()


# ============================================================
# SHOW REPORT
# ============================================================

if st.session_state.report:

    display_report(
        st.session_state.report
    )

