# ============================================================
# AI RESEARCH AGENT
# One CrewAI Agent + Groq + DDGS + Webpage Reading
# ============================================================

import json
import re
import traceback
from io import BytesIO
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import plotly.express as px

from crewai import Agent, Task, Crew, LLM

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b1220;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263449;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Main text */
    .main-text {
        color: #e5e7eb;
    }

    /* Report container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.4rem;
        border: 1px solid #d9dee7;
    }

    /* Report text */
    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] li {
        color: #111827 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] h1,
    div[data-testid="stVerticalBlockBorderWrapper"] h2,
    div[data-testid="stVerticalBlockBorderWrapper"] h3,
    div[data-testid="stVerticalBlockBorderWrapper"] h4 {
        color: #111827 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Text area */
    textarea {
        border-radius: 10px !important;
    }

    /* Links */
    a {
        color: #2563eb !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263449;
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "research_question" not in st.session_state:
    st.session_state.research_question = ""


# ============================================================
# HEADER
# ============================================================

st.title("🔎 AI Research Agent")
st.caption(
    "Live web research • Source verification • Evidence-based reports • One AI agent"
)

st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research Settings")

    st.markdown(
        """
        **How it works**

        1. You enter a research question.
        2. The system searches the live web.
        3. Relevant sources are selected.
        4. Source pages are read.
        5. One CrewAI agent analyzes the evidence.
        6. A structured report is generated.
        """
    )

    st.divider()

    st.subheader("📚 Research modes")

    st.markdown(
        """
        **General Research**  
        Structured findings and evidence.

        **Historical Research**  
        Chronology, timeline and tables.

        **Comparison**  
        Side-by-side comparison.

        **Risk / Impact**  
        Major issues, evidence and implications.

        **Data / Trends**  
        Numbers, tables and charts when data is available.
        """
    )

    st.divider()

    st.caption("Powered by CrewAI + Groq + DDGS")


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Try a research question")

examples = [
    "Tell me about the Prime Ministers of Pakistan from 1947 until now",
    "How is artificial intelligence dangerous for us?",
    "What are the major causes of climate change?",
    "Compare solar energy and nuclear energy",
]

cols = st.columns(4)

for i, question in enumerate(examples):
    with cols[i]:
        if st.button(
            question,
            key=f"example_{i}",
            width="stretch",
        ):
            st.session_state.research_question = question


# ============================================================
# INPUT
# ============================================================

question = st.text_area(
    "🔍 What do you want to research?",
    value=st.session_state.research_question,
    height=110,
    placeholder="Example: Tell me about the history of artificial intelligence...",
)

research_button = st.button(
    "🚀 Start Research",
    type="primary",
    width="stretch",
)


# ============================================================
# RESEARCH MODE DETECTION
# ============================================================

def detect_research_mode(question):
    q = question.lower()

    historical_words = [
        "history",
        "historical",
        "from 1947",
        "from 1940",
        "from 1950",
        "from 1900",
        "until now",
        "until today",
        "timeline",
        "president",
        "prime minister",
        "prime ministers",
        "rulers",
        "kings",
        "leaders",
        "era",
        "dynasty",
    ]

    comparison_words = [
        "compare",
        "comparison",
        "difference between",
        "versus",
        "vs",
        "better than",
        "similarities",
        "differences",
    ]

    risk_words = [
        "danger",
        "dangerous",
        "risk",
        "risks",
        "harm",
        "harmful",
        "threat",
        "threats",
        "negative effects",
        "problems",
        "disadvantages",
        "impact",
    ]

    data_words = [
        "statistics",
        "statistic",
        "data",
        "trend",
        "trends",
        "growth",
        "numbers",
        "percentage",
        "percent",
        "rate",
        "over time",
    ]

    if any(word in q for word in historical_words):
        return "historical"

    if any(word in q for word in comparison_words):
        return "comparison"

    if any(word in q for word in risk_words):
        return "risk"

    if any(word in q for word in data_words):
        return "data"

    return "general"


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def create_search_queries(question, mode):

    if mode == "historical":
        return [
            question,
            f"{question} official history chronology",
            f"{question} timeline dates official",
            f"{question} historical records",
        ]

    if mode == "risk":
        return [
            question,
            f"{question} scientific evidence research",
            f"{question} risks report",
            f"{question} academic study",
        ]

    if mode == "comparison":
        return [
            question,
            f"{question} comparison research",
            f"{question} advantages disadvantages",
            f"{question} official data",
        ]

    if mode == "data":
        return [
            question,
            f"{question} statistics official data",
            f"{question} dataset report",
            f"{question} trends statistics",
        ]

    return [
        question,
        f"{question} official information",
        f"{question} research report",
        f"{question} evidence study",
    ]


# ============================================================
# SOURCE SCORING
# ============================================================

def score_source(result, question, mode):

    title = str(result.get("title", "")).lower()
    body = str(result.get("body", "")).lower()
    url = str(result.get("href", ""))

    text = f"{title} {body}"

    score = 0

    # Official / academic sources
    official_domains = [
        ".gov",
        ".gov.pk",
        ".edu",
        ".ac.uk",
        ".int",
        "who.int",
        "un.org",
        "worldbank.org",
        "oecd.org",
        "unesco.org",
        "nih.gov",
        "nasa.gov",
        "nature.com",
        "science.org",
    ]

    for domain in official_domains:
        if domain in url.lower():
            score += 8

    # News / established sources
    news_domains = [
        "reuters.com",
        "bbc.com",
        "apnews.com",
        "aljazeera.com",
    ]

    for domain in news_domains:
        if domain in url.lower():
            score += 5

    # Mode-specific terms
    if mode == "risk":
        risk_terms = [
            "risk",
            "safety",
            "harm",
            "threat",
            "privacy",
            "bias",
            "misinformation",
            "security",
            "research",
        ]

        for term in risk_terms:
            if term in text:
                score += 2

        # Penalize generic product pages
        generic_terms = [
            "chatgpt",
            "google ai",
            "gemini",
            "ai products",
            "product overview",
        ]

        for term in generic_terms:
            if term in text:
                score -= 3

    if mode == "historical":
        historical_terms = [
            "history",
            "timeline",
            "chronology",
            "dates",
            "term",
            "served",
            "appointed",
            "elected",
        ]

        for term in historical_terms:
            if term in text:
                score += 2

    if mode == "data":
        data_terms = [
            "statistics",
            "dataset",
            "data",
            "percentage",
            "annual",
            "report",
        ]

        for term in data_terms:
            if term in text:
                score += 2

    # Query relevance
    question_words = set(
        re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
    )

    for word in question_words:
        if word in text:
            score += 0.5

    return score


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(question, mode):

    queries = create_search_queries(question, mode)

    all_results = []

    try:
        with DDGS() as ddgs:

            for query in queries:

                try:
                    results = ddgs.text(
                        query,
                        max_results=5,
                    )

                    for result in results:

                        if not result:
                            continue

                        url = result.get("href", "")

                        if not url:
                            continue

                        all_results.append(
                            {
                                "title": result.get("title", ""),
                                "body": result.get("body", ""),
                                "url": url,
                                "score": score_source(
                                    result,
                                    question,
                                    mode,
                                ),
                            }
                        )

                except Exception:
                    continue

    except Exception as e:
        raise RuntimeError(
            f"Web search could not start: {str(e)}"
        )

    # Remove duplicate URLs
    unique = {}

    for item in all_results:
        unique[item["url"]] = item

    results = list(unique.values())

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:6]


# ============================================================
# READ WEBPAGE
# ============================================================

def read_webpage(url):

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=12,
        )

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        # Remove unnecessary content
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
                "header",
                "form",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text[:3500]

    except Exception:
        return ""


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence(sources):

    evidence = []

    for i, source in enumerate(sources, start=1):

        page_text = read_webpage(
            source["url"]
        )

        # Keep the prompt small because the current
        # Groq organization has a limited TPM.

        page_text = page_text[:1800]

        evidence.append(
            {
                "source_number": i,
                "title": source["title"],
                "url": source["url"],
                "search_summary": source["body"][:700],
                "page_text": page_text,
            }
        )

    return evidence


# ============================================================
# FIND TABLES IN WEBPAGES
# ============================================================

def extract_tables_from_url(url):

    tables = []

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=12,
        )

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        for table in soup.find_all("table"):

            rows = []

            for tr in table.find_all("tr"):

                cells = tr.find_all(
                    ["th", "td"]
                )

                row = [
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                    for cell in cells
                ]

                if row:
                    rows.append(row)

            if len(rows) >= 3:
                tables.append(rows)

    except Exception:
        pass

    return tables


# ============================================================
# HISTORICAL TABLE DETECTION
# ============================================================

def find_useful_historical_table(
    question,
    sources,
):

    candidates = []

    keywords = [
        "prime minister",
        "president",
        "leader",
        "term",
        "date",
        "served",
        "year",
        "period",
    ]

    for source in sources:

        tables = extract_tables_from_url(
            source["url"]
        )

        for table in tables:

            combined = " ".join(
                " ".join(row)
                for row in table[:15]
            ).lower()

            relevance = 0

            for keyword in keywords:
                if keyword in combined:
                    relevance += 1

            if len(table) >= 5:
                relevance += 2

            candidates.append(
                (
                    relevance,
                    table,
                    source,
                )
            )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_score, best_table, best_source = (
        candidates[0]
    )

    if best_score < 3:
        return None, None

    return best_table, best_source


# ============================================================
# CREATE ONE CREWAI AGENT
# ============================================================

def create_agent():

    if "GROQ_API_KEY" not in st.secrets:
        raise RuntimeError(
            "GROQ_API_KEY was not found in Streamlit Secrets."
        )

    groq_api_key = st.secrets[
        "GROQ_API_KEY"
    ]

    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=1200,
    )

    agent = Agent(
        role="Senior Evidence-Based Research Analyst",
        goal=(
            "Research the user's question using only the supplied "
            "web evidence, identify important facts, explain them "
            "clearly, distinguish evidence from uncertainty, and "
            "produce a useful structured research report."
        ),
        backstory=(
            "You are a careful research analyst. You never invent "
            "sources, statistics, dates, quotations, people, or facts. "
            "You use the supplied source evidence and cite the source "
            "numbers that support important claims."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return agent


# ============================================================
# RUN ONE AGENT
# ============================================================

def run_agent(
    question,
    mode,
    evidence,
):

    agent = create_agent()

    compact_evidence = []

    for item in evidence:

        compact_evidence.append(
            {
                "source": item["source_number"],
                "title": item["title"],
                "url": item["url"],
                "evidence": (
                    item["page_text"]
                    or item["search_summary"]
                )[:1800],
            }
        )

    evidence_text = json.dumps(
        compact_evidence,
        ensure_ascii=False,
    )

    task_description = f"""
Research question:
{question}

Research mode:
{mode}

SOURCE EVIDENCE:
{evidence_text}

IMPORTANT RULES:

1. Use only information supported by the supplied evidence.
2. Do not invent facts.
3. Do not invent URLs.
4. Do not invent statistics.
5. Do not invent quotations.
6. If evidence is insufficient, say so.
7. Mention source numbers such as [Source 1] after important claims.
8. Focus on the actual question.
9. Do not talk about the research process.
10. Do not output HTML.
11. Do not output SVG.
12. Do not output Markdown tables.
13. Return valid JSON only.

Return this JSON structure:

{{
  "title": "clear report title",
  "summary": "2-4 sentence executive summary",
  "sections": [
    {{
      "heading": "section heading",
      "content": "well-written paragraphs"
    }}
  ],
  "key_findings": [
    "important finding",
    "important finding"
  ],
  "limitations": "brief evidence limitations"
}}

For historical questions:
- Explain the chronology in the sections.
- Do NOT attempt to create a huge table.
- The application will create tables from source data when possible.

For comparison questions:
- Clearly explain the similarities and differences.

For risk questions:
- Explain the major risks individually.
- Explain why each matters.
- Use evidence from the sources.

For general questions:
- Focus on the most important information rather than generic filler.
"""

    task = Task(
        description=task_description,
        expected_output="Valid JSON containing the research report.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)


# ============================================================
# EXTRACT JSON FROM AGENT RESPONSE
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code fences if the model adds them.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[
            start : end + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {
        "title": "Research Report",
        "summary": text[:1500],
        "sections": [],
        "key_findings": [],
        "limitations": (
            "The research response could not be "
            "converted completely into structured JSON."
        ),
    }


# ============================================================
# CREATE TEXT REPORT
# ============================================================

def create_text_report(
    report,
    sources,
    historical_table=None,
):

    lines = []

    lines.append(
        report.get(
            "title",
            "Research Report",
        )
    )

    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append(
        report.get(
            "summary",
            "",
        )
    )

    lines.append("")

    for section in report.get(
        "sections",
        [],
    ):

        lines.append(
            section.get(
                "heading",
                "Section",
            ).upper()
        )

        lines.append(
            section.get(
                "content",
                "",
            )
        )

        lines.append("")

    findings = report.get(
        "key_findings",
        [],
    )

    if findings:

        lines.append(
            "KEY FINDINGS"
        )

        for finding in findings:
            lines.append(
                f"- {finding}"
            )

        lines.append("")

    if historical_table:

        lines.append(
            "HISTORICAL DATA"
        )

        for row in historical_table:
            lines.append(
                " | ".join(row)
            )

        lines.append("")

    lines.append(
        "LIMITATIONS"
    )

    lines.append(
        report.get(
            "limitations",
            "",
        )
    )

    lines.append("")
    lines.append("SOURCES")

    for i, source in enumerate(
        sources,
        start=1,
    ):

        lines.append(
            f"[Source {i}] "
            f"{source['title']}"
        )

        lines.append(
            source["url"]
        )

    return "\n".join(lines)


# ============================================================
# PDF GENERATION
# ============================================================

def create_pdf(
    report,
    sources,
    historical_table=None,
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
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
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=8,
    )

    story = []

    story.append(
        Paragraph(
            report.get(
                "title",
                "Research Report",
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
                "summary",
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
                    "Section",
                ),
                heading_style,
            )
        )

        content = section.get(
            "content",
            "",
        )

        paragraphs = content.split(
            "\n"
        )

        for paragraph in paragraphs:

            if paragraph.strip():

                story.append(
                    Paragraph(
                        paragraph.strip(),
                        body_style,
                    )
                )

    findings = report.get(
        "key_findings",
        [],
    )

    if findings:

        story.append(
            Paragraph(
                "Key Findings",
                heading_style,
            )
        )

        for finding in findings:

            story.append(
                Paragraph(
                    f"• {finding}",
                    body_style,
                )
            )

    if historical_table:

        story.append(
            Spacer(
                1,
                10,
            )
        )

        story.append(
            Paragraph(
                "Historical Data",
                heading_style,
            )
        )

        table_data = []

        for row in historical_table:

            table_data.append(
                [
                    Paragraph(
                        str(cell),
                        body_style,
                    )
                    for cell in row
                ]
            )

        table = Table(
            table_data,
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#e5e7eb"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
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
                ]
            )
        )

        story.append(table)

    story.append(
        Paragraph(
            "Sources",
            heading_style,
        )
    )

    for i, source in enumerate(
        sources,
        start=1,
    ):

        story.append(
            Paragraph(
                f"[Source {i}] "
                f"{source['title']}",
                body_style,
            )
        )

        story.append(
            Paragraph(
                source["url"],
                body_style,
            )
        )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# DISPLAY REPORT
# ============================================================

def display_report(
    report,
    sources,
    mode,
    historical_table=None,
    historical_source=None,
):

    st.divider()

    st.subheader("📄 Research Report")

    # White report area
    with st.container(border=True):

        st.title(
            report.get(
                "title",
                "Research Report",
            )
        )

        st.caption(
            f"Research type: {mode.title()}"
        )

        st.subheader(
            "Executive Summary"
        )

        st.write(
            report.get(
                "summary",
                "",
            )
        )

        st.divider()

        sections = report.get(
            "sections",
            [],
        )

        for section in sections:

            st.subheader(
                section.get(
                    "heading",
                    "Section",
                )
            )

            st.write(
                section.get(
                    "content",
                    "",
                )
            )

        findings = report.get(
            "key_findings",
            [],
        )

        if findings:

            st.subheader(
                "Key Findings"
            )

            for finding in findings:

                st.markdown(
                    f"- {finding}"
                )

        # Historical table
        if historical_table:

            st.divider()

            st.subheader(
                "📊 Historical Timeline"
            )

            try:

                max_columns = max(
                    len(row)
                    for row in historical_table
                )

                normalized = []

                for row in historical_table:

                    row = list(row)

                    while len(row) < max_columns:
                        row.append("")

                    normalized.append(
                        row[:max_columns]
                    )

                headers = normalized[0]

                data = normalized[1:]

                df = pd.DataFrame(
                    data,
                    columns=headers,
                )

                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True,
                )

                # Create a useful chart when
                # there are enough rows.
                if len(df) >= 5:

                    st.subheader(
                        "📈 Timeline Visualization"
                    )

                    # Find likely name and date columns
                    name_col = None
                    date_cols = []

                    for col in df.columns:

                        col_lower = str(
                            col
                        ).lower()

                        if (
                            name_col is None
                            and any(
                                word in col_lower
                                for word in [
                                    "name",
                                    "prime",
                                    "president",
                                    "leader",
                                    "person",
                                ]
                            )
                        ):
                            name_col = col

                        if any(
                            word in col_lower
                            for word in [
                                "start",
                                "from",
                                "begin",
                            ]
                        ):
                            date_cols.append(col)

                        if any(
                            word in col_lower
                            for word in [
                                "end",
                                "to",
                            ]
                        ):
                            date_cols.append(col)

                    if (
                        name_col
                        and len(date_cols) >= 2
                    ):

                        start_col = date_cols[0]
                        end_col = date_cols[1]

                        chart_df = df.copy()

                        chart_df[
                            "Start"
                        ] = pd.to_datetime(
                            chart_df[start_col],
                            errors="coerce",
                        )

                        chart_df[
                            "End"
                        ] = pd.to_datetime(
                            chart_df[end_col],
                            errors="coerce",
                        )

                        chart_df = chart_df.dropna(
                            subset=[
                                "Start",
                                "End",
                            ]
                        )

                        if not chart_df.empty:

                            fig = px.timeline(
                                chart_df,
                                x_start="Start",
                                x_end="End",
                                y=name_col,
                                title="Historical Timeline",
                            )

                            fig.update_yaxes(
                                autorange="reversed"
                            )

                            st.plotly_chart(
                                fig,
                                width="stretch",
                            )

            except Exception:
                st.info(
                    "The source contains historical data, "
                    "but it could not be converted into a chart."
                )

        st.divider()

        st.subheader(
            "⚠️ Limitations"
        )

        st.write(
            report.get(
                "limitations",
                "No additional limitations were reported.",
            )
        )

    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader("🔗 Sources")

    for i, source in enumerate(
        sources,
        start=1,
    ):

        st.markdown(
            f"**Source {i}: {source['title']}**"
        )

        st.markdown(
            source["url"]
        )

        if source.get("body"):

            with st.expander(
                f"View search summary — Source {i}"
            ):

                st.write(
                    source["body"]
                )

    # ========================================================
    # EXPORT
    # ========================================================

    text_report = create_text_report(
        report,
        sources,
        historical_table,
    )

    pdf_report = create_pdf(
        report,
        sources,
        historical_table,
    )

    st.divider()

    st.subheader(
        "📥 Export Report"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "⬇️ Download TXT",
            data=text_report,
            file_name="research_report.txt",
            mime="text/plain",
            width="stretch",
        )

    with col2:

        st.download_button(
            "📄 Download PDF",
            data=pdf_report,
            file_name="research_report.pdf",
            mime="application/pdf",
            width="stretch",
        )


# ============================================================
# MAIN RESEARCH PROCESS
# ============================================================

if research_button:

    if not question.strip():

        st.warning(
            "Please enter a research question first."
        )

        st.stop()

    st.session_state.research_question = question

    mode = detect_research_mode(
        question
    )

    st.session_state.research_result = None

    # --------------------------------------------------------
    # Step 1: Search
    # --------------------------------------------------------

    with st.status(
        "🔎 Searching the live web...",
        expanded=True,
    ) as status:

        st.write(
            f"Research mode detected: **{mode.title()}**"
        )

        try:

            sources = search_web(
                question,
                mode,
            )

            if not sources:

                status.update(
                    label="No useful sources found",
                    state="error",
                )

                st.error(
                    "The web search did not return useful sources. "
                    "Please try a more specific question."
                )

                st.stop()

            st.write(
                f"Found {len(sources)} candidate sources."
            )

        except Exception as e:

            status.update(
                label="Web search failed",
                state="error",
            )

            st.error(
                f"Web search failed: {str(e)}"
            )

            st.stop()

    # --------------------------------------------------------
    # Step 2: Read sources
    # --------------------------------------------------------

    with st.status(
        "📖 Reading source pages...",
        expanded=True,
    ) as status:

        try:

            evidence = build_evidence(
                sources
            )

            readable = sum(
                1
                for item in evidence
                if item["page_text"]
            )

            st.write(
                f"Successfully read {readable} "
                f"of {len(evidence)} source pages."
            )

        except Exception as e:

            status.update(
                label="Source reading failed",
                state="error",
            )

            st.error(
                f"Could not read source pages: {str(e)}"
            )

            st.stop()

    # --------------------------------------------------------
    # Step 3: Historical structured data
    # --------------------------------------------------------

    historical_table = None
    historical_source = None

    if mode == "historical":

        with st.status(
            "📊 Looking for structured historical data...",
            expanded=True,
        ) as status:

            try:

                (
                    historical_table,
                    historical_source,
                ) = find_useful_historical_table(
                    question,
                    sources,
                )

                if historical_table:

                    st.write(
                        "A structured historical table "
                        "was found in the source."
                    )

                else:

                    st.write(
                        "No suitable structured table was found. "
                        "The report will use the source evidence directly."
                    )

            except Exception:

                historical_table = None
                historical_source = None

    # --------------------------------------------------------
    # Step 4: ONE AI agent
    # --------------------------------------------------------

    with st.status(
        "🤖 Analyzing evidence with the research agent...",
        expanded=True,
    ) as status:

        try:

            raw_result = run_agent(
                question,
                mode,
                evidence,
            )

            report = extract_json(
                raw_result
            )

            status.update(
                label="Research completed",
                state="complete",
            )

        except Exception as e:

            status.update(
                label="Research failed",
                state="error",
            )

            error_text = traceback.format_exc()

            if "413" in error_text:

                st.error(
                    "The Groq request was too large for the "
                    "current organization token limit."
                )

                st.info(
                    "The application already limits the evidence, "
                    "but this particular research question still "
                    "produced a request that was too large."
                )

            elif "cache_breakpoint" in error_text:

                st.error(
                    "Groq/LiteLLM sent an unsupported cache setting."
                )

            else:

                st.error(
                    f"Research failed: {str(e)}"
                )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    error_text,
                    language="text",
                )

            st.stop()

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    st.session_state.research_result = report
    st.session_state.sources = sources

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_report(
        report,
        sources,
        mode,
        historical_table,
        historical_source,
    )


# ============================================================
# DISPLAY PREVIOUS RESULT
# ============================================================

elif st.session_state.research_result:

    report = st.session_state.research_result

    sources = st.session_state.sources

    mode = detect_research_mode(
        st.session_state.research_question
    )

    display_report(
        report,
        sources,
        mode,
    )
