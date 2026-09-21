import json
import re
import html
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
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


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

        result = {}

        for key, item in value.items():

            if key in {
                "cache_breakpoint",
                "cache_control",
            }:
                continue

            result[key] = clean_litellm_parameters(item)

        return result

    if isinstance(value, list):

        return [
            clean_litellm_parameters(item)
            for item in value
        ]

    return value


def safe_completion(*args, **kwargs):

    kwargs = clean_litellm_parameters(kwargs)

    model = str(kwargs.get("model", ""))

    if "gpt-oss" in model:

        kwargs["reasoning_effort"] = "low"
        kwargs["include_reasoning"] = False

        # Keep the request under the current
        # organization token limit.
        kwargs["max_tokens"] = 1800

    return _original_completion(
        *args,
        **kwargs
    )


litellm.completion = safe_completion


if _original_acompletion is not None:

    async def safe_acompletion(*args, **kwargs):

        kwargs = clean_litellm_parameters(kwargs)

        model = str(kwargs.get("model", ""))

        if "gpt-oss" in model:

            kwargs["reasoning_effort"] = "low"
            kwargs["include_reasoning"] = False
            kwargs["max_tokens"] = 1800

        return await _original_acompletion(
            *args,
            **kwargs
        )

    litellm.acompletion = safe_acompletion


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #07111f;
    }

    .main {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] {
        background: #081321;
    }

    .hero {
        padding: 34px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #10233d,
            #0b1728
        );
        border: 1px solid #1e3a5f;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #94a3b8;
        font-size: 17px;
        line-height: 1.6;
    }

    .section-title {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .small-text {
        color: #94a3b8;
    }

    .source-card {
        padding: 18px;
        border-radius: 14px;
        background: #0c1a2c;
        border: 1px solid #1c304a;
        margin-bottom: 12px;
    }

    .source-title {
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 700;
    }

    .source-domain {
        color: #60a5fa;
        font-size: 13px;
        margin-top: 5px;
    }

    .source-url {
        color: #64748b;
        font-size: 12px;
        margin-top: 6px;
        word-break: break-all;
    }

    .info-card {
        padding: 18px;
        border-radius: 14px;
        background: #0c1a2c;
        border: 1px solid #1c304a;
        min-height: 105px;
    }

    .info-number {
        color: #60a5fa;
        font-size: 25px;
        font-weight: 800;
    }

    .info-title {
        color: #f8fafc;
        font-weight: 700;
    }

    .info-text {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 5px;
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

if "result" not in st.session_state:
    st.session_state.result = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "topic" not in st.session_state:
    st.session_state.topic = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔎 Research Agent")

    st.caption(
        "AI-powered structured web research"
    )

    st.divider()

    st.subheader("Architecture")

    st.write("🤖 One CrewAI Agent")
    st.write("🌐 Live Web Research")
    st.write("📚 Evidence Extraction")
    st.write("📊 Automatic Charts")
    st.write("📋 Automatic Tables")
    st.write("🔗 Source Citations")

    st.divider()

    st.subheader("AI Model")

    st.write("Groq")
    st.write("GPT-OSS 20B")
    st.write("Low reasoning mode")

    st.divider()

    st.caption(
        "The agent does not rely on a fixed answer. "
        "It researches the question and builds the "
        "appropriate report structure."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🔎 AI Research Agent</h1>

        <p>
        Ask a research question and receive a structured,
        source-based report with timelines, tables,
        charts, key findings, and original sources.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WORKFLOW
# ============================================================

cols = st.columns(5)

workflow = [
    ("01", "Question", "Understand the research request"),
    ("02", "Search", "Find current sources"),
    ("03", "Evidence", "Read original pages"),
    ("04", "Analyze", "One CrewAI agent"),
    ("05", "Report", "Table + chart + findings"),
]

for col, item in zip(cols, workflow):

    with col:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-number">
                    {item[0]}
                </div>

                <div class="info-title">
                    {item[1]}
                </div>

                <div class="info-text">
                    {item[2]}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# EXAMPLES
# ============================================================

st.markdown(
    '<div class="section-title">Try an example</div>',
    unsafe_allow_html=True,
)

examples = [
    "Tell me about the Prime Ministers of Pakistan from 1947 until now.",
    "Explain the history of artificial intelligence from its beginning until now.",
    "What are the major developments in renewable energy from 2000 to 2026?",
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
# INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Research Question</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Research Question",
    key="research_question",
    height=140,
    placeholder=(
        "Example: Tell me about the Prime Ministers "
        "of Pakistan from 1947 until now."
    ),
    label_visibility="collapsed",
)


start = st.button(
    "🔎 Start Deep Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# UTILITY FUNCTIONS
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


def search_web(topic):

    queries = [
        f"{topic} 2026",
        f"{topic} official source",
        f"{topic} history timeline evidence",
    ]

    results = []
    seen = set()

    try:

        with DDGS() as ddgs:

            for query in queries:

                try:

                    found = ddgs.text(
                        query,
                        max_results=3,
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
                                "domain": get_domain(url),
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
            f"Search failed: {e}"
        )

    return results


def score_source(source):

    domain = source["domain"]

    score = 0

    if domain.endswith(".gov"):
        score += 20

    if domain.endswith(".edu"):
        score += 15

    if domain.endswith(".org"):
        score += 8

    important_domains = [
        "na.gov.pk",
        "pmo.gov.pk",
        "un.org",
        "who.int",
        "worldbank.org",
        "oecd.org",
        "nih.gov",
        "nasa.gov",
        "nature.com",
        "reuters.com",
        "bbc.com",
    ]

    for domain_name in important_domains:

        if domain_name in domain:
            score += 25

    if len(source["snippet"]) > 100:
        score += 3

    return score


def choose_sources(results):

    ranked = sorted(
        results,
        key=score_source,
        reverse=True,
    )

    return ranked[:5]


def read_webpage(source):

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
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

        return text[:3000]

    except Exception:

        return source["snippet"]


def build_evidence(sources):

    evidence_parts = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        page = read_webpage(source)

        evidence_parts.append(
            f"""
SOURCE {index}

Title:
{source['title']}

Domain:
{source['domain']}

URL:
{source['url']}

Search snippet:
{source['snippet'][:600]}

Page evidence:
{page[:3000]}
""".strip()
        )

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# CREWAI AGENT
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
        temperature=0.4,
    )

    agent = Agent(

        role="Senior Research Analyst",

        goal=(
            "Conduct evidence-based research and "
            "produce structured, useful reports. "
            "Determine what format best answers the "
            "question, including timelines, tables, "
            "charts, comparisons, historical eras, "
            "statistics, or other appropriate structures."
        ),

        backstory=(
            "You are a professional research analyst "
            "who studies supplied web evidence carefully. "
            "You organize information chronologically "
            "when the question is historical, compare "
            "entities when comparison is requested, "
            "extract data when statistics are requested, "
            "and never invent facts or sources."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,

        max_iter=1,
    )

    return agent


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text):

    text = str(text).strip()

    # Remove markdown code fences.
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
    )

    text = text.strip()

    # Direct JSON.
    try:

        return json.loads(text)

    except Exception:
        pass

    # Find JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[
            start:end + 1
        ]

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

def run_agent(question, evidence):

    agent = create_agent()

    task_prompt = f"""
You are answering this research question:

{question}

You have been given web evidence below.

Your job is to create a STRUCTURED research report.

IMPORTANT:

- Use only the supplied evidence.
- Do not invent facts.
- Do not invent dates.
- Do not invent statistics.
- Do not invent quotations.
- Do not invent URLs.
- Prefer primary/official sources.
- If sources disagree, mention the disagreement.
- If evidence is missing, explicitly say that it is missing.
- Do not make political recommendations or rankings.
- For historical questions, organize the answer chronologically.
- For people/office-holder history, include each relevant term.
- For timeline questions, create timeline data.
- For numerical/time-series questions, create chart data.
- For comparison questions, create a comparison table.
- The output must be useful, not a generic essay.

CURRENT DATE:
2026-09-21

RETURN ONLY VALID JSON.

Use EXACTLY this structure:

{{
  "title": "Report title",
  "executive_summary": "Short but informative summary",

  "sections": [
    {{
      "heading": "Section heading",
      "content": "Detailed factual content"
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
      "name": "Person/event",
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "era": "Historical era or period",
      "description": "Short factual description"
    }}
  ],

  "chart": {{
    "type": "timeline",
    "title": "Chart title",
    "x_label": "Time",
    "y_label": "Person/event",
    "data": [
      {{
        "name": "Person/event",
        "start": "YYYY-MM-DD",
        "end": "YYYY-MM-DD"
      }}
    ]
  }},

  "sources": [
    {{
      "title": "Exact source title",
      "url": "Exact URL",
      "why_relevant": "Why this source was used"
    }}
  ]
}}

FORMAT RULES:

For a historical timeline question:

- Fill the timeline completely.
- Include all relevant people/events supported by the evidence.
- Put terms in chronological order.
- Include the historical era.
- Create a table containing the chronological list.
- Create a timeline chart.
- Do not collapse multiple terms of the same person into one term when the source treats them as separate terms.

For ordinary research questions:

- Use only the structures that make sense.
- An empty timeline is acceptable.
- An empty table is acceptable.
- An empty chart is acceptable.

WEB EVIDENCE:

{evidence}
"""

    task = Task(
        description=task_prompt,

        expected_output=(
            "Valid JSON containing a structured "
            "research report, table, timeline, "
            "chart data, and source list."
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
# PDF
# ============================================================

def make_pdf(result):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            html.escape(
                result.get(
                    "title",
                    "Research Report",
                )
            ),
            styles["Title"],
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "<b>Executive Summary</b>",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            html.escape(
                result.get(
                    "executive_summary",
                    "",
                )
            ),
            styles["BodyText"],
        )
    )

    for section in result.get(
        "sections",
        [],
    ):

        story.append(
            Spacer(1, 12)
        )

        story.append(
            Paragraph(
                html.escape(
                    section.get(
                        "heading",
                        "",
                    )
                ),
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    section.get(
                        "content",
                        "",
                    )
                ),
                styles["BodyText"],
            )
        )

    table_data = result.get(
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
            Spacer(1, 15)
        )

        story.append(
            Paragraph(
                html.escape(
                    table_data.get(
                        "title",
                        "Data",
                    )
                ),
                styles["Heading2"],
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
                        colors.HexColor(
                            "#17365D"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
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

        story.append(
            pdf_table
        )

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# RESEARCH EXECUTION
# ============================================================

if start:

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

        st.stop()

    st.session_state.result = None
    st.session_state.sources = []
    st.session_state.topic = question

    with st.status(
        "🔎 Deep research in progress...",
        expanded=True,
    ) as status:

        # ----------------------------------------------
        # SEARCH
        # ----------------------------------------------

        st.write(
            "🌐 Searching current web sources..."
        )

        try:

            results = search_web(
                question
            )

        except Exception as e:

            status.update(
                label="Search failed",
                state="error",
            )

            st.error(
                str(e)
            )

            st.stop()

        if not results:

            status.update(
                label="No sources found",
                state="error",
            )

            st.error(
                "No web sources were found. "
                "Try a more specific question."
            )

            st.stop()

        st.write(
            f"✓ Found {len(results)} web results"
        )

        # ----------------------------------------------
        # SOURCE SELECTION
        # ----------------------------------------------

        st.write(
            "📚 Selecting authoritative sources..."
        )

        selected = choose_sources(
            results
        )

        st.write(
            f"✓ Selected {len(selected)} sources"
        )

        # ----------------------------------------------
        # WEB EXTRACTION
        # ----------------------------------------------

        st.write(
            "📄 Reading original source pages..."
        )

        evidence = build_evidence(
            selected
        )

        st.write(
            "✓ Source evidence collected"
        )

        # ----------------------------------------------
        # AGENT
        # ----------------------------------------------

        st.write(
            "🤖 One CrewAI research agent is "
            "organizing the evidence..."
        )

        try:

            result = run_agent(
                question,
                evidence,
            )

        except Exception as e:

            error = str(e)

            status.update(
                label="Agent failed",
                state="error",
            )

            st.error(
                "The web research worked, but the "
                "CrewAI analysis failed."
            )

            if (
                "413" in error
                or "rate_limit" in error.lower()
                or "tokens per minute" in error.lower()
            ):

                st.warning(
                    "Groq rejected the request because "
                    "of the current token limit. "
                    "Wait a few seconds and try again."
                )

            with st.expander(
                "Technical error"
            ):

                st.code(error)

            st.session_state.sources = selected

            st.stop()

        status.update(
            label="✅ Research completed",
            state="complete",
        )

    st.session_state.result = result
    st.session_state.sources = selected


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.result


if result:

    st.divider()

    st.title(
        result.get(
            "title",
            "Research Report",
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "Executive Summary"
    )

    st.info(
        result.get(
            "executive_summary",
            "",
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    sections = result.get(
        "sections",
        [],
    )

    if sections:

        st.subheader(
            "Research Findings"
        )

        for section in sections:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### {section.get('heading', '')}"
                )

                st.write(
                    section.get(
                        "content",
                        "",
                    )
                )

    # ========================================================
    # CHART
    # ========================================================

    chart = result.get(
        "chart",
        {},
    )

    chart_data = chart.get(
        "data",
        [],
    )

    if chart_data:

        st.subheader(
            "📊 Visual Overview"
        )

        chart_type = chart.get(
            "type",
            "",
        ).lower()

        if chart_type == "timeline":

            df = pd.DataFrame(
                chart_data
            )

            if {
                "name",
                "start",
                "end",
            }.issubset(df.columns):

                df["start"] = pd.to_datetime(
                    df["start"],
                    errors="coerce",
                )

                df["end"] = pd.to_datetime(
                    df["end"],
                    errors="coerce",
                )

                df = df.dropna(
                    subset=[
                        "start",
                        "end",
                    ]
                )

                if not df.empty:

                    fig = px.timeline(
                        df,
                        x_start="start",
                        x_end="end",
                        y="name",
                        title=chart.get(
                            "title",
                            "Timeline",
                        ),
                    )

                    fig.update_yaxes(
                        autorange="reversed"
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=max(
                            500,
                            len(df) * 28,
                        ),
                        margin=dict(
                            l=20,
                            r=20,
                            t=60,
                            b=20,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

        elif chart_type == "line":

            df = pd.DataFrame(
                chart_data
            )

            x = chart.get(
                "x",
                "x",
            )

            y = chart.get(
                "y",
                "y",
            )

            if x in df.columns and y in df.columns:

                fig = px.line(
                    df,
                    x=x,
                    y=y,
                    title=chart.get(
                        "title",
                        "Trend",
                    ),
                    markers=True,
                )

                fig.update_layout(
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        elif chart_type == "bar":

            df = pd.DataFrame(
                chart_data
            )

            x = chart.get(
                "x",
                "x",
            )

            y = chart.get(
                "y",
                "y",
            )

            if x in df.columns and y in df.columns:

                fig = px.bar(
                    df,
                    x=x,
                    y=y,
                    title=chart.get(
                        "title",
                        "Comparison",
                    ),
                )

                fig.update_layout(
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    # ========================================================
    # TABLE
    # ========================================================

    table = result.get(
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

        st.subheader(
            f"📋 {table.get('title', 'Research Data')}"
        )

        dataframe = pd.DataFrame(
            rows,
            columns=columns,
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # TIMELINE
    # ========================================================

    timeline = result.get(
        "timeline",
        [],
    )

    if timeline:

        st.subheader(
            "🕒 Detailed Timeline"
        )

        timeline_df = pd.DataFrame(
            timeline
        )

        st.dataframe(
            timeline_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader(
        "🔗 Sources"
    )

    agent_sources = result.get(
        "sources",
        [],
    )

    if agent_sources:

        for source in agent_sources:

            title = html.escape(
                str(
                    source.get(
                        "title",
                        "Source",
                    )
                )
            )

            url = source.get(
                "url",
                "",
            )

            why = html.escape(
                str(
                    source.get(
                        "why_relevant",
                        "",
                    )
                )
            )

            st.markdown(
                f"**{title}**"
            )

            if url:

                st.markdown(
                    f"[Open original source]({url})"
                )

            if why:

                st.caption(
                    why
                )

            st.divider()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "⬇️ Download"
    )

    text_report = []

    text_report.append(
        result.get(
            "title",
            "Research Report",
        )
    )

    text_report.append("")

    text_report.append(
        "EXECUTIVE SUMMARY"
    )

    text_report.append(
        result.get(
            "executive_summary",
            "",
        )
    )

    for section in sections:

        text_report.append("")
        text_report.append(
            section.get(
                "heading",
                "",
            )
        )
        text_report.append(
            section.get(
                "content",
                "",
            )
        )

    if columns and rows:

        text_report.append("")
        text_report.append(
            table.get(
                "title",
                "Data",
            )
        )

        text_report.append(
            " | ".join(columns)
        )

        for row in rows:

            text_report.append(
                " | ".join(
                    str(x)
                    for x in row
                )
            )

    final_text = "\n".join(
        text_report
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "📄 Download TXT",
            data=final_text,
            file_name="research_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:

        try:

            pdf = make_pdf(
                result
            )

            st.download_button(
                "📕 Download PDF",
                data=pdf,
                file_name="research_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as e:

            st.warning(
                f"PDF could not be created: {e}"
            )
