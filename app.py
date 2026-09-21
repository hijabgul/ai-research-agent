import io
import re
import requests
import streamlit as st

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchAI",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "research_result" not in st.session_state:
    st.session_state["research_result"] = ""

if "research_topic" not in st.session_state:
    st.session_state["research_topic"] = ""


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0B1020;
        color: #F8FAFC;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 17px;
        margin-bottom: 28px;
    }

    .research-box {
        background-color: #111827;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #263244;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    textarea {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        caret-color: #60A5FA !important;
    }

    textarea::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }

    label {
        color: #E2E8F0 !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    .workflow-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        min-height: 130px;
    }

    .workflow-icon {
        font-size: 28px;
        margin-bottom: 7px;
    }

    .workflow-title {
        font-weight: 700;
        color: #F8FAFC;
        font-size: 15px;
    }

    .workflow-description {
        color: #94A3B8;
        font-size: 12px;
        margin-top: 5px;
    }

    .report-container {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 28px;
        margin-top: 20px;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263244;
        padding: 15px;
        border-radius: 12px;
    }

    hr {
        border-color: #263244 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 ResearchAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An evidence-based AI research agent that searches the live web, '
    'reads original sources, cross-checks evidence, and produces a '
    'traceable research report.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GROQ API KEY
# ============================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error(
        "GROQ_API_KEY was not found in Streamlit Secrets."
    )
    st.stop()

groq_api_key = st.secrets["GROQ_API_KEY"]


# ============================================================
# GROQ MODEL
# ============================================================

GROQ_MODEL = "groq/openai/gpt-oss-120b"

crew_llm = LLM(
    model=GROQ_MODEL,
    api_key=groq_api_key,
    temperature=0.2
)


# ============================================================
# WORKFLOW
# ============================================================

st.markdown("### ⚙️ How ResearchAI Works")

workflow = [
    ("📝", "Input", "Your research question"),
    ("🧠", "Plan", "Break the question into research areas"),
    ("🌐", "Search", "Search current web sources"),
    ("📄", "Extract", "Read original webpages"),
    ("🔍", "Cross-check", "Compare important evidence"),
    ("🤖", "Analyze", "Analyze collected evidence"),
    ("📑", "Report", "Generate the final report")
]

columns = st.columns(len(workflow))

for column, item in zip(columns, workflow):

    icon, title, description = item

    with column:

        st.markdown(
            f"""
            <div class="workflow-card">
                <div class="workflow-icon">{icon}</div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-description">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# WEBPAGE EXTRACTION
# ============================================================

def extract_webpage(url, max_chars=12000):

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/130 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        for tag in [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]:

            for element in soup.find_all(tag):
                element.decompose()

        text = " ".join(
            soup.get_text(
                " ",
                strip=True
            ).split()
        )

        return {
            "success": True,
            "url": url,
            "domain": urlparse(url).netloc,
            "text": text[:max_chars]
        }

    except Exception as e:

        return {
            "success": False,
            "url": url,
            "error": str(e)
        }


# ============================================================
# WEB SEARCH
# ============================================================

def research_search(query, max_results=8):

    try:

        search_results = DDGS().text(
            query,
            max_results=max_results
        )

    except Exception as e:

        return [
            {
                "error": f"Search failed: {str(e)}"
            }
        ]

    sources = []

    for result in search_results:

        title = result.get("title", "")
        url = result.get("href", "")
        snippet = result.get("body", "")

        if not url:
            continue

        page = extract_webpage(url)

        sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "accessible": page.get(
                    "success",
                    False
                ),
                "domain": page.get(
                    "domain",
                    ""
                ),
                "content": (
                    page.get("text", "")
                    if page.get("success")
                    else ""
                )
            }
        )

    return sources


# ============================================================
# CREWAI TOOL
# ============================================================

class ResearchSearchInput(BaseModel):

    query: str = Field(
        ...,
        description="The web research query."
    )


class ResearchSearchTool(BaseTool):

    name: str = "web_research_search"

    description: str = (
        "Search the live web using DDGS and read accessible "
        "web pages. Returns original URLs and source content."
    )

    args_schema: type[BaseModel] = ResearchSearchInput

    def _run(self, query: str) -> str:

        sources = research_search(
            query=query,
            max_results=8
        )

        if not sources:
            return "No usable sources were found."

        output = []

        for i, source in enumerate(
            sources,
            1
        ):

            if "error" in source:

                output.append(
                    source["error"]
                )

                continue

            output.append(
                f"""
SOURCE {i}

TITLE:
{source["title"]}

ORIGINAL URL:
{source["url"]}

DOMAIN:
{source["domain"]}

SEARCH SNIPPET:
{source["snippet"]}

SOURCE CONTENT:
{source["content"][:8000]}
"""
            )

        return "\n".join(output)


research_tool = ResearchSearchTool()


# ============================================================
# SINGLE CREWAI AGENT
# ============================================================

research_agent = Agent(

    role="Senior Evidence-Based Research Analyst",

    goal="""
    Conduct detailed, current, evidence-based research on
    the user's research question.

    Search the live web, inspect actual source pages,
    prioritize authoritative and original sources,
    compare evidence, identify conflicting information,
    and produce a detailed research report.

    Never invent sources, URLs, statistics, quotations,
    facts, or evidence.
    """,

    backstory="""
    You are a meticulous professional research analyst.

    You use live web research for current factual information.

    You inspect original source material whenever possible.

    You distinguish documented facts from interpretation.

    You cross-check important claims against multiple sources.

    When sources disagree, clearly explain the disagreement.

    You never manufacture evidence or citations.

    If reliable evidence is insufficient, explicitly say so.
    """,

    tools=[research_tool],

    llm=crew_llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# RESEARCH TASK
# ============================================================

research_task = Task(

    description="""
    Research this question:

    {topic}

    Follow these steps:

    1. Understand the research question.

    2. Break it into useful sub-questions when necessary.

    3. Perform multiple focused web searches.

    4. Use the web_research_search tool.

    5. Prefer primary and authoritative sources.

    6. Read actual webpage content.

    7. For current questions, prioritize recent information.

    8. Cross-check important factual claims.

    9. Identify meaningful disagreements between sources.

    10. Never invent facts, sources, URLs, statistics,
        quotations, or evidence.

    11. Include original source URLs.

    12. Distinguish facts from interpretation.

    13. Explain important limitations.

    The final report MUST contain:

    # Executive Summary

    # Research Question

    # Key Findings

    # Background

    # Detailed Analysis

    # Evidence

    # Conflicting Information

    # Limitations

    # Conclusion

    # Sources

    Important factual claims must be traceable to sources.
    """,

    expected_output="""
    A detailed professional research report containing:

    Executive Summary
    Research Question
    Key Findings
    Background
    Detailed Analysis
    Evidence
    Conflicting Information
    Limitations
    Conclusion
    Sources

    Include original URLs.

    Never fabricate information.
    """,

    agent=research_agent
)


# ============================================================
# CREW
# ============================================================

research_crew = Crew(

    agents=[research_agent],

    tasks=[research_task],

    process=Process.sequential,

    verbose=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("---")

st.markdown("### 📝 What do you want to research?")

st.markdown(
    """
    <p style="color:#94A3B8;">
    Enter a question that requires research. ResearchAI will
    search current web sources, inspect evidence, cross-check
    information, and prepare a detailed report.
    </p>
    """,
    unsafe_allow_html=True
)

topic = st.text_area(
    "Research question",
    placeholder=(
        "Example: What are the major applications of "
        "artificial intelligence in education in 2026?"
    ),
    height=150,
    key="research_input"
)

research_button = st.button(
    "🔎 Start Research",
    type="primary",
    use_container_width=True
)


# ============================================================
# PDF CREATION
# ============================================================

def clean_text(text):

    text = text.replace(
        "**",
        ""
    )

    text = text.replace(
        "__",
        ""
    )

    return text.strip()


def create_pdf(report, topic):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceBefore=16,
        spaceAfter=9,
        textColor=colors.HexColor("#1E3A8A")
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        spaceAfter=9
    )

    story = []

    story.append(
        Paragraph(
            "ResearchAI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Evidence-Based Research Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "<b>Research Question:</b> "
            + clean_text(topic),
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    for raw_line in report.splitlines():

        line = raw_line.strip()

        if not line:
            story.append(
                Spacer(
                    1,
                    5
                )
            )
            continue

        if line.startswith("# "):

            heading = clean_text(
                line[2:]
            )

            story.append(
                Paragraph(
                    heading,
                    heading_style
                )
            )

        elif line.startswith("## "):

            heading = clean_text(
                line[3:]
            )

            story.append(
                Paragraph(
                    heading,
                    heading_style
                )
            )

        elif line.startswith("- ") or line.startswith("* "):

            bullet = clean_text(
                line[2:]
            )

            story.append(
                Paragraph(
                    "• " + bullet,
                    body_style
                )
            )

        else:

            story.append(
                Paragraph(
                    clean_text(line),
                    body_style
                )
            )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:

    if not topic.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        st.session_state["research_result"] = ""
        st.session_state["research_topic"] = topic.strip()

        st.markdown("---")

        st.markdown(
            "### 🔄 Research Progress"
        )

        progress = st.progress(0)

        status = st.empty()

        try:

            # STEP 1
            progress.progress(10)

            status.info(
                "📝 **Step 1/7 — Input received**\n\n"
                "Your research question has been received."
            )

            # STEP 2
            progress.progress(20)

            status.info(
                "🧠 **Step 2/7 — Planning research**\n\n"
                "The AI agent is determining what information "
                "needs to be investigated."
            )

            # STEP 3
            progress.progress(30)

            status.info(
                "🌐 **Step 3/7 — Searching the live web**\n\n"
                "The agent is searching for relevant sources."
            )

            # STEP 4
            progress.progress(40)

            status.info(
                "📄 **Step 4/7 — Reading source pages**\n\n"
                "Accessible webpages are being extracted and analyzed."
            )

            # ACTUAL CREWAI RESEARCH
            with st.spinner(
                "🤖 CrewAI agent is researching and analyzing..."
            ):

                result = research_crew.kickoff(
                    inputs={
                        "topic": topic.strip()
                    }
                )

            # STEP 5
            progress.progress(70)

            status.info(
                "🔍 **Step 5/7 — Cross-checking evidence**\n\n"
                "The agent is comparing important claims and sources."
            )

            # STEP 6
            progress.progress(85)

            status.info(
                "🧠 **Step 6/7 — Building the report**\n\n"
                "The evidence is being organized into a structured report."
            )

            # STEP 7
            progress.progress(100)

            status.success(
                "📑 **Step 7/7 — Report completed**\n\n"
                "Your research report is ready."
            )

            st.session_state["research_result"] = str(result)

        except Exception as e:

            progress.progress(100)

            status.error(
                f"Research failed: {str(e)}"
            )


# ============================================================
# DISPLAY REPORT
# ============================================================

if st.session_state.get("research_result"):

    report = st.session_state["research_result"]

    research_topic = st.session_state["research_topic"]

    st.markdown("---")

    st.markdown("## 📑 Research Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Research Engine",
            "CrewAI"
        )

    with col2:
        st.metric(
            "Web Research",
            "Live"
        )

    with col3:
        st.metric(
            "Research Mode",
            "Evidence-based"
        )

    st.markdown(
        '<div class="report-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        report
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # DOWNLOAD SECTION

    st.markdown("---")

    st.markdown(
        "### 📥 Download Research"
    )

    try:

        pdf_data = create_pdf(
            report,
            research_topic
        )

        safe_name = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            research_topic[:50]
        ).strip("_")

        if not safe_name:
            safe_name = "research_report"

        pdf_filename = (
            safe_name
            + "_ResearchAI.pdf"
        )

        text_filename = (
            safe_name
            + "_ResearchAI.txt"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "📄 Download PDF",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )

        with col2:

            st.download_button(
                "📝 Download Text",
                data=report,
                file_name=text_filename,
                mime="text/plain",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Could not generate PDF: {str(e)}"
        )

    st.info(
        "🔗 The report includes source URLs so that "
        "important claims can be independently checked."
    )
