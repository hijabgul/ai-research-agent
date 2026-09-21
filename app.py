python
import io
import re
import requests
import streamlit as st

from bs4 import BeautifulSoup
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem
)
from xml.sax.saxutils import escape


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
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #94A3B8;
        margin-bottom: 30px;
    }

    .workflow-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px 10px;
        text-align: center;
        min-height: 120px;
    }

    .workflow-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .workflow-title {
        font-size: 15px;
        font-weight: 700;
        color: #F8FAFC;
    }

    .workflow-description {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #F8FAFC;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Research input */
    textarea {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        caret-color: #60A5FA !important;
    }

    textarea::placeholder {
        color: #64748B !important;
    }

    /* Report */
    .report-box {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 25px;
        margin-top: 15px;
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
    """
    <div class="subtitle">
        AI-powered web research using current online sources,
        evidence extraction, cross-checking, and detailed
        report generation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORKFLOW
# ============================================================

st.markdown(
    '<div class="section-title">Research Workflow</div>',
    unsafe_allow_html=True
)

workflow = st.columns(7)

workflow_steps = [
    ("📝", "Input", "Research question"),
    ("🧠", "Plan", "Research strategy"),
    ("🔎", "Search", "Find sources"),
    ("📄", "Extract", "Read webpages"),
    ("⚖️", "Cross-check", "Compare evidence"),
    ("🤖", "Analyze", "AI reasoning"),
    ("📊", "Report", "Final report"),
]

for col, step in zip(workflow, workflow_steps):

    with col:

        st.markdown(
            f"""
            <div class="workflow-card">
                <div class="workflow-icon">{step[0]}</div>
                <div class="workflow-title">{step[1]}</div>
                <div class="workflow-description">
                    {step[2]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# GROQ API KEY
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "GROQ_API_KEY was not found in Streamlit Cloud Secrets."
    )

    st.info(
        "Go to Manage app → Settings → Secrets and make sure "
        "GROQ_API_KEY is configured."
    )

    st.stop()


# ============================================================
# GROQ + CREWAI MODEL CONFIGURATION
# ============================================================

# Groq-hosted model
GROQ_MODEL = "groq/openai/gpt-oss-120b"


# CrewAI LLM
crew_llm = LLM(
    model=GROQ_MODEL,
    api_key=groq_api_key,
    temperature=0.2
)


# ============================================================
# WEBPAGE EXTRACTION
# ============================================================

def extract_webpage(url):

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        # Remove unnecessary page elements
        for element in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form"
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # Remove excessive spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        # Limit webpage size
        return text[:15000]

    except Exception as e:

        return (
            f"Unable to extract webpage content. "
            f"Error: {str(e)}"
        )


# ============================================================
# WEB SEARCH
# ============================================================

def research_search(
    query,
    max_results=8
):

    results = []

    try:

        with DDGS() as ddgs:

            search_results = list(
                ddgs.text(
                    query,
                    max_results=max_results
                )
            )

        for result in search_results:

            title = result.get(
                "title",
                ""
            )

            url = result.get(
                "href",
                ""
            )

            snippet = result.get(
                "body",
                ""
            )

            if not url:
                continue

            content = extract_webpage(url)

            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "content": content
                }
            )

    except Exception as e:

        results.append(
            {
                "title": "Search Error",
                "url": "",
                "snippet": str(e),
                "content": ""
            }
        )

    return results


# ============================================================
# CREWAI SEARCH TOOL
# ============================================================

class ResearchSearchInput(BaseModel):

    query: str = Field(
        ...,
        description="The web research query to search for."
    )


class ResearchSearchTool(BaseTool):

    name: str = "web_research_search"

    description: str = """
    Search the current web for reliable information.

    The tool returns search results, original URLs,
    snippets, and extracted webpage content.
    """

    args_schema: type[BaseModel] = ResearchSearchInput

    def _run(
        self,
        query: str
    ) -> str:

        results = research_search(
            query=query,
            max_results=8
        )

        if not results:

            return "No web results were found."

        output = []

        for index, result in enumerate(
            results,
            start=1
        ):

            output.append(
                f"""
SOURCE {index}

Title:
{result["title"]}

URL:
{result["url"]}

Search snippet:
{result["snippet"]}

Extracted webpage content:
{result["content"][:10000]}

--------------------------------------------------
"""
            )

        return "\n".join(output)


# Create the search tool
research_tool = ResearchSearchTool()


# ============================================================
# ONE CREWAI AGENT
# ============================================================

research_agent = Agent(

    role="Senior Web Research Analyst",

    goal="""
    Conduct accurate and current web research and produce
    a detailed evidence-based research report using reliable
    online sources.
    """,

    backstory="""
    You are an experienced web research analyst.

    You investigate research questions by searching the
    current web, reading original webpages, comparing
    information from multiple sources, identifying
    disagreements, and separating verified facts from
    uncertain information.

    You prioritize:

    - Government sources
    - Official organizations
    - Universities
    - Research institutions
    - Academic publications
    - Original reports
    - Primary sources

    You must never invent:

    - Sources
    - URLs
    - Statistics
    - Quotations
    - Research findings
    - Evidence

    When important claims appear in multiple sources,
    compare those sources instead of simply repeating
    the same claim.

    Every important factual claim should be supported
    by an appropriate source URL.
    """,

    tools=[
        research_tool
    ],

    llm=crew_llm,

    verbose=False,

    allow_delegation=False
)


# ============================================================
# RESEARCH TASK
# ============================================================

research_task = Task(

    description="""
    Research the following topic thoroughly:

    {topic}

    Follow this research process carefully.

    STEP 1 — UNDERSTAND

    Understand the exact research question and identify
    what information the user needs.

    STEP 2 — PLAN

    Break the topic into important subtopics and determine
    what evidence is required.

    STEP 3 — SEARCH

    Search the current web for relevant information.

    Use multiple searches when necessary.

    STEP 4 — SOURCE SELECTION

    Prioritize authoritative and primary sources.

    Prefer official organizations, governments,
    universities, research institutions, academic
    publications, and original reports.

    STEP 5 — READ SOURCES

    Do not rely only on search-result snippets.

    Read the extracted webpage content and use the
    actual source material.

    STEP 6 — CROSS-CHECK

    Compare important claims across multiple sources.

    Identify:

    - Agreement
    - Disagreement
    - Different interpretations
    - Missing information
    - Uncertainty

    STEP 7 — ANALYSIS

    Analyze the evidence carefully.

    Do not present unsupported assumptions as facts.

    STEP 8 — REPORT

    Create a detailed professional research report.

    The report must contain the following sections:

    # Executive Summary

    Provide a detailed overview of the main findings.

    # Introduction

    Explain the topic, background, and why it matters.

    # Key Findings

    Present the major findings and supporting evidence.

    # Detailed Analysis

    Divide the topic into logical subsections and
    explain each one thoroughly.

    # Evidence and Sources

    Explain what the collected sources actually support.

    # Conflicting or Different Evidence

    Clearly identify disagreements or different
    interpretations between credible sources.

    # Limitations

    Explain important limitations, missing information,
    uncertainty, or weaknesses in the available evidence.

    # Conclusion

    Summarize the evidence without introducing
    unsupported claims.

    # Sources

    Provide a numbered list of the original URLs
    used in the research.

    IMPORTANT RULES:

    1. Never invent a source.

    2. Never invent a URL.

    3. Never invent a statistic.

    4. Never invent a quotation.

    5. Never claim that a source supports something
       unless the source actually supports it.

    6. Prefer current information when the topic requires
       current information.

    7. Clearly distinguish factual evidence from
       interpretation.

    8. If reliable evidence is unavailable, explicitly
       state that the evidence is limited.

    9. Include original URLs.

    10. Do not hide uncertainty.
    """,

    expected_output="""
    A detailed and professionally structured research report
    containing:

    - Executive Summary
    - Introduction
    - Key Findings
    - Detailed Analysis
    - Evidence and Sources
    - Conflicting Evidence
    - Limitations
    - Conclusion
    - Original Source URLs
    """,

    agent=research_agent
)


# ============================================================
# CREW
# ============================================================

research_crew = Crew(

    agents=[
        research_agent
    ],

    tasks=[
        research_task
    ],

    process=Process.sequential,

    verbose=False
)


# ============================================================
# USER INPUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    'What would you like to research?'
    '</div>',
    unsafe_allow_html=True
)

topic = st.text_area(

    "Research Topic",

    placeholder=(
        "Example: What are the latest developments "
        "in artificial intelligence education in 2026?"
    ),

    height=140,

    label_visibility="collapsed"
)


# ============================================================
# START RESEARCH BUTTON
# ============================================================

research_button = st.button(
    "🔎 Start Research",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:

    if not topic.strip():

        st.warning(
            "Please enter a research question or topic first."
        )

        st.stop()

    st.session_state["research_topic"] = topic

    with st.status(
        "Starting research...",
        expanded=True
    ) as status:

        # Input
        st.write(
            "📝 Research question received."
        )

        # Planning
        st.write(
            "🧠 Planning the research strategy..."
        )

        # Search
        st.write(
            "🔎 Searching current web sources..."
        )

        # Extraction
        st.write(
            "📄 Extracting information from webpages..."
        )

        # Cross-check
        st.write(
            "⚖️ Cross-checking evidence across sources..."
        )

        # Analysis
        st.write(
            "🤖 AI is analyzing the collected evidence..."
        )

        try:

            result = research_crew.kickoff(
                inputs={
                    "topic": topic
                }
            )

            report = str(result)

            st.session_state["research_result"] = report

            st.write(
                "📊 Preparing the final research report..."
            )

            status.update(
                label="Research completed successfully!",
                state="complete",
                expanded=False
            )

        except Exception as e:

            status.update(
                label="Research failed",
                state="error",
                expanded=True
            )

            st.error(
                "An error occurred while running "
                "the research agent."
            )

            st.exception(e)

            st.stop()


# ============================================================
# DISPLAY REPORT
# ============================================================

if st.session_state["research_result"]:

    st.markdown(
        '<div class="section-title">'
        '📊 Research Report'
        '</div>',
        unsafe_allow_html=True
    )

    report = st.session_state[
        "research_result"
    ]

    st.markdown(
        '<div class="report-box">',
        unsafe_allow_html=True
    )

    st.markdown(report)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PDF CREATION
# ============================================================

def create_pdf(report_text):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "ResearchTitle",

        parent=styles["Title"],

        fontSize=22,

        leading=28,

        alignment=TA_CENTER,

        spaceAfter=20
    )

    heading_style = ParagraphStyle(

        "ResearchHeading",

        parent=styles["Heading2"],

        fontSize=15,

        leading=20,

        spaceBefore=14,

        spaceAfter=8
    )

    body_style = ParagraphStyle(

        "ResearchBody",

        parent=styles["BodyText"],

        fontSize=10,

        leading=16,

        spaceAfter=9
    )

    small_style = ParagraphStyle(

        "ResearchSmall",

        parent=styles["BodyText"],

        fontSize=8,

        leading=11,

        spaceAfter=5
    )

    story = []

    story.append(
        Paragraph(
            "ResearchAI — Research Report",
            title_style
        )
    )

    lines = report_text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:

            story.append(
                Spacer(1, 6)
            )

            continue

        # Main heading
        if line.startswith("# "):

            heading = line[2:].strip()

            story.append(
                Paragraph(
                    escape(heading),
                    title_style
                )
            )

        # Second-level heading
        elif line.startswith("## "):

            heading = line[3:].strip()

            story.append(
                Paragraph(
                    escape(heading),
                    heading_style
                )
            )

        # Third-level heading
        elif line.startswith("### "):

            heading = line[4:].strip()

            story.append(
                Paragraph(
                    escape(heading),
                    heading_style
                )
            )

        # Bullet point
        elif (
            line.startswith("- ")
            or line.startswith("* ")
        ):

            bullet_text = line[2:].strip()

            story.append(
                ListFlowable(

                    [
                        ListItem(
                            Paragraph(
                                escape(bullet_text),
                                body_style
                            )
                        )
                    ],

                    bulletType="bullet",

                    leftIndent=18
                )
            )

        # Numbered source
        elif re.match(
            r"^\d+\.\s+",
            line
        ):

            source_text = re.sub(
                r"^\d+\.\s+",
                "",
                line
            )

            story.append(
                Paragraph(
                    escape(source_text),
                    small_style
                )
            )

        # Normal paragraph
        else:

            safe_text = escape(line)

            story.append(
                Paragraph(
                    safe_text,
                    body_style
                )
            )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# DOWNLOAD REPORT
# ============================================================

if st.session_state["research_result"]:

    st.markdown(
        '<div class="section-title">'
        'Download Report'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # PDF
    with col1:

        try:

            pdf_data = create_pdf(
                st.session_state["research_result"]
            )

            st.download_button(

                label="📄 Download PDF",

                data=pdf_data,

                file_name="research_report.pdf",

                mime="application/pdf",

                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Could not create PDF: {e}"
            )

    # Text
    with col2:

        st.download_button(

            label="📝 Download Text",

            data=st.session_state["research_result"],

            file_name="research_report.txt",

            mime="text/plain",

            use_container_width=True
        )
```
