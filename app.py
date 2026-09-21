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
PageBreak,
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
""" <style>

```
/* Main application */
.stApp {
    background-color: #0B1020;
    color: #F8FAFC;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Header */
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

/* Research input card */
.research-box {
    background: #111827;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #263244;
    margin-top: 20px;
    margin-bottom: 25px;
}

/* Text area */
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

/* Text area label */
label {
    color: #E2E8F0 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

/* Workflow cards */
.workflow-card {
    background: #111827;
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

/* Report */
.report-container {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
}

/* Status */
.status-box {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 14px;
    padding: 20px;
    margin-top: 20px;
}

/* Source cards */
.source-card {
    background: #0F172A;
    border: 1px solid #263244;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}

.source-title {
    color: #F8FAFC;
    font-weight: 700;
}

.source-domain {
    color: #60A5FA;
    font-size: 13px;
}

/* Divider */
hr {
    border-color: #263244 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #263244;
    padding: 15px;
    border-radius: 12px;
}

</style>
""",
unsafe_allow_html=True
```

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

# The provider is explicitly set to Groq.

GROQ_MODEL = "groq/openai/gpt-oss-120b"

crew_llm = LLM(
model=GROQ_MODEL,
api_key=groq_api_key,
temperature=0.2
)

# ============================================================

# WORKFLOW DISPLAY

# ============================================================

st.markdown("### ⚙️ How ResearchAI Works")

workflow = [
("📝", "Input", "Your research question"),
("🧠", "Plan", "Break the question into research areas"),
("🌐", "Search", "Search current web sources"),
("📄", "Extract", "Read original webpages"),
("🔍", "Cross-check", "Compare important evidence"),
("🤖", "Analyze", "Reason over the collected evidence"),
("📑", "Report", "Generate a detailed cited report"),
]

columns = st.columns(len(workflow))

for column, item in zip(columns, workflow):

```
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
```

# ============================================================

# WEBPAGE EXTRACTION

# ============================================================

def extract_webpage(url, max_chars=12000):

```
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
```

# ============================================================

# WEB RESEARCH

# ============================================================

def research_search(query, max_results=8):

```
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
```

# ============================================================

# CREWAI TOOL INPUT

# ============================================================

class ResearchSearchInput(BaseModel):

```
query: str = Field(
    ...,
    description=(
        "The web research query to search for."
    )
)
```

# ============================================================

# CREWAI RESEARCH TOOL

# ============================================================

class ResearchSearchTool(BaseTool):

```
name: str = "web_research_search"

description: str = (
    "Search the live web using DDGS and read accessible "
    "web pages. Returns original URLs, domains, snippets, "
    "and extracted source content."
)

args_schema: type[BaseModel] = ResearchSearchInput

def _run(self, query: str) -> str:

    sources = research_search(
        query=query,
        max_results=8
    )

    if not sources:

        return (
            "No usable sources were found."
        )

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
```

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

```
    return "\n".join(output)
```

research_tool = ResearchSearchTool()

# ============================================================

# SINGLE CREWAI AGENT

# ============================================================

research_agent = Agent(

```
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
```

)

# ============================================================

# RESEARCH TASK

# ============================================================

research_task = Task(

```
description="""

Research this question:

{topic}

Follow this research process:

STEP 1 — Understand the question.

Identify the main research question and determine
what information is necessary to answer it properly.

STEP 2 — Build a research plan.

Break the question into useful sub-questions when
necessary.

STEP 3 — Perform web research.

Use the web_research_search tool multiple times with
focused queries.

STEP 4 — Inspect original sources.

Read the actual webpage content instead of relying
only on search-result snippets.

STEP 5 — Prioritize sources.

Prefer primary sources, official organizations,
government sources, academic publications,
institutional reports, and reputable organizations
whenever relevant.

STEP 6 — Cross-check evidence.

Important factual claims should be checked against
more than one reliable source when possible.

STEP 7 — Identify disagreements.

If credible sources disagree, clearly explain what
they disagree about and identify the sources.

STEP 8 — Analyze.

Organize the evidence logically and distinguish
documented facts from interpretation.

STEP 9 — Report limitations.

Explain important limitations in the available
evidence.

STEP 10 — Produce the final report.

The final report MUST contain these sections:

# Executive Summary

Give a concise but informative overview.

# Research Question

State the question being investigated.

# Key Findings

Present the most important findings in clear
bullet points.

# Background

Explain the necessary context.

# Detailed Analysis

Provide detailed paragraphs explaining the evidence.

# Evidence

Connect important claims to the supporting sources.

# Conflicting Information

Explain meaningful disagreements between credible
sources. If there are none, say so.

# Limitations

Explain limitations in the evidence, sources,
methodology, or available information.

# Conclusion

Summarize what the evidence supports.

# Sources

Provide a numbered list of the ORIGINAL URLs used
during the research.

IMPORTANT:

Never invent facts.

Never invent statistics.

Never invent quotations.

Never invent sources.

Never invent URLs.

Never claim to have accessed a source that was not
actually returned by the research tool.

Every important factual claim should be traceable
to a source.

Use current information when the research question
requires current information.

Write detailed, professional paragraphs rather than
extremely short generic statements.
""",

expected_output="""

A comprehensive professional research report with:

1. Executive Summary
2. Research Question
3. Key Findings
4. Background
5. Detailed Analysis
6. Evidence
7. Conflicting Information
8. Limitations
9. Conclusion
10. Sources

The report must include original source URLs.

It must clearly distinguish evidence from interpretation.

It must not fabricate information.
""",

agent=research_agent
```

)

# ============================================================

# CREW

# ============================================================

research_crew = Crew(

```
agents=[research_agent],

tasks=[research_task],

process=Process.sequential,

verbose=True
```

)

# ============================================================

# RESEARCH INPUT

# ============================================================

st.markdown("---")

st.markdown("### 📝 Enter Your Research Question")

st.markdown(
""" <div style="
     color:#94A3B8;
     margin-bottom:10px;
     font-size:14px;
 ">
Ask a question that requires research. ResearchAI will
search current web sources and build an evidence-based report. </div>
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

# PDF GENERATION

# ============================================================

def clean_markdown_for_pdf(text):

```
text = text.replace(
    "**",
    ""
)

text = text.replace(
    "__",
    ""
)

text = re.sub(
    r"\[([^\]]+)\]\([^)]+\)",
    r"\1",
    text
)

return text.strip()
```

def create_pdf(report, topic):

```
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
    "ResearchTitle",
    parent=styles["Title"],
    fontSize=22,
    leading=28,
    alignment=TA_CENTER,
    spaceAfter=14
)

subtitle_style = ParagraphStyle(
    "ResearchSubtitle",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor(
        "#64748B"
    ),
    alignment=TA_CENTER,
    spaceAfter=20
)

heading_style = ParagraphStyle(
    "ResearchHeading",
    parent=styles["Heading1"],
    fontSize=16,
    leading=20,
    spaceBefore=16,
    spaceAfter=9,
    textColor=colors.HexColor(
        "#1E3A8A"
    )
)

subheading_style = ParagraphStyle(
    "ResearchSubHeading",
    parent=styles["Heading2"],
    fontSize=13,
    leading=17,
    spaceBefore=12,
    spaceAfter=7
)

body_style = ParagraphStyle(
    "ResearchBody",
    parent=styles["BodyText"],
    fontSize=10.5,
    leading=16,
    spaceAfter=9
)

bullet_style = ParagraphStyle(
    "ResearchBullet",
    parent=body_style,
    leftIndent=15,
    firstLineIndent=0
)

url_style = ParagraphStyle(
    "ResearchURL",
    parent=body_style,
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor(
        "#1D4ED8"
    )
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
        f"<b>Research Question:</b> "
        f"{clean_markdown_for_pdf(topic)}",
        body_style
    )
)

story.append(
    Spacer(
        1,
        10
    )
)

lines = report.splitlines()

bullet_items = []

def flush_bullets():

    nonlocal bullet_items

    if bullet_items:

        story.append(
            ListFlowable(
                [
                    ListItem(
                        Paragraph(
                            item,
                            bullet_style
                        )
                    )
                    for item in bullet_items
                ],
                bulletType="bullet",
                leftIndent=20
            )
        )

        story.append(
            Spacer(
                1,
                5
            )
        )

        bullet_items = []

for raw_line in lines:

    line = raw_line.strip()

    if not line:
        flush_bullets()
        continue

    # Main heading
    if line.startswith("# "):

        flush_bullets()

        heading = clean_markdown_for_pdf(
            line[2:]
        )

        story.append(
            Paragraph(
                heading,
                heading_style
            )
        )

    # Secondary heading
    elif line.startswith("## "):

        flush_bullets()

        heading = clean_markdown_for_pdf(
            line[3:]
        )

        story.append(
            Paragraph(
                heading,
                subheading_style
            )
        )

    # Bullet point
    elif line.startswith("- ") or line.startswith("* "):

        bullet_items.append(
            clean_markdown_for_pdf(
                line[2:]
            )
        )

    # Numbered list
    elif re.match(
        r"^\d+\.\s+",
        line
    ):

        flush_bullets()

        cleaned = re.sub(
            r"^\d+\.\s+",
            "",
            line
        )

        story.append(
            Paragraph(
                clean_markdown_for_pdf(
                    cleaned
                ),
                body_style
            )
        )

    # URL
    elif line.startswith("http://") or line.startswith("https://"):

        flush_bullets()

        story.append(
            Paragraph(
                line,
                url_style
            )
        )

    # Normal paragraph
    else:

        flush_bullets()

        cleaned = clean_markdown_for_pdf(
            line
        )

        story.append(
            Paragraph(
                cleaned,
                body_style
            )
        )

flush_bullets()

document.build(story)

buffer.seek(0)

return buffer.getvalue()
```

# ============================================================

# RUN RESEARCH

# ============================================================

if research_button:

```
if not topic.strip():

    st.warning(
        "Please enter a research question."
    )

else:

    st.session_state[
        "research_result"
    ] = ""

    st.session_state[
        "research_topic"
    ] = topic.strip()

    st.markdown("---")

    st.markdown(
        "### 🔄 Research Progress"
    )

    progress = st.progress(0)

    stage_container = st.container()

    try:

        # ------------------------------------------------
        # STAGE 1
        # ------------------------------------------------

        progress.progress(10)

        with stage_container:

            st.info(
                "📝 **Step 1 — Input received**\n\n"
                "ResearchAI has received your research question "
                "and is preparing the research task."
            )

        # ------------------------------------------------
        # STAGE 2
        # ------------------------------------------------

        progress.progress(20)

        with stage_container:

            st.info(
                "🧠 **Step 2 — Research planning**\n\n"
                "The CrewAI agent is analyzing the question "
                "and identifying the information it needs."
            )

        # ------------------------------------------------
        # STAGE 3
        # ------------------------------------------------

        progress.progress(30)

        with stage_container:

            st.info(
                "🌐 **Step 3 — Live web research**\n\n"
                "The agent is searching the web using the "
                "research tool."
            )

        # ------------------------------------------------
        # STAGE 4
        # ------------------------------------------------

        progress.progress(45)

        with stage_container:

            st.info(
                "📄 **Step 4 — Source extraction**\n\n"
                "Search results are being opened and readable "
                "content is being extracted from accessible pages."
            )

        # ------------------------------------------------
        # STAGE 5
        # ------------------------------------------------

        progress.progress(60)

        with stage_container:

            st.info(
                "🔍 **Step 5 — Evidence cross-checking**\n\n"
                "The agent is comparing sources and looking "
                "for supporting or conflicting evidence."
            )

        # ------------------------------------------------
        # ACTUAL CREWAI EXECUTION
        # ------------------------------------------------

        with st.spinner(
            "🤖 CrewAI agent is analyzing the collected evidence..."
        ):

            result = research_crew.kickoff(
                inputs={
                    "topic": topic.strip()
                }
            )

        # ------------------------------------------------
        # STAGE 6
        # ------------------------------------------------

        progress.progress(85)

        with stage_container:

            st.info(
                "🤖 **Step 6 — Evidence analysis**\n\n"
                "The agent is organizing the evidence and "
                "building the final research response."
            )

        # ------------------------------------------------
        # STAGE 7
        # ------------------------------------------------

        progress.progress(100)

        with stage_container:

            st.success(
                "📑 **Step 7 — Report generated**\n\n"
                "The research report has been completed."
            )

        st.session_state[
            "research_result"
        ] = str(result)

    except Exception as e:

        progress.progress(100)

        st.error(
            f"Research failed: {str(e)}"
        )
```

# ============================================================

# DISPLAY REPORT

# ============================================================

if st.session_state.get(
"research_result"
):

```
report = st.session_state[
    "research_result"
]

research_topic = st.session_state[
    "research_topic"
]

st.markdown("---")

st.markdown(
    "## 📑 Research Report"
)

# --------------------------------------------------------
# REPORT METADATA
# --------------------------------------------------------

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
        "Evidence Mode",
        "Cross-checked"
    )

# --------------------------------------------------------
# REPORT DISPLAY
# --------------------------------------------------------

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

# --------------------------------------------------------
# DOWNLOAD SECTION
# --------------------------------------------------------

st.markdown("---")

st.markdown(
    "### 📥 Download Your Research"
)

st.write(
    "Save the completed research as a professionally "
    "formatted PDF or as a text report."
)

# Create PDF
try:

    pdf_bytes = create_pdf(
        report,
        research_topic
    )

    safe_filename = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        research_topic[:50]
    ).strip("_")

    if not safe_filename:

        safe_filename = "research_report"

    pdf_filename = (
        f"{safe_filename}_ResearchAI.pdf"
    )

    txt_filename = (
        f"{safe_filename}_ResearchAI.txt"
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True
        )

    with download_col2:

        st.download_button(
            "📝 Download Text Report",
            data=report,
            file_name=txt_filename,
            mime="text/plain",
            use_container_width=True
        )

except Exception as e:

    st.error(
        f"PDF generation failed: {str(e)}"
    )

# --------------------------------------------------------
# TRACEABILITY NOTE
# --------------------------------------------------------

st.markdown("---")

st.info(
    "🔗 **Source traceability:** "
    "The report is instructed to include original URLs "
    "used during the research. Always review the cited "
    "sources when using the report for important decisions."
)
```
