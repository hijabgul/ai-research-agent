# ============================================================
# AI RESEARCH AGENT
# One CrewAI Agent + DDGS Web Research + Groq
# Streamlit Community Cloud
# ============================================================

import re
import time
from io import BytesIO
from urllib.parse import urlparse

import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

import litellm
from crewai import Agent, Task, Crew, Process, LLM

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
)


# ============================================================
# 1. LITELLM SAFETY PATCH
# ============================================================
# Some CrewAI/LiteLLM versions may send cache_breakpoint to
# Groq. Groq rejects this parameter, so remove it safely.

_original_completion = litellm.completion
_original_acompletion = getattr(litellm, "acompletion", None)


def _remove_cache_breakpoints(value):
    if isinstance(value, dict):
        cleaned = {}

        for key, item in value.items():
            if key == "cache_breakpoint":
                continue

            cleaned[key] = _remove_cache_breakpoints(item)

        return cleaned

    if isinstance(value, list):
        return [_remove_cache_breakpoints(item) for item in value]

    return value


def _safe_completion(*args, **kwargs):
    cleaned_kwargs = _remove_cache_breakpoints(kwargs)
    return _original_completion(*args, **cleaned_kwargs)


litellm.completion = _safe_completion


if _original_acompletion is not None:

    async def _safe_acompletion(*args, **kwargs):
        cleaned_kwargs = _remove_cache_breakpoints(kwargs)
        return await _original_acompletion(*args, **cleaned_kwargs)

    litellm.acompletion = _safe_acompletion


# ============================================================
# 2. STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.12),
                transparent 35%
            ),
            #08111f;
        color: #e5e7eb;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        padding: 30px 10px 20px 10px;
        text-align: center;
    }

    .hero-icon {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #94a3b8;
        max-width: 750px;
        margin: auto;
        line-height: 1.6;
    }

    /* Workflow cards */
    .workflow-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
        margin: 25px 0;
    }

    .workflow-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 14px 8px;
        text-align: center;
        min-height: 90px;
    }

    .workflow-number {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #2563eb;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .workflow-title {
        font-size: 12px;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Source card */
    .source-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .source-title {
        color: #dbeafe;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .source-url {
        color: #60a5fa;
        font-size: 12px;
        word-break: break-all;
    }

    .source-domain {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Report box */
    .report-box {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 14px;
        padding: 22px;
        line-height: 1.75;
    }

    /* Small badge */
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.14);
        color: #93c5fd;
        font-size: 12px;
        margin-right: 6px;
        border: 1px solid rgba(96, 165, 250, 0.18);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #07101d;
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    /* Text area */
    textarea {
        background-color: #0f1b2d !important;
        color: #f8fafc !important;
        border: 1px solid #243b5a !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 700;
    }

    /* Responsive workflow */
    @media (max-width: 900px) {
        .workflow-container {
            grid-template-columns: repeat(3, 1fr);
        }
    }

    @media (max-width: 600px) {
        .workflow-container {
            grid-template-columns: repeat(2, 1fr);
        }

        .hero-title {
            font-size: 31px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">🔎</div>

        <div class="hero-title">
            AI Research Agent
        </div>

        <div class="hero-subtitle">
            Research current topics using live web sources,
            cross-check evidence, and generate a detailed
            source-backed research report.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. WORKFLOW
# ============================================================

st.markdown(
    """
    <div class="workflow-container">

        <div class="workflow-card">
            <div class="workflow-number">1</div>
            <div class="workflow-title">Question</div>
        </div>

        <div class="workflow-card">
            <div class="workflow-number">2</div>
            <div class="workflow-title">Web Search</div>
        </div>

        <div class="workflow-card">
            <div class="workflow-number">3</div>
            <div class="workflow-title">Extract</div>
        </div>

        <div class="workflow-card">
            <div class="workflow-number">4</div>
            <div class="workflow-title">Cross-check</div>
        </div>

        <div class="workflow-card">
            <div class="workflow-number">5</div>
            <div class="workflow-title">AI Analysis</div>
        </div>

        <div class="workflow-card">
            <div class="workflow-number">6</div>
            <div class="workflow-title">Report</div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Research Settings")

    st.markdown(
        """
        <span class="badge">1 CrewAI Agent</span>
        <span class="badge">Live Web</span>
        <span class="badge">Groq</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 📚 Research method")

    st.write(
        """
        The application searches the current web first,
        reads selected source pages, and then sends a
        compact evidence package to one CrewAI research agent.
        """
    )

    st.markdown("---")

    st.markdown("### 🛡️ Evidence rules")

    st.write(
        """
        • Prefer authoritative sources  
        • Preserve original URLs  
        • Compare multiple sources  
        • Do not invent evidence  
        • Clearly identify uncertainty
        """
    )

    st.markdown("---")

    st.caption(
        "Built with Streamlit, CrewAI, Groq, DDGS and Python."
    )


# ============================================================
# 7. GROQ SECRET
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        """
        GROQ_API_KEY was not found in Streamlit Secrets.

        Go to:

        Streamlit Cloud → App → Settings → Secrets

        and make sure you have:

        GROQ_API_KEY = "your_key_here"
        """
    )

    st.stop()


# ============================================================
# 8. MODEL CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


crew_llm = LLM(
    model=GROQ_MODEL,
    api_key=groq_api_key,
    api_base=GROQ_BASE_URL,
    temperature=0.15,
)


# ============================================================
# 9. SEARCH CONFIGURATION
# ============================================================

MAX_SEARCH_RESULTS_PER_QUERY = 3
MAX_FINAL_SOURCES = 5

# IMPORTANT:
# Keep this small because the user's current Groq limit is
# 8K TPM. We deliberately send a compact evidence package.
MAX_SOURCE_TEXT = 850

REQUEST_TIMEOUT = 12


# ============================================================
# 10. URL / DOMAIN HELPERS
# ============================================================

def get_domain(url):
    """Return a clean domain name."""

    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
        return domain

    except Exception:
        return url


def normalize_url(url):
    """Remove tracking parameters from URLs where possible."""

    try:

        parsed = urlparse(url)

        return (
            f"{parsed.scheme}://"
            f"{parsed.netloc}"
            f"{parsed.path}"
        ).rstrip("/")

    except Exception:

        return url


# ============================================================
# 11. WEBPAGE EXTRACTION
# ============================================================

def extract_webpage(url):
    """
    Download a webpage and extract readable text.

    The extracted text is intentionally limited so that
    the final LLM request stays below the user's Groq TPM limit.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            return ""

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml" not in content_type
        ):
            return ""

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        # Remove unnecessary elements.
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "svg",
                "iframe",
            ]
        ):
            element.decompose()

        # Prefer article/main content.
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )

        text = main_content.get_text(
            separator=" ",
            strip=True,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        # Keep only a compact evidence section.
        if len(text) > MAX_SOURCE_TEXT:
            text = text[:MAX_SOURCE_TEXT] + "..."

        return text

    except Exception:
        return ""


# ============================================================
# 12. SEARCH QUERY GENERATOR
# ============================================================

def create_search_queries(topic):
    """
    Create three compact search queries.

    We intentionally use a small number of searches to avoid
    unnecessary requests and keep the application fast.
    """

    current_year = 2026

    return [
        f"{topic} {current_year}",
        f"{topic} latest research evidence",
        f"{topic} official report statistics",
    ]


# ============================================================
# 13. WEB SEARCH
# ============================================================

def perform_web_search(topic, progress_callback=None):
    """
    Search DDGS and return candidate sources.
    """

    queries = create_search_queries(topic)

    all_results = []

    try:

        with DDGS() as ddgs:

            for index, query in enumerate(
                queries,
                start=1,
            ):

                if progress_callback:

                    progress_callback(
                        f"🔎 Searching the web — "
                        f"query {index}/{len(queries)}"
                    )

                try:

                    results = list(
                        ddgs.text(
                            query,
                            max_results=MAX_SEARCH_RESULTS_PER_QUERY,
                        )
                    )

                except Exception as search_error:

                    if progress_callback:

                        progress_callback(
                            f"⚠️ Search {index} had an issue: "
                            f"{str(search_error)[:100]}"
                        )

                    results = []

                for result in results:

                    url = result.get(
                        "href",
                        "",
                    )

                    title = result.get(
                        "title",
                        "Untitled source",
                    )

                    snippet = result.get(
                        "body",
                        "",
                    )

                    if not url:
                        continue

                    all_results.append(
                        {
                            "title": title,
                            "url": normalize_url(url),
                            "snippet": snippet,
                        }
                    )

                time.sleep(0.2)

    except Exception as error:

        raise RuntimeError(
            f"Web search failed: {error}"
        )


    # --------------------------------------------------------
    # Remove duplicate URLs.
    # --------------------------------------------------------

    unique = {}

    for result in all_results:

        url = result["url"]

        if url not in unique:

            unique[url] = result

    return list(unique.values())


# ============================================================
# 14. SOURCE QUALITY SCORING
# ============================================================

def source_score(source):
    """
    Basic source prioritization.

    This is not claiming that a domain is always authoritative.
    It simply gives common official/academic domains a higher
    starting priority.
    """

    domain = get_domain(
        source.get("url", "")
    )

    score = 0

    trusted_endings = [
        ".gov",
        ".edu",
        ".org",
    ]

    for ending in trusted_endings:

        if domain.endswith(ending):
            score += 4

    trusted_domains = [
        "who.int",
        "worldbank.org",
        "un.org",
        "oecd.org",
        "nih.gov",
        "nature.com",
        "sciencedirect.com",
        "reuters.com",
        "bbc.com",
        "nasa.gov",
        "europa.eu",
    ]

    for trusted in trusted_domains:

        if trusted in domain:
            score += 5

    return score


# ============================================================
# 15. READ SELECTED SOURCES
# ============================================================

def collect_evidence(
    candidates,
    progress_callback=None,
):
    """
    Read a small number of webpages.

    Maximum:
        5 sources
        ~850 characters of page text each

    This is deliberate so that the final CrewAI request
    remains comfortably smaller than the 8K TPM limit.
    """

    candidates = sorted(
        candidates,
        key=source_score,
        reverse=True,
    )

    selected_sources = []

    for candidate in candidates:

        if len(selected_sources) >= MAX_FINAL_SOURCES:
            break

        if progress_callback:

            progress_callback(
                f"📄 Reading source "
                f"{len(selected_sources) + 1}/"
                f"{MAX_FINAL_SOURCES}: "
                f"{candidate['title'][:65]}"
            )

        content = extract_webpage(
            candidate["url"]
        )

        # If webpage extraction fails, we can still use the
        # search snippet as limited evidence.
        if not content:

            content = candidate.get(
                "snippet",
                "",
            )

        if not content:
            continue

        selected_sources.append(
            {
                "title": candidate["title"],
                "url": candidate["url"],
                "domain": get_domain(
                    candidate["url"]
                ),
                "snippet": candidate.get(
                    "snippet",
                    "",
                )[:300],
                "content": content[:MAX_SOURCE_TEXT],
            }
        )

    return selected_sources


# ============================================================
# 16. BUILD COMPACT EVIDENCE PACKAGE
# ============================================================

def build_evidence_package(
    topic,
    sources,
):
    """
    Build a compact prompt for the ONE CrewAI agent.

    The evidence is deliberately limited.
    """

    sections = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        section = f"""
SOURCE {index}
Title: {source['title']}
URL: {source['url']}
Domain: {source['domain']}

Evidence:
{source['content']}
"""

        sections.append(section.strip())

    evidence = "\n\n".join(
        sections
    )

    return f"""
RESEARCH QUESTION:
{topic}

CURRENT YEAR:
2026

VERIFIED WEB SOURCES:
{evidence}

IMPORTANT:
The URLs above are the actual URLs collected by the
application. Do not invent additional URLs.

Use only the supplied evidence for source-specific claims.
You may explain general reasoning, but do not invent
statistics, quotations, studies, organizations, or facts.

If sources disagree, explicitly mention the disagreement.
If evidence is insufficient, say so.
""".strip()


# ============================================================
# 17. CREATE THE SINGLE CREWAI AGENT
# ============================================================

research_agent = Agent(
    role="Senior Research Analyst",

    goal=(
        "Analyze verified web evidence and produce a clear, "
        "accurate, detailed research report without inventing "
        "sources, statistics, quotations, or evidence."
    ),

    backstory=(
        "You are a careful research analyst. "
        "You distinguish facts from interpretation, "
        "compare evidence across sources, identify "
        "conflicting information, and clearly communicate "
        "uncertainty. You never fabricate citations."
    ),

    llm=crew_llm,

    allow_delegation=False,

    verbose=False,
)


# ============================================================
# 18. CREATE RESEARCH TASK
# ============================================================

def create_research_task(
    topic,
    evidence_package,
):

    task_description = f"""
You are the only AI research agent in this application.

Your job is to analyze the supplied web evidence and produce
a professional research report about the user's question.

USER QUESTION:
{topic}

EVIDENCE PACKAGE:
{evidence_package}

------------------------------------------------------------
REPORT REQUIREMENTS
------------------------------------------------------------

Write a detailed but focused report.

Use these sections:

# Research Report

## 1. Executive Summary
Give a concise overview of the main findings.

## 2. Introduction
Explain what the research question is about and why it
matters.

## 3. Key Findings
Explain the most important findings in clear paragraphs.

## 4. Evidence Analysis
Compare the supplied sources.
Explain which findings are supported by multiple sources.

## 5. Conflicting or Uncertain Evidence
If sources disagree or evidence is incomplete, explain that
clearly.

Do NOT pretend conflicting evidence does not exist.

## 6. Important Facts and Data
Include important statistics or factual information only
when supported by the supplied evidence.

Never invent numbers.

## 7. Practical Implications
Explain what the findings mean in practical terms.

## 8. Conclusion
Summarize the evidence-based conclusion.

## 9. Sources
List the supplied sources using their exact titles and URLs.

------------------------------------------------------------
SOURCE RULES
------------------------------------------------------------

1. Never invent a URL.
2. Never invent a source.
3. Never invent a statistic.
4. Never invent a quotation.
5. Do not claim that you personally visited a source.
6. Use the exact URLs supplied in the evidence package.
7. Distinguish facts from interpretation.
8. If evidence is insufficient, explicitly say so.
9. Prefer evidence supported by more than one source.
10. Keep the report readable and well organized.

Return ONLY the research report.
"""

    return Task(
        description=task_description,
        agent=research_agent,
        expected_output=(
            "A detailed, evidence-based research report with "
            "clear sections and a source list."
        ),
    )


# ============================================================
# 19. CREATE CREW
# ============================================================

def create_research_crew(
    topic,
    evidence_package,
):

    research_task = create_research_task(
        topic,
        evidence_package,
    )

    return Crew(
        agents=[
            research_agent
        ],

        tasks=[
            research_task
        ],

        process=Process.sequential,

        verbose=False,
    )


# ============================================================
# 20. CLEAN CREW OUTPUT
# ============================================================

def clean_report(result):
    """
    Convert CrewAI output to normal text.
    """

    if result is None:
        return ""

    # CrewAI TaskOutput commonly has raw.
    if hasattr(result, "raw"):

        text = result.raw

    else:

        text = str(result)

    text = text.strip()

    return text


# ============================================================
# 21. PDF GENERATOR
# ============================================================

def create_pdf(
    topic,
    report,
    sources,
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
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=8,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    story.append(
        Paragraph(
            "AI Research Agent",
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Research Question:</b> "
            f"{topic}",
            body_style,
        )
    )

    story.append(
        Spacer(
            1,
            0.15 * inch,
        )
    )

    # --------------------------------------------------------
    # Convert report markdown-ish formatting to PDF.
    # --------------------------------------------------------

    lines = report.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            story.append(
                Spacer(
                    1,
                    0.06 * inch,
                )
            )
            continue

        # Remove markdown heading symbols.
        if line.startswith("# "):

            text = line[2:].strip()

            story.append(
                Paragraph(
                    text,
                    title_style,
                )
            )

        elif line.startswith("## "):

            text = line[3:].strip()

            story.append(
                Paragraph(
                    text,
                    heading_style,
                )
            )

        elif line.startswith("### "):

            text = line[4:].strip()

            story.append(
                Paragraph(
                    text,
                    heading_style,
                )
            )

        else:

            # Escape problematic HTML characters.
            safe_line = (
                line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            # Basic bold markdown.
            safe_line = re.sub(
                r"\*\*(.*?)\*\*",
                r"<b>\1</b>",
                safe_line,
            )

            story.append(
                Paragraph(
                    safe_line,
                    body_style,
                )
            )

    # --------------------------------------------------------
    # Verified sources section.
    # --------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Verified Sources Collected",
            heading_style,
        )
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):

        source_text = (
            f"<b>{index}. "
            f"{source['title']}</b><br/>"
            f"{source['url']}"
        )

        story.append(
            Paragraph(
                source_text,
                small_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.12 * inch,
            )
        )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 22. USER INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📝 What would you like to research?</div>',
    unsafe_allow_html=True,
)

topic = st.text_area(
    "Research question",
    placeholder=(
        "Example: What are the latest developments "
        "in artificial intelligence in 2026?"
    ),
    height=130,
    label_visibility="collapsed",
)


# ============================================================
# 23. EXAMPLE QUESTIONS
# ============================================================

st.markdown("**💡 Example research questions**")

example_columns = st.columns(3)

example_questions = [
    "What are the latest developments in AI in 2026?",
    "What are the current benefits and risks of renewable energy?",
    "What recent research exists on cybersecurity threats?",
]


for column, question in zip(
    example_columns,
    example_questions,
):

    with column:

        if st.button(
            question,
            use_container_width=True,
        ):

            topic = question


# ============================================================
# 24. START RESEARCH
# ============================================================

st.markdown("")

start_research = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 25. RESEARCH EXECUTION
# ============================================================

if start_research:

    if not topic or not topic.strip():

        st.warning(
            "Please enter a research question first."
        )

        st.stop()

    topic = topic.strip()

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    st.session_state["last_topic"] = topic
    st.session_state["last_report"] = ""
    st.session_state["last_sources"] = []

    # --------------------------------------------------------
    # Status container
    # --------------------------------------------------------

    with st.status(
        "🔬 Research in progress...",
        expanded=True,
    ) as status:

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        st.write(
            "📝 Research question received."
        )

        st.write(
            f"**Question:** {topic}"
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        st.write(
            "🧠 Preparing the research strategy..."
        )

        time.sleep(0.2)

        # ----------------------------------------------------
        # Stage 3 — SEARCH
        # ----------------------------------------------------

        st.write(
            "🔎 Searching current web sources..."
        )

        try:

            candidates = perform_web_search(
                topic
            )

        except Exception as search_error:

            status.update(
                label="❌ Research failed",
                state="error",
            )

            st.error(
                f"Web search failed: {search_error}"
            )

            st.stop()

        if not candidates:

            status.update(
                label="⚠️ No sources found",
                state="error",
            )

            st.warning(
                "No web sources were found. "
                "Try a different or more specific question."
            )

            st.stop()

        st.write(
            f"✅ Found {len(candidates)} candidate sources."
        )

        # ----------------------------------------------------
        # Stage 4 — EXTRACTION
        # ----------------------------------------------------

        st.write(
            "📄 Reading selected source webpages..."
        )

        sources = collect_evidence(
            candidates,
            progress_callback=st.write,
        )

        if not sources:

            status.update(
                label="⚠️ Could not read sources",
                state="error",
            )

            st.warning(
                "Search results were found, but the webpages "
                "could not be read."
            )

            st.stop()

        st.write(
            f"✅ Collected evidence from "
            f"{len(sources)} sources."
        )

        # ----------------------------------------------------
        # Stage 5 — COMPACT EVIDENCE
        # ----------------------------------------------------

        st.write(
            "⚖️ Comparing and compacting evidence..."
        )

        evidence_package = build_evidence_package(
            topic,
            sources,
        )

        # Display approximate character count.
        evidence_chars = len(
            evidence_package
        )

        st.write(
            f"📦 Evidence package prepared "
            f"({evidence_chars:,} characters)."
        )

        # ----------------------------------------------------
        # Stage 6 — SINGLE CREWAI AGENT
        # ----------------------------------------------------

        st.write(
            "🤖 Sending the evidence to the single "
            "CrewAI research agent..."
        )

        st.write(
            "This is the only LLM stage of the workflow, "
            "so the app avoids repeatedly sending large "
            "webpage contents to Groq."
        )

        try:

            research_crew = create_research_crew(
                topic,
                evidence_package,
            )

            result = research_crew.kickoff()

            report = clean_report(
                result
            )

        except Exception as agent_error:

            error_text = str(
                agent_error
            )

            status.update(
                label="❌ AI analysis failed",
                state="error",
            )

            # ------------------------------------------------
            # Friendly Groq token-limit message.
            # ------------------------------------------------

            if (
                "413" in error_text
                or "Request too large" in error_text
                or "tokens per minute" in error_text
                or "Requested" in error_text
            ):

                st.error(
                    """
                    The Groq request was still too large for
                    the current 8K TPM limit.

                    The application has already been designed
                    to keep the research package small, but
                    CrewAI itself adds some prompt overhead.

                    Try a shorter research question. If this
                    still happens, the evidence limit can be
                    reduced further.
                    """
                )

            else:

                st.error(
                    "An error occurred while running the "
                    "research agent."
                )

                st.code(
                    error_text
                )

            st.stop()

        if not report:

            status.update(
                label="⚠️ No report generated",
                state="error",
            )

            st.warning(
                "The research agent did not return a report."
            )

            st.stop()

        # ----------------------------------------------------
        # Stage 7
        # ----------------------------------------------------

        st.write(
            "📊 Formatting the research report..."
        )

        time.sleep(0.2)

        st.write(
            "🔗 Attaching verified source URLs..."
        )

        time.sleep(0.2)

        # ----------------------------------------------------
        # Complete
        # ----------------------------------------------------

        status.update(
            label="✅ Research complete",
            state="complete",
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    st.session_state["last_report"] = report
    st.session_state["last_sources"] = sources


# ============================================================
# 26. DISPLAY REPORT
# ============================================================

if st.session_state.get(
    "last_report",
    "",
):

    report = st.session_state[
        "last_report"
    ]

    sources = st.session_state.get(
        "last_sources",
        [],
    )

    topic_used = st.session_state.get(
        "last_topic",
        "Research",
    )

    st.markdown(
        '<div class="section-title">📑 Research Report</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="badge">AI Generated</span>
        <span class="badge">Web Evidence</span>
        <span class="badge">Source Checked</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------
    # Main report
    # --------------------------------------------------------

    st.markdown(
        '<div class="report-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        report
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # VERIFIED SOURCES
    # ========================================================

    st.markdown(
        '<div class="section-title">🔗 Verified Sources Used</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "These URLs were collected directly by the application's "
        "web-search process. They are shown separately so the "
        "research report does not depend on the AI inventing URLs."
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):

        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-title">
                    {index}. {source['title']}
                </div>

                <div class="source-url">
                    {source['url']}
                </div>

                <div class="source-domain">
                    🌐 {source['domain']}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.link_button(
            "🔗 Open source",
            source["url"],
        )

    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.markdown(
        '<div class="section-title">⬇️ Download Research</div>',
        unsafe_allow_html=True,
    )

    pdf_bytes = create_pdf(
        topic_used,
        report,
        sources,
    )

    download_columns = st.columns(2)

    with download_columns[0]:

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="AI_Research_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with download_columns[1]:

        st.download_button(
            label="📝 Download TXT Report",
            data=report,
            file_name="AI_Research_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ============================================================
# 27. FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:50px;
        padding:20px;
        color:#64748b;
        font-size:13px;
    ">
        AI Research Agent • One CrewAI Agent •
        Live Web Research • Groq
    </div>
    """,
    unsafe_allow_html=True,
)
