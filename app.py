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
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CSS
# ============================================================

st.markdown(
    """
    <style>

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

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

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

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 25px;
        margin-bottom: 12px;
    }

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

    .report-box {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 14px;
        padding: 22px;
        line-height: 1.75;
    }

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

    [data-testid="stSidebar"] {
        background: #07101d;
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    textarea {
        background-color: #0f1b2d !important;
        color: #f8fafc !important;
        border: 1px solid #243b5a !important;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 700;
    }

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
        The application searches the current web,
        reads selected source pages, compares the evidence,
        and sends a compact evidence package to one
        CrewAI research agent.
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
        • Identify uncertainty
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
# 8. GROQ MODEL
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# IMPORTANT TOKEN SETTINGS
# ============================================================
#
# Your current Groq organization has an 8K TPM limit.
#
# Therefore:
#
# 1. We keep webpage evidence very small.
# 2. We keep the task prompt short.
# 3. We tell the model to use LOW reasoning.
# 4. We allow only ONE agent iteration.
#
# ============================================================


crew_llm = LLM(
    model=GROQ_MODEL,
    api_key=groq_api_key,
    api_base=GROQ_BASE_URL,
    temperature=0.2,
    max_tokens=1800,
    reasoning_effort="low",
)


# ============================================================
# 9. SEARCH SETTINGS
# ============================================================

MAX_SEARCH_RESULTS_PER_QUERY = 3

MAX_FINAL_SOURCES = 4

MAX_SOURCE_TEXT = 600

REQUEST_TIMEOUT = 10


# ============================================================
# 10. URL HELPERS
# ============================================================

def get_domain(url):

    try:

        domain = urlparse(url).netloc.lower()

        return domain.replace(
            "www.",
            ""
        )

    except Exception:

        return url


def normalize_url(url):

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

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

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

        return text[:MAX_SOURCE_TEXT]

    except Exception:

        return ""


# ============================================================
# 12. SEARCH QUERIES
# ============================================================

def create_search_queries(topic):

    return [
        f"{topic} 2026",
        f"{topic} latest research",
        f"{topic} official report",
    ]


# ============================================================
# 13. DDGS WEB SEARCH
# ============================================================

def perform_web_search(
    topic,
    progress_callback=None,
):

    queries = create_search_queries(
        topic
    )

    all_results = []

    try:

        with DDGS() as ddgs:

            for index, query in enumerate(
                queries,
                start=1,
            ):

                if progress_callback:

                    progress_callback(
                        f"🔎 Searching "
                        f"{index}/{len(queries)}: "
                        f"{query}"
                    )

                try:

                    results = list(
                        ddgs.text(
                            query,
                            max_results=MAX_SEARCH_RESULTS_PER_QUERY,
                        )
                    )

                except Exception:

                    results = []

                for result in results:

                    url = result.get(
                        "href",
                        "",
                    )

                    if not url:

                        continue

                    all_results.append(
                        {
                            "title": result.get(
                                "title",
                                "Untitled source",
                            ),
                            "url": normalize_url(
                                url
                            ),
                            "snippet": result.get(
                                "body",
                                "",
                            ),
                        }
                    )

                time.sleep(0.2)

    except Exception as error:

        raise RuntimeError(
            f"Web search failed: {error}"
        )


    # Remove duplicates.

    unique = {}

    for result in all_results:

        if result["url"] not in unique:

            unique[result["url"]] = result

    return list(
        unique.values()
    )


# ============================================================
# 14. SOURCE SCORING
# ============================================================

def source_score(source):

    domain = get_domain(
        source.get(
            "url",
            "",
        )
    )

    score = 0

    if domain.endswith(".gov"):
        score += 5

    if domain.endswith(".edu"):
        score += 4

    if domain.endswith(".org"):
        score += 2

    important_domains = [
        "who.int",
        "worldbank.org",
        "un.org",
        "oecd.org",
        "nih.gov",
        "nasa.gov",
        "europa.eu",
        "nature.com",
        "reuters.com",
    ]

    for trusted in important_domains:

        if trusted in domain:

            score += 5

    return score


# ============================================================
# 15. COLLECT EVIDENCE
# ============================================================

def collect_evidence(
    candidates,
    progress_callback=None,
):

    candidates = sorted(
        candidates,
        key=source_score,
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        if len(selected) >= MAX_FINAL_SOURCES:

            break

        if progress_callback:

            progress_callback(
                f"📄 Reading source "
                f"{len(selected) + 1}/"
                f"{MAX_FINAL_SOURCES}: "
                f"{candidate['title'][:60]}"
            )

        content = extract_webpage(
            candidate["url"]
        )

        if not content:

            content = candidate.get(
                "snippet",
                "",
            )[:MAX_SOURCE_TEXT]

        if not content:

            continue

        selected.append(
            {
                "title": candidate["title"],
                "url": candidate["url"],
                "domain": get_domain(
                    candidate["url"]
                ),
                "content": content,
            }
        )

    return selected


# ============================================================
# 16. BUILD SMALL EVIDENCE PACKAGE
# ============================================================

def build_evidence_package(
    topic,
    sources,
):

    parts = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        parts.append(
            f"""
SOURCE {index}
TITLE: {source['title']}
URL: {source['url']}
EVIDENCE: {source['content']}
""".strip()
        )

    evidence = "\n\n".join(
        parts
    )

    return f"""
QUESTION:
{topic}

YEAR:
2026

WEB EVIDENCE:
{evidence}
""".strip()


# ============================================================
# 17. SINGLE CREWAI AGENT
# ============================================================

research_agent = Agent(

    role="Research Analyst",

    goal=(
        "Analyze the supplied web evidence and "
        "write an accurate research report."
    ),

    backstory=(
        "You are a careful research analyst. "
        "Use evidence, compare sources, and never "
        "invent sources, URLs, statistics, or quotations."
    ),

    llm=crew_llm,

    allow_delegation=False,

    verbose=False,

    max_iter=1,

    reasoning=False,
)


# ============================================================
# 18. RESEARCH TASK
# ============================================================

def create_research_task(
    topic,
    evidence_package,
):

    description = f"""
Research question:
{topic}

Use ONLY the following verified web evidence:

{evidence_package}

Write a useful research report.

Use these sections:

# Research Report

## 1. Executive Summary
Summarize the main findings.

## 2. Introduction
Explain the topic.

## 3. Key Findings
Explain the important evidence.

## 4. Evidence Comparison
Compare what the sources say.

## 5. Uncertainty or Conflicting Evidence
Mention disagreements or missing evidence.

## 6. Conclusion
Give an evidence-based conclusion.

## 7. Sources
List the exact source titles and URLs supplied above.

Rules:

- Never invent a source.
- Never invent a URL.
- Never invent statistics.
- Never invent quotations.
- Use the supplied URLs exactly.
- If evidence is insufficient, say so.
- Keep the answer detailed but concise.
"""

    return Task(
        description=description,
        agent=research_agent,
        expected_output=(
            "A structured research report with "
            "evidence and source URLs."
        ),
    )


# ============================================================
# 19. CREATE CREW
# ============================================================

def create_research_crew(
    topic,
    evidence_package,
):

    task = create_research_task(
        topic,
        evidence_package,
    )

    return Crew(
        agents=[
            research_agent
        ],
        tasks=[
            task
        ],
        process=Process.sequential,
        verbose=False,
    )


# ============================================================
# 20. CLEAN REPORT
# ============================================================

def clean_report(result):

    if result is None:

        return ""

    if hasattr(
        result,
        "raw",
    ):

        return result.raw.strip()

    return str(
        result
    ).strip()


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
        textColor=colors.HexColor(
            "#334155"
        ),
    )

    story = []

    story.append(
        Paragraph(
            "AI Research Agent",
            title_style,
        )
    )

    safe_topic = (
        topic
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            f"<b>Research Question:</b> "
            f"{safe_topic}",
            body_style,
        )
    )

    story.append(
        Spacer(
            1,
            0.15 * inch,
        )
    )

    for line in report.split("\n"):

        line = line.strip()

        if not line:

            story.append(
                Spacer(
                    1,
                    0.06 * inch,
                )
            )

            continue

        safe_line = (
            line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        safe_line = re.sub(
            r"\*\*(.*?)\*\*",
            r"<b>\1</b>",
            safe_line,
        )

        if safe_line.startswith(
            "# "
        ):

            story.append(
                Paragraph(
                    safe_line[2:],
                    title_style,
                )
            )

        elif safe_line.startswith(
            "## "
        ):

            story.append(
                Paragraph(
                    safe_line[3:],
                    heading_style,
                )
            )

        elif safe_line.startswith(
            "### "
        ):

            story.append(
                Paragraph(
                    safe_line[4:],
                    heading_style,
                )
            )

        else:

            story.append(
                Paragraph(
                    safe_line,
                    body_style,
                )
            )

    story.append(
        PageBreak()
    )

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

        safe_title = (
            source["title"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        safe_url = (
            source["url"]
            .replace("&", "&amp;")
        )

        story.append(
            Paragraph(
                f"<b>{index}. "
                f"{safe_title}</b><br/>"
                f"{safe_url}",
                small_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.12 * inch,
            )
        )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 22. SESSION STATE
# ============================================================

if "last_report" not in st.session_state:

    st.session_state[
        "last_report"
    ] = ""


if "last_sources" not in st.session_state:

    st.session_state[
        "last_sources"
    ] = []


if "last_topic" not in st.session_state:

    st.session_state[
        "last_topic"
    ] = ""


# ============================================================
# 23. USER QUESTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📝 What would you like to research?
    </div>
    """,
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
# 24. EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    "**💡 Example research questions**"
)


example_columns = st.columns(
    3
)


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
# 25. START BUTTON
# ============================================================

st.markdown("")


start_research = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 26. RESEARCH PROCESS
# ============================================================

if start_research:

    if not topic or not topic.strip():

        st.warning(
            "Please enter a research question first."
        )

        st.stop()

    topic = topic.strip()

    st.session_state[
        "last_topic"
    ] = topic

    st.session_state[
        "last_report"
    ] = ""

    st.session_state[
        "last_sources"
    ] = []


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    with st.status(
        "🔬 Research in progress...",
        expanded=True,
    ) as status:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        st.write(
            "📝 Research question received."
        )

        st.write(
            f"**Question:** {topic}"
        )


        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        st.write(
            "🧠 Preparing research strategy..."
        )


        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        st.write(
            "🔎 Searching current web sources..."
        )

        try:

            candidates = perform_web_search(
                topic,
                progress_callback=st.write,
            )

        except Exception as error:

            status.update(
                label="❌ Web search failed",
                state="error",
            )

            st.error(
                str(error)
            )

            st.stop()


        if not candidates:

            status.update(
                label="⚠️ No sources found",
                state="error",
            )

            st.warning(
                "No web sources were found. "
                "Try another question."
            )

            st.stop()


        st.write(
            f"✅ Found {len(candidates)} "
            f"candidate sources."
        )


        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        st.write(
            "📄 Extracting information "
            "from webpages..."
        )

        sources = collect_evidence(
            candidates,
            progress_callback=st.write,
        )


        if not sources:

            status.update(
                label="⚠️ Source extraction failed",
                state="error",
            )

            st.warning(
                "Search results were found, but "
                "their content could not be extracted."
            )

            st.stop()


        st.write(
            f"✅ Extracted evidence from "
            f"{len(sources)} sources."
        )


        # ----------------------------------------------------
        # STEP 5
        # ----------------------------------------------------

        st.write(
            "⚖️ Cross-checking evidence "
            "across sources..."
        )

        evidence_package = (
            build_evidence_package(
                topic,
                sources,
            )
        )


        st.write(
            f"📦 Compact evidence prepared "
            f"({len(evidence_package):,} characters)."
        )


        # ----------------------------------------------------
        # STEP 6
        # ----------------------------------------------------

        st.write(
            "🤖 AI is analyzing the "
            "collected evidence..."
        )

        try:

            research_crew = (
                create_research_crew(
                    topic,
                    evidence_package,
                )
            )

            result = (
                research_crew.kickoff()
            )

            report = clean_report(
                result
            )


        except Exception as error:

            error_text = str(
                error
            )

            status.update(
                label="❌ AI analysis failed",
                state="error",
            )

            st.error(
                "The AI analysis failed."
            )

            st.markdown(
                "**Actual error from Groq/CrewAI:**"
            )

            st.code(
                error_text
            )

            if (
                "413" in error_text
                or "Requested" in error_text
                or "tokens per minute" in error_text
            ):

                st.warning(
                    """
                    The request is still exceeding the
                    current Groq token limit.

                    The app has already reduced the research
                    evidence and agent iterations. If this
                    message appears again, we will reduce the
                    final report/output budget further instead
                    of changing the interface.
                    """
                )

            st.stop()


        # ----------------------------------------------------
        # STEP 7
        # ----------------------------------------------------

        if not report:

            status.update(
                label="⚠️ No report generated",
                state="error",
            )

            st.warning(
                "The research agent returned no report."
            )

            st.stop()


        st.write(
            "📊 Formatting the research report..."
        )

        time.sleep(
            0.2
        )


        st.write(
            "🔗 Adding verified source URLs..."
        )


        status.update(
            label="✅ Research complete",
            state="complete",
        )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    st.session_state[
        "last_report"
    ] = report

    st.session_state[
        "last_sources"
    ] = sources


# ============================================================
# 27. DISPLAY REPORT
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


    # --------------------------------------------------------
    # REPORT HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            📑 Research Report
        </div>
        """,
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
    # REPORT
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
        """
        <div class="section-title">
            🔗 Verified Sources Used
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.caption(
        "These URLs were collected directly by the "
        "web-search process."
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
        """
        <div class="section-title">
            ⬇️ Download Research
        </div>
        """,
        unsafe_allow_html=True,
    )


    pdf_bytes = create_pdf(
        topic_used,
        report,
        sources,
    )


    download_columns = st.columns(
        2
    )


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
# 28. FOOTER
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
