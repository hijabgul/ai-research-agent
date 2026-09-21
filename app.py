import re
import html
from io import BytesIO
from urllib.parse import urlparse

import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

import litellm
from crewai import Agent, Task, Crew, Process, LLM

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ============================================================
# GROQ / LITELLM SAFETY PATCH
# ============================================================

_original_completion = litellm.completion
_original_acompletion = getattr(litellm, "acompletion", None)


def remove_unsupported_parameters(value):
    """
    Recursively removes parameters that Groq does not accept.
    """
    if isinstance(value, dict):
        cleaned = {}

        for key, item in value.items():
            if key in {
                "cache_breakpoint",
                "cache_control",
            }:
                continue

            cleaned[key] = remove_unsupported_parameters(item)

        return cleaned

    if isinstance(value, list):
        return [
            remove_unsupported_parameters(item)
            for item in value
        ]

    return value


def safe_completion(*args, **kwargs):
    """
    Keeps the Groq request small enough for the user's
    current token-per-minute limit.
    """

    kwargs = remove_unsupported_parameters(kwargs)

    model = str(kwargs.get("model", ""))

    if "gpt-oss" in model:

        # GPT-OSS supports low reasoning.
        kwargs["reasoning_effort"] = "low"

        # Do not allow CrewAI/LiteLLM to request a huge completion.
        kwargs["max_tokens"] = 1200

        # GPT-OSS can return reasoning separately.
        kwargs["include_reasoning"] = False

    return _original_completion(*args, **kwargs)


litellm.completion = safe_completion


if _original_acompletion is not None:

    async def safe_acompletion(*args, **kwargs):

        kwargs = remove_unsupported_parameters(kwargs)

        model = str(kwargs.get("model", ""))

        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = 1200
            kwargs["include_reasoning"] = False

        return await _original_acompletion(
            *args,
            **kwargs
        )

    litellm.acompletion = safe_acompletion


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

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
            #07111f;
        color: #e5edf8;
    }

    .main {
        padding-top: 1rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label, span {
        color: #cbd5e1;
    }


    /* ---------- HERO ---------- */

    .hero {
        padding: 28px;
        border-radius: 22px;
        margin-bottom: 24px;

        background:
            linear-gradient(
                135deg,
                rgba(15, 35, 64, 0.98),
                rgba(9, 24, 45, 0.98)
            );

        border: 1px solid rgba(96, 165, 250, 0.20);

        box-shadow:
            0 20px 50px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #94a3b8;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 7px;
        margin-top: 12px;

        background: rgba(37,99,235,0.14);
        border: 1px solid rgba(96,165,250,0.20);

        color: #93c5fd;
        font-size: 13px;
        font-weight: 600;
    }


    /* ---------- WORKFLOW ---------- */

    .workflow-card {
        background: rgba(15, 31, 52, 0.90);
        border: 1px solid rgba(148,163,184,0.13);

        border-radius: 16px;
        padding: 18px;
        text-align: center;

        min-height: 125px;
    }

    .workflow-number {
        font-size: 25px;
        font-weight: 800;
        color: #60a5fa;
    }

    .workflow-title {
        font-weight: 700;
        margin-top: 5px;
        color: #f8fafc;
    }

    .workflow-text {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 5px;
    }


    /* ---------- INPUT ---------- */

    .input-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 8px;
    }


    /* ---------- REPORT ---------- */

    .report-header {
        margin-top: 30px;
        margin-bottom: 12px;
        font-size: 25px;
        font-weight: 800;
    }

    .report-box {
        background: #0b1728;
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 18px;
        padding: 26px;
        line-height: 1.75;
    }


    /* ---------- SOURCES ---------- */

    .source-card {
        background: #0b1728;
        border: 1px solid rgba(148,163,184,0.13);
        border-radius: 14px;
        padding: 17px;
        margin-bottom: 12px;
    }

    .source-title {
        font-size: 16px;
        font-weight: 700;
        color: #e2e8f0;
    }

    .source-domain {
        font-size: 12px;
        color: #60a5fa;
        margin-top: 5px;
    }

    .source-url {
        font-size: 12px;
        color: #94a3b8;
        word-break: break-all;
        margin-top: 7px;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;

        border: 1px solid rgba(96,165,250,0.25);

        background: #10233d;
        color: #dbeafe;

        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        background: #15304f;
        color: white;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #07111f;
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
    }

    .sidebar-item {
        background: #0d1b2e;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid rgba(148,163,184,0.10);
    }

    .sidebar-label {
        color: #94a3b8;
        font-size: 12px;
    }

    .sidebar-value {
        color: #e2e8f0;
        font-weight: 700;
        margin-top: 3px;
    }


    /* ---------- STATUS ---------- */

    .status-box {
        background: rgba(15,31,52,0.85);
        border: 1px solid rgba(96,165,250,0.15);
        padding: 15px;
        border-radius: 14px;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "research_input" not in st.session_state:
    st.session_state.research_input = ""

if "last_report" not in st.session_state:
    st.session_state.last_report = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🔎 AI Research Agent</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-item">
            <div class="sidebar-label">ARCHITECTURE</div>
            <div class="sidebar-value">1 CrewAI Agent</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">WEB RESEARCH</div>
            <div class="sidebar-value">DuckDuckGo + Web Pages</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">LLM</div>
            <div class="sidebar-value">Groq GPT-OSS 20B</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">REASONING</div>
            <div class="sidebar-value">Low / Token Safe</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "The agent searches the live web, extracts evidence, "
        "cross-checks sources, and generates a cited research report."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🔎 AI Research Agent
        </div>

        <div class="hero-subtitle">
            Research current topics using live web sources,
            evidence extraction, source verification, and
            a single CrewAI research agent.
        </div>

        <div>
            <span class="badge">ONE CREWAI AGENT</span>
            <span class="badge">LIVE WEB</span>
            <span class="badge">GROQ</span>
            <span class="badge">SOURCE CITATIONS</span>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORKFLOW
# ============================================================

st.subheader("How it works")

workflow = st.columns(5)

steps = [
    ("01", "Question", "Enter a research topic"),
    ("02", "Search", "Find current web sources"),
    ("03", "Extract", "Read source evidence"),
    ("04", "Analyze", "CrewAI research agent"),
    ("05", "Report", "Cited research result"),
]

for col, step in zip(workflow, steps):

    with col:

        st.markdown(
            f"""
            <div class="workflow-card">

                <div class="workflow-number">
                    {step[0]}
                </div>

                <div class="workflow-title">
                    {step[1]}
                </div>

                <div class="workflow-text">
                    {step[2]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="input-title">💡 Example Research Questions</div>',
    unsafe_allow_html=True
)

example_columns = st.columns(3)

examples = [
    "What are the latest AI trends in 2026?",
    "What is the impact of AI on education?",
    "What are the latest developments in renewable energy?",
]

for i, example in enumerate(examples):

    with example_columns[i]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):

            st.session_state.research_input = example
            st.rerun()


# ============================================================
# USER INPUT
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="input-title">📝 Research Question</div>',
    unsafe_allow_html=True
)

topic = st.text_area(
    "Research topic",
    key="research_input",
    height=130,
    placeholder=(
        "Example: What are the latest developments "
        "in artificial intelligence in 2026?"
    ),
    label_visibility="collapsed",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

MAX_RESULTS_PER_QUERY = 2
MAX_SELECTED_SOURCES = 3
MAX_PAGE_TEXT = 500
REQUEST_TIMEOUT = 10


def clean_text(text):
    """
    Cleans whitespace and removes excessive characters.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def search_web(query):
    """
    Search current web using DDGS.
    """

    results = []

    queries = [
        f"{query} 2026",
        f"{query} latest research evidence",
        f"{query} official report",
    ]

    seen_urls = set()

    try:

        with DDGS() as ddgs:

            for search_query in queries:

                try:

                    found = ddgs.text(
                        search_query,
                        max_results=MAX_RESULTS_PER_QUERY
                    )

                    for item in found:

                        url = item.get("href") or item.get("url")

                        if not url:
                            continue

                        if url in seen_urls:
                            continue

                        seen_urls.add(url)

                        results.append(
                            {
                                "title": clean_text(
                                    item.get("title", "Untitled source")
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
            f"Web search could not start: {str(e)}"
        )

    return results


def source_score(source):
    """
    Gives priority to authoritative domains.
    """

    domain = source.get("domain", "")

    score = 0

    if domain.endswith(".gov"):
        score += 10

    if domain.endswith(".edu"):
        score += 8

    if domain.endswith(".org"):
        score += 5

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
            score += 10

    if len(source.get("snippet", "")) > 100:
        score += 2

    return score


def extract_page(source):
    """
    Reads the actual webpage.
    """

    url = source["url"]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            return source["snippet"]

        soup = BeautifulSoup(
            response.text,
            "lxml"
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

        if len(text) > MAX_PAGE_TEXT:
            text = text[:MAX_PAGE_TEXT]

        return text or source["snippet"]

    except Exception:
        return source["snippet"]


def select_sources(results):
    """
    Selects the strongest sources.
    """

    ranked = sorted(
        results,
        key=source_score,
        reverse=True
    )

    selected = []

    for item in ranked:

        if len(selected) >= MAX_SELECTED_SOURCES:
            break

        selected.append(item)

    return selected


def build_evidence(selected_sources):
    """
    Creates a very compact evidence package.
    Keeping this small is important because the
    current Groq organization has an 8K TPM limit.
    """

    evidence = []

    for number, source in enumerate(
        selected_sources,
        start=1
    ):

        page_text = extract_page(source)

        evidence.append(
            f"""
SOURCE {number}
Title: {source['title']}
Domain: {source['domain']}
URL: {source['url']}

Evidence:
{page_text[:MAX_PAGE_TEXT]}
""".strip()
        )

    return "\n\n".join(evidence)


def create_agent():
    """
    Creates exactly ONE CrewAI agent.
    """

    groq_key = st.secrets.get("GROQ_API_KEY")

    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in Streamlit Secrets."
        )

    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_key,
        api_base="https://api.groq.com/openai/v1",
        temperature=0.3,
    )

    agent = Agent(
        role="Senior Web Research Analyst",

        goal=(
            "Analyze the supplied web evidence and create "
            "an accurate research report. Never invent "
            "facts, statistics, quotations, sources, or URLs."
        ),

        backstory=(
            "You are a careful research analyst. "
            "You rely only on the evidence supplied to you. "
            "When evidence is uncertain or conflicting, "
            "clearly say so."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=False,

        max_iter=1,
    )

    return agent


def run_research(topic, evidence):
    """
    Runs ONE CrewAI agent with ONE task.
    """

    agent = create_agent()

    task_description = f"""
Research question:

{topic}

Use ONLY the evidence supplied below.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not invent statistics.
3. Do not invent quotations.
4. Do not invent sources.
5. Do not create URLs that were not supplied.
6. If the evidence is insufficient, say so.
7. Distinguish facts from uncertainty.
8. Mention conflicting evidence when present.
9. Keep the answer useful and factual.
10. Cite sources using their supplied URLs.

Write a concise but useful report using these sections:

## Executive Summary

## Key Findings

## Evidence Analysis

## Important Facts

## Uncertainty or Conflicting Evidence

## Practical Implications

## Conclusion

## Sources

The report should be approximately 600-900 words.

WEB EVIDENCE:

{evidence}
"""

    task = Task(
        description=task_description,

        expected_output=(
            "A factual research report based only on "
            "the supplied web evidence, including "
            "the supplied source URLs."
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

    return str(result)


def create_pdf(topic, report, sources):
    """
    Creates a downloadable PDF.
    """

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

    story = []

    story.append(
        Paragraph(
            "AI Research Agent Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            f"<b>Research Question:</b> "
            f"{html.escape(topic)}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    for paragraph in report.split("\n"):

        paragraph = paragraph.strip()

        if not paragraph:
            story.append(Spacer(1, 8))
            continue

        safe = html.escape(paragraph)

        if safe.startswith("## "):

            story.append(
                Paragraph(
                    safe.replace("## ", ""),
                    styles["Heading2"]
                )
            )

        else:

            story.append(
                Paragraph(
                    safe,
                    styles["BodyText"]
                )
            )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Verified Web Sources",
            styles["Heading2"]
        )
    )

    for source in sources:

        story.append(
            Paragraph(
                html.escape(source["title"]),
                styles["BodyText"]
            )
        )

        story.append(
            Paragraph(
                html.escape(source["url"]),
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(1, 8)
        )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# START RESEARCH
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

start = st.button(
    "🔎 Start Research",
    type="primary",
    use_container_width=True,
)


if start:

    if not topic.strip():

        st.warning(
            "Please enter a research question first."
        )

        st.stop()

    st.session_state.last_report = ""
    st.session_state.last_sources = []
    st.session_state.last_topic = topic

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    with st.status(
        "🔎 Researching your topic...",
        expanded=True
    ) as status:

        st.write(
            "🌐 Searching the live web..."
        )

        try:

            results = search_web(topic)

        except Exception as e:

            status.update(
                label="❌ Web search failed",
                state="error"
            )

            st.error(str(e))

            st.stop()

        if not results:

            status.update(
                label="❌ No web sources found",
                state="error"
            )

            st.error(
                "No web sources were found. "
                "Try a more specific research question."
            )

            st.stop()

        st.write(
            f"✓ Found {len(results)} web results"
        )

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        st.write(
            "📚 Selecting relevant sources..."
        )

        selected_sources = select_sources(
            results
        )

        st.write(
            f"✓ Selected {len(selected_sources)} sources"
        )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        st.write(
            "📄 Reading source pages..."
        )

        evidence = build_evidence(
            selected_sources
        )

        st.write(
            "✓ Evidence extracted"
        )

        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        st.write(
            "🤖 Running the single CrewAI research agent..."
        )

        try:

            report = run_research(
                topic,
                evidence
            )

        except Exception as e:

            error_text = str(e)

            status.update(
                label="❌ AI analysis failed",
                state="error"
            )

            st.error(
                "The web research completed, but the AI "
                "analysis request was rejected."
            )

            if "413" in error_text or "rate_limit" in error_text:

                st.warning(
                    "Groq rejected the request because of "
                    "the current token-per-minute limit. "
                    "This version already minimizes the "
                    "request, so wait a few seconds and "
                    "try again."
                )

            with st.expander(
                "Technical error"
            ):
                st.code(error_text)

            st.session_state.last_sources = selected_sources

            st.stop()

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        status.update(
            label="✅ Research completed",
            state="complete"
        )

    st.session_state.last_report = report
    st.session_state.last_sources = selected_sources


# ============================================================
# DISPLAY REPORT
# ============================================================

if st.session_state.last_report:

    st.markdown(
        '<div class="report-header">📊 Research Report</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="report-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.last_report
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

if st.session_state.last_sources:

    st.markdown(
        '<div class="report-header">🔗 Verified Web Sources</div>',
        unsafe_allow_html=True
    )

    for source in st.session_state.last_sources:

        title = html.escape(
            source.get("title", "Untitled")
        )

        domain = html.escape(
            source.get("domain", "")
        )

        url = html.escape(
            source.get("url", "")
        )

        snippet = html.escape(
            source.get("snippet", "")
        )

        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-title">
                    {title}
                </div>

                <div class="source-domain">
                    {domain}
                </div>

                <div class="source-url">
                    {url}
                </div>

                <div style="
                    margin-top:10px;
                    color:#94a3b8;
                    font-size:13px;
                    line-height:1.5;
                ">
                    {snippet}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DOWNLOADS
# ============================================================

if st.session_state.last_report:

    st.markdown(
        '<div class="report-header">⬇️ Download Report</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    report_text = (
        f"AI RESEARCH AGENT\n\n"
        f"Research Question:\n"
        f"{st.session_state.last_topic}\n\n"
        f"{st.session_state.last_report}\n\n"
        f"VERIFIED SOURCES\n\n"
    )

    for source in st.session_state.last_sources:

        report_text += (
            f"{source['title']}\n"
            f"{source['url']}\n\n"
        )

    with col1:

        st.download_button(
            "📄 Download TXT",
            data=report_text,
            file_name="research_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:

        try:

            pdf = create_pdf(
                st.session_state.last_topic,
                st.session_state.last_report,
                st.session_state.last_sources,
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
                f"PDF generation unavailable: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
        border-top:1px solid rgba(148,163,184,0.10);
    ">

        AI Research Agent ·
        One CrewAI Agent ·
        Live Web Research ·
        Groq

    </div>
    """,
    unsafe_allow_html=True
)
