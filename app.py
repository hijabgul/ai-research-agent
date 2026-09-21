%%writefile app.py

import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchAI",
    page_icon="🔎",
    layout="wide"
)


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

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .research-box {
        background-color: #111827;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #263244;
        margin-top: 20px;
    }

    .source-box {
        background-color: #111827;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #263244;
        margin-bottom: 10px;
    }

    .source-url {
        color: #818CF8;
        word-break: break-all;
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
    'Evidence-based AI research using live web sources, '
    'source extraction, and CrewAI.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GROQ API KEY
# ============================================================

if "GROQ_API_KEY" not in st.secrets:

    st.error(
        "GROQ_API_KEY was not found in Streamlit Secrets. "
        "Please add it in your Streamlit Cloud Secrets."
    )

    st.stop()


groq_api_key = st.secrets["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"] = groq_api_key


# ============================================================
# GROQ MODEL
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"

crew_llm = LLM(
    model=GROQ_MODEL,
    api_key=groq_api_key,
    temperature=0.2
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

        # Remove unnecessary webpage elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):

            tag.decompose()

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
# WEB RESEARCH
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

        page = extract_webpage(url)

        source = {
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
            "content": page.get(
                "text",
                ""
            )
            if page.get("success")
            else ""
        }

        sources.append(source)

    return sources


# ============================================================
# CREWAI TOOL INPUT
# ============================================================

class ResearchSearchInput(BaseModel):

    query: str = Field(
        ...,
        description="The web research query to search for."
    )


# ============================================================
# CREWAI CUSTOM TOOL
# ============================================================

class ResearchSearchTool(BaseTool):

    name: str = "web_research_search"

    description: str = (
        "Search the live web using DDGS and read accessible "
        "web pages. Returns titles, original URLs, snippets, "
        "domains, and extracted webpage content."
    )

    args_schema: type[BaseModel] = ResearchSearchInput

    def _run(
        self,
        query: str
    ) -> str:

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

ACCESSIBLE:
{source["accessible"]}

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
    Conduct detailed, current, evidence-based research on the
    user's research question.

    Search the live web, inspect actual source pages, prioritize
    authoritative and original sources, compare evidence,
    identify conflicting information, and produce a detailed
    research report.

    Never invent sources, URLs, statistics, quotations, facts,
    or evidence.
    """,

    backstory="""
    You are a meticulous professional research analyst.

    You understand that an AI model's internal knowledge should
    not be treated as sufficient evidence for current factual
    claims.

    You therefore use live web research and inspect source
    material whenever possible.

    You distinguish documented facts from interpretation.

    When reliable sources disagree, you clearly explain the
    disagreement.

    When sufficient evidence cannot be found, you explicitly
    say so rather than inventing an answer.

    Every important factual claim should be traceable to a source.
    """,

    tools=[
        research_tool
    ],

    llm=crew_llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# RESEARCH TASK
# ============================================================

research_task = Task(

    description="""

    Research the following question:

    {topic}

    Follow these requirements:

    1. Understand the research question.

    2. Break it into important sub-questions when necessary.

    3. Perform multiple focused web searches.

    4. Use the web_research_search tool.

    5. Prefer primary and authoritative sources.

    6. Read actual webpage content rather than relying only
       on search snippets.

    7. For current questions, prioritize recent information.

    8. Cross-check important factual claims.

    9. Identify meaningful disagreements between sources.

    10. Never invent facts, sources, URLs, statistics,
        quotations, or evidence.

    11. Do not claim a source supports something unless its
        retrieved content supports the claim.

    12. If reliable evidence is insufficient, say so clearly.

    13. Include original source URLs.

    14. Distinguish facts from interpretation.

    15. Explain important limitations.

    Produce a detailed report containing:

    - Executive Summary
    - Research Question
    - Key Findings
    - Background
    - Detailed Analysis
    - Evidence
    - Conflicting Information
    - Limitations
    - Conclusion
    - Sources

    Important factual claims must be traceable to retrieved
    sources.
    """,

    expected_output="""
    A detailed evidence-based research report containing:

    1. Executive Summary
    2. Research Question
    3. Key Findings
    4. Background
    5. Detailed Analysis
    6. Evidence
    7. Conflicting Information
    8. Limitations
    9. Conclusion
    10. Numbered Sources with original URLs

    Do not fabricate information.
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

    verbose=True
)


# ============================================================
# STREAMLIT INPUT
# ============================================================

st.markdown(
    '<div class="research-box">',
    unsafe_allow_html=True
)

st.subheader(
    "What do you want to research?"
)

topic = st.text_area(
    "Research question",
    placeholder=(
        "Example: What are the major applications of "
        "artificial intelligence in education in 2026?"
    ),
    height=130
)

research_button = st.button(
    "🔎 Start Research",
    type="primary",
    use_container_width=True
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:

    if not topic.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        with st.status(
            "Researching the web and analyzing sources...",
            expanded=True
        ):

            try:

                result = research_crew.kickoff(
                    inputs={
                        "topic": topic.strip()
                    }
                )

                st.session_state[
                    "research_result"
                ] = str(result)

            except Exception as e:

                st.error(
                    f"Research failed: {str(e)}"
                )


# ============================================================
# DISPLAY RESULT
# ============================================================

if "research_result" in st.session_state:

    st.markdown("---")

    st.subheader(
        "📄 Research Report"
    )

    st.markdown(
        st.session_state["research_result"]
    )
