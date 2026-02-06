import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

from rag_practice_arxiv.config import ARXIV_API_BASE, ARXIV_MAX_RESULTS, ARXIV_RATE_LIMIT_SECONDS
from rag_practice_arxiv.models import Frequency, InputType, Paper, SearchRequest

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_last_request_time: float = 0.0

ARXIV_CATEGORIES = {
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.IR",
    "cs.NE",
    "cs.RO",
    "cs.SE",
    "cs.DS",
    "cs.CR",
    "stat.ML",
    "math.OC",
    "eess.SP",
    "q-bio.QM",
    "quant-ph",
    "physics.comp-ph",
}

ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def classify_input(user_input: str) -> InputType:
    cleaned = user_input.strip()
    # Check for arXiv URL
    url_match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", cleaned)
    if cleaned.startswith("http") and url_match:
        return InputType.ARXIV_ID
    # Check for arXiv ID
    if ARXIV_ID_PATTERN.match(cleaned):
        return InputType.ARXIV_ID
    # Check for category
    if cleaned in ARXIV_CATEGORIES or re.match(r"^[a-z-]+\.[A-Z]{2}$", cleaned):
        return InputType.CATEGORY
    return InputType.KEYWORDS


def extract_arxiv_id(user_input: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", user_input.strip())
    if match:
        return match.group(1)
    return user_input.strip()


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < ARXIV_RATE_LIMIT_SECONDS:
        time.sleep(ARXIV_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _build_query(request: SearchRequest) -> str:
    params: dict[str, str | int] = {"max_results": min(request.max_results, ARXIV_MAX_RESULTS)}

    if request.input_type == InputType.ARXIV_ID:
        arxiv_id = extract_arxiv_id(request.query)
        params["id_list"] = arxiv_id
    elif request.input_type == InputType.CATEGORY:
        params["search_query"] = f"cat:{request.query}"
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"
    else:
        params["search_query"] = f"all:{request.query}"
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"

    return f"{ARXIV_API_BASE}?{urllib.parse.urlencode(params)}"


def _parse_entry(entry: ET.Element) -> Paper | None:
    title_el = entry.find(f"{ATOM_NS}title")
    abstract_el = entry.find(f"{ATOM_NS}summary")
    published_el = entry.find(f"{ATOM_NS}published")
    updated_el = entry.find(f"{ATOM_NS}updated")
    id_el = entry.find(f"{ATOM_NS}id")

    if title_el is None or abstract_el is None or id_el is None:
        return None

    raw_id = id_el.text or ""
    arxiv_id_match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", raw_id)
    if not arxiv_id_match:
        return None
    arxiv_id = arxiv_id_match.group(1)

    title = " ".join((title_el.text or "").split())
    abstract = " ".join((abstract_el.text or "").strip().split())

    authors = []
    for author_el in entry.findall(f"{ATOM_NS}author"):
        name_el = author_el.find(f"{ATOM_NS}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text)

    categories = []
    for cat_el in entry.findall(f"{ATOM_NS}category"):
        term = cat_el.get("term")
        if term:
            categories.append(term)

    pdf_url = ""
    for link_el in entry.findall(f"{ATOM_NS}link"):
        if link_el.get("title") == "pdf":
            pdf_url = link_el.get("href", "")
            break

    published = datetime.fromisoformat((published_el.text or "").replace("Z", "+00:00"))
    updated = datetime.fromisoformat((updated_el.text or "").replace("Z", "+00:00"))

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
    )


def _get_date_cutoff(frequency: Frequency) -> datetime:
    now = datetime.now(UTC)
    if frequency == Frequency.DAILY:
        return now - timedelta(days=1)
    return now - timedelta(days=7)


def _filter_by_date(papers: list[Paper], frequency: Frequency) -> list[Paper]:
    cutoff = _get_date_cutoff(frequency)
    return [p for p in papers if p.published >= cutoff]


def fetch_papers(request: SearchRequest) -> list[Paper]:
    url = _build_query(request)
    _rate_limit()

    req = urllib.request.Request(url, headers={"User-Agent": "rag-practice-arxiv/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper)

    # Don't date-filter single paper lookups
    if request.input_type == InputType.ARXIV_ID:
        return papers

    return _filter_by_date(papers, request.frequency)
