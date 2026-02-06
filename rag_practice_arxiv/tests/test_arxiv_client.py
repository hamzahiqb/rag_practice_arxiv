import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from rag_practice_arxiv.arxiv_client import (
    _build_query,
    _filter_by_date,
    _parse_entry,
    classify_input,
    extract_arxiv_id,
)
from rag_practice_arxiv.models import Frequency, InputType, SearchRequest

SAMPLE_ENTRY_XML = """
<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/2401.12345v1</id>
  <title>Attention Is All You Need Revisited</title>
  <summary>We revisit the transformer architecture.</summary>
  <published>2024-01-15T00:00:00Z</published>
  <updated>2024-01-16T00:00:00Z</updated>
  <author><name>Alice Smith</name></author>
  <author><name>Bob Jones</name></author>
  <category term="cs.CL"/>
  <category term="cs.AI"/>
  <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
</entry>
"""


def test_classify_input_keywords():
    assert classify_input("transformer attention") == InputType.KEYWORDS
    assert classify_input("reinforcement learning") == InputType.KEYWORDS


def test_classify_input_category():
    assert classify_input("cs.AI") == InputType.CATEGORY
    assert classify_input("cs.CL") == InputType.CATEGORY
    assert classify_input("stat.ML") == InputType.CATEGORY


def test_classify_input_arxiv_id():
    assert classify_input("2401.12345") == InputType.ARXIV_ID
    assert classify_input("2401.12345v2") == InputType.ARXIV_ID


def test_classify_input_arxiv_url():
    assert classify_input("https://arxiv.org/abs/2401.12345") == InputType.ARXIV_ID
    assert classify_input("https://arxiv.org/pdf/2401.12345v1") == InputType.ARXIV_ID


def test_extract_arxiv_id():
    assert extract_arxiv_id("2401.12345") == "2401.12345"
    assert extract_arxiv_id("2401.12345v2") == "2401.12345"
    assert extract_arxiv_id("https://arxiv.org/abs/2401.12345v1") == "2401.12345"


def test_parse_entry():
    entry = ET.fromstring(SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper is not None
    assert paper.arxiv_id == "2401.12345"
    assert paper.title == "Attention Is All You Need Revisited"
    assert paper.abstract == "We revisit the transformer architecture."
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.categories == ["cs.CL", "cs.AI"]
    assert paper.published == datetime(2024, 1, 15, tzinfo=UTC)
    assert paper.pdf_url == "http://arxiv.org/pdf/2401.12345v1"


def test_build_query_keywords():
    request = SearchRequest(query="attention", input_type=InputType.KEYWORDS)
    url = _build_query(request)
    assert "search_query=all%3Aattention" in url
    assert "sortBy=submittedDate" in url


def test_build_query_category():
    request = SearchRequest(query="cs.AI", input_type=InputType.CATEGORY)
    url = _build_query(request)
    assert "search_query=cat%3Acs.AI" in url


def test_build_query_arxiv_id():
    request = SearchRequest(query="2401.12345", input_type=InputType.ARXIV_ID)
    url = _build_query(request)
    assert "id_list=2401.12345" in url


def test_filter_by_date(sample_papers):
    # All sample papers are from Jan 2024, so they should all be filtered out
    # with a daily or weekly window relative to "now"
    filtered = _filter_by_date(sample_papers, Frequency.DAILY)
    assert len(filtered) == 0

    filtered = _filter_by_date(sample_papers, Frequency.WEEKLY)
    assert len(filtered) == 0
