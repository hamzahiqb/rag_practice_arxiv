from unittest.mock import patch

from rag_practice_arxiv.agent import (
    build_request_node,
    classify_input_node,
    fetch_papers_node,
    has_papers,
)


def test_classify_input_node_keywords():
    state = {"user_input": "transformer attention"}
    result = classify_input_node(state)
    assert result["input_type"] == "keywords"


def test_classify_input_node_category():
    state = {"user_input": "cs.AI"}
    result = classify_input_node(state)
    assert result["input_type"] == "category"


def test_classify_input_node_arxiv_id():
    state = {"user_input": "2401.12345"}
    result = classify_input_node(state)
    assert result["input_type"] == "arxiv_id"


def test_build_request_node():
    state = {"user_input": "attention", "input_type": "keywords", "frequency": "daily"}
    result = build_request_node(state)
    assert result["search_request"] is not None
    assert result["search_request"].query == "attention"
    assert result["search_request"].frequency.value == "daily"


def test_has_papers_with_papers(sample_papers):
    state = {"papers": sample_papers}
    assert has_papers(state) == "store"


def test_has_papers_empty():
    state = {"papers": []}
    assert has_papers(state) == "end"


@patch("rag_practice_arxiv.agent.fetch_papers")
def test_fetch_papers_node(mock_fetch, sample_papers):
    from rag_practice_arxiv.models import Frequency, InputType, SearchRequest

    request = SearchRequest(query="test", input_type=InputType.KEYWORDS, frequency=Frequency.WEEKLY)
    mock_fetch.return_value = sample_papers

    state = {"search_request": request}
    result = fetch_papers_node(state)

    assert len(result["papers"]) == 5
    assert "Fetched 5 papers" in result["status"]


def test_fetch_papers_node_no_request():
    state = {"search_request": None}
    result = fetch_papers_node(state)
    assert result["papers"] == []
