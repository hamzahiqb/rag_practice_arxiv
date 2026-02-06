import json
from unittest.mock import MagicMock, patch

from rag_practice_arxiv.models import Paper
from rag_practice_arxiv.ranking import PaperRanker


@patch("rag_practice_arxiv.ranking.ChatAnthropic")
def test_rank_papers(mock_llm_cls, sample_papers: list[Paper]):
    mock_llm = MagicMock()
    scores = [
        {"arxiv_id": p.arxiv_id, "score": 10 - i, "reasoning": f"Reason {i}"} for i, p in enumerate(sample_papers)
    ]
    mock_response = MagicMock()
    mock_response.content = json.dumps(scores)
    mock_llm.invoke.return_value = mock_response
    mock_llm_cls.return_value = mock_llm

    ranker = PaperRanker()
    ranked = ranker.rank_papers("machine learning", sample_papers, top_k=3)

    assert len(ranked) == 3
    assert ranked[0].score >= ranked[1].score >= ranked[2].score
    assert ranked[0].paper.arxiv_id == sample_papers[0].arxiv_id


@patch("rag_practice_arxiv.ranking.ChatAnthropic")
def test_rank_papers_empty(mock_llm_cls):
    mock_llm_cls.return_value = MagicMock()
    ranker = PaperRanker()
    ranked = ranker.rank_papers("query", [])
    assert ranked == []


@patch("rag_practice_arxiv.ranking.ChatAnthropic")
def test_rank_papers_fallback_parsing(mock_llm_cls, sample_papers: list[Paper]):
    mock_llm = MagicMock()
    # Response with markdown code block wrapping
    scores = [{"arxiv_id": p.arxiv_id, "score": 5, "reasoning": "Okay"} for p in sample_papers]
    mock_response = MagicMock()
    mock_response.content = f"```json\n{json.dumps(scores)}\n```"
    mock_llm.invoke.return_value = mock_response
    mock_llm_cls.return_value = mock_llm

    ranker = PaperRanker()
    ranked = ranker.rank_papers("test", sample_papers, top_k=5)

    assert len(ranked) == 5


@patch("rag_practice_arxiv.ranking.ChatAnthropic")
def test_rank_papers_unparseable(mock_llm_cls, sample_papers: list[Paper]):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is not valid JSON at all"
    mock_llm.invoke.return_value = mock_response
    mock_llm_cls.return_value = mock_llm

    ranker = PaperRanker()
    ranked = ranker.rank_papers("test", sample_papers, top_k=5)

    # Fallback should give all papers score 5
    assert len(ranked) == 5
    assert all(r.score == 5 for r in ranked)
