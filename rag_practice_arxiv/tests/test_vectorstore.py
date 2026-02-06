from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_practice_arxiv.models import Paper
from rag_practice_arxiv.vectorstore import PaperVectorStore


@patch("rag_practice_arxiv.vectorstore.chromadb.PersistentClient")
def test_add_papers(mock_chroma_client, tmp_chroma_path: Path, sample_papers: list[Paper]):
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    mock_collection.count.return_value = 0
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection

    store = PaperVectorStore(persist_path=tmp_chroma_path)
    count = store.add_papers(sample_papers)

    assert count == len(sample_papers)
    mock_collection.add.assert_called_once()


@patch("rag_practice_arxiv.vectorstore.chromadb.PersistentClient")
def test_add_papers_dedup(mock_chroma_client, tmp_chroma_path: Path, sample_paper: Paper):
    mock_collection = MagicMock()
    # Simulate paper already exists
    mock_collection.get.return_value = {"ids": [sample_paper.arxiv_id]}
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection

    store = PaperVectorStore(persist_path=tmp_chroma_path)
    count = store.add_papers([sample_paper])

    assert count == 0
    mock_collection.add.assert_not_called()


@patch("rag_practice_arxiv.vectorstore.chromadb.PersistentClient")
def test_query(mock_chroma_client, tmp_chroma_path: Path):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "ids": [["2401.10000", "2401.10001"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"title": "Paper 0"}, {"title": "Paper 1"}]],
        "distances": [[0.1, 0.2]],
    }
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection

    store = PaperVectorStore(persist_path=tmp_chroma_path)
    results = store.query("attention mechanisms", top_k=5)

    assert len(results) == 2
    assert results[0]["arxiv_id"] == "2401.10000"
    assert results[1]["arxiv_id"] == "2401.10001"


@patch("rag_practice_arxiv.vectorstore.chromadb.PersistentClient")
def test_count(mock_chroma_client, tmp_chroma_path: Path):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection

    store = PaperVectorStore(persist_path=tmp_chroma_path)
    assert store.count() == 42
