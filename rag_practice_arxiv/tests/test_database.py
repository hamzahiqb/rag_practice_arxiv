from pathlib import Path

from rag_practice_arxiv.database import PaperDatabase
from rag_practice_arxiv.models import Paper


def test_upsert_and_get(tmp_db_path: Path, sample_paper: Paper):
    db = PaperDatabase(db_path=tmp_db_path)
    db.upsert_paper(sample_paper)

    retrieved = db.get_paper("2401.12345")
    assert retrieved is not None
    assert retrieved.arxiv_id == sample_paper.arxiv_id
    assert retrieved.title == sample_paper.title
    assert retrieved.authors == sample_paper.authors
    assert retrieved.categories == sample_paper.categories


def test_upsert_papers_batch(tmp_db_path: Path, sample_papers: list[Paper]):
    db = PaperDatabase(db_path=tmp_db_path)
    count = db.upsert_papers(sample_papers)
    assert count == 5
    assert db.count() == 5


def test_upsert_deduplication(tmp_db_path: Path, sample_paper: Paper):
    db = PaperDatabase(db_path=tmp_db_path)
    db.upsert_paper(sample_paper)
    db.upsert_paper(sample_paper)
    assert db.count() == 1


def test_get_nonexistent(tmp_db_path: Path):
    db = PaperDatabase(db_path=tmp_db_path)
    assert db.get_paper("9999.99999") is None


def test_get_all_papers(tmp_db_path: Path, sample_papers: list[Paper]):
    db = PaperDatabase(db_path=tmp_db_path)
    db.upsert_papers(sample_papers)
    all_papers = db.get_all_papers()
    assert len(all_papers) == 5


def test_count_empty(tmp_db_path: Path):
    db = PaperDatabase(db_path=tmp_db_path)
    assert db.count() == 0
