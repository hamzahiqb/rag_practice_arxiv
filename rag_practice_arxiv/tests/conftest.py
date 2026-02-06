from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_practice_arxiv.models import Paper


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2401.12345",
        title="Attention Is All You Need Revisited",
        abstract="We revisit the transformer architecture and propose improvements.",
        authors=["Alice Smith", "Bob Jones"],
        categories=["cs.CL", "cs.AI"],
        published=datetime(2024, 1, 15, tzinfo=UTC),
        updated=datetime(2024, 1, 16, tzinfo=UTC),
        pdf_url="http://arxiv.org/pdf/2401.12345v1",
    )


@pytest.fixture
def sample_papers() -> list[Paper]:
    return [
        Paper(
            arxiv_id=f"2401.{10000 + i}",
            title=f"Paper Title {i}",
            abstract=f"This is the abstract for paper {i} about machine learning topics.",
            authors=[f"Author {i}"],
            categories=["cs.LG"],
            published=datetime(2024, 1, i + 1, tzinfo=UTC),
            updated=datetime(2024, 1, i + 1, tzinfo=UTC),
            pdf_url=f"http://arxiv.org/pdf/2401.{10000 + i}v1",
        )
        for i in range(5)
    ]


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_papers.db"


@pytest.fixture
def tmp_chroma_path(tmp_path: Path) -> Path:
    return tmp_path / "test_chroma"
