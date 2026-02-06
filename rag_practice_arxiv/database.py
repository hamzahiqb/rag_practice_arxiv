import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from rag_practice_arxiv.config import SQLITE_DB_PATH
from rag_practice_arxiv.models import Paper

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

UPSERT_SQL = """
INSERT INTO papers (arxiv_id, title, abstract, authors, categories, published, updated, pdf_url)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(arxiv_id) DO UPDATE SET
    title = excluded.title,
    abstract = excluded.abstract,
    authors = excluded.authors,
    categories = excluded.categories,
    updated = excluded.updated,
    pdf_url = excluded.pdf_url
"""


class PaperDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def upsert_paper(self, paper: Paper) -> None:
        with self._connect() as conn:
            conn.execute(
                UPSERT_SQL,
                (
                    paper.arxiv_id,
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.authors),
                    json.dumps(paper.categories),
                    paper.published.isoformat(),
                    paper.updated.isoformat(),
                    paper.pdf_url,
                ),
            )

    def upsert_papers(self, papers: list[Paper]) -> int:
        with self._connect() as conn:
            for paper in papers:
                conn.execute(
                    UPSERT_SQL,
                    (
                        paper.arxiv_id,
                        paper.title,
                        paper.abstract,
                        json.dumps(paper.authors),
                        json.dumps(paper.categories),
                        paper.published.isoformat(),
                        paper.updated.isoformat(),
                        paper.pdf_url,
                    ),
                )
        return len(papers)

    def get_paper(self, arxiv_id: str) -> Paper | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def get_all_papers(self) -> list[Paper]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM papers ORDER BY published DESC").fetchall()
        return [self._row_to_paper(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
        return result[0] if result else 0

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        return Paper(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors"]),
            categories=json.loads(row["categories"]),
            published=datetime.fromisoformat(row["published"]).replace(tzinfo=UTC),
            updated=datetime.fromisoformat(row["updated"]).replace(tzinfo=UTC),
            pdf_url=row["pdf_url"],
        )
