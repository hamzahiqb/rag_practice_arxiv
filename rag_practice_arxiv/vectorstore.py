from pathlib import Path

import chromadb

from rag_practice_arxiv.config import CHROMA_DB_PATH
from rag_practice_arxiv.models import Paper


class PaperVectorStore:
    def __init__(self, persist_path: Path | None = None):
        self.persist_path = persist_path or CHROMA_DB_PATH
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        # Uses ChromaDB's default embedding function (all-MiniLM-L6-v2, runs locally)
        self._collection = self._client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"hnsw:space": "cosine"},
        )

    def add_papers(self, papers: list[Paper]) -> int:
        # Filter out papers already in the collection
        new_papers = []
        for paper in papers:
            existing = self._collection.get(ids=[paper.arxiv_id])
            if not existing["ids"]:
                new_papers.append(paper)

        if not new_papers:
            return 0

        self._collection.add(
            ids=[p.arxiv_id for p in new_papers],
            documents=[p.embedding_text for p in new_papers],
            metadatas=[
                {
                    "title": p.title,
                    "authors": ", ".join(p.authors),
                    "categories": ", ".join(p.categories),
                    "published": p.published.isoformat(),
                }
                for p in new_papers
            ],
        )
        return len(new_papers)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, arxiv_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "arxiv_id": arxiv_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
            )
        return output

    def count(self) -> int:
        return self._collection.count()
