import json
import re

from langchain_anthropic import ChatAnthropic

from rag_practice_arxiv.config import ANTHROPIC_API_KEY, LLM_MODEL, RANKING_BATCH_SIZE, RANKING_TOP_K
from rag_practice_arxiv.models import Paper, RankedPaper

RANKING_PROMPT = """You are a research paper ranking assistant. \
Given a search query and a batch of arXiv papers, score each paper on relevance from 0 to 10.

Search query: {query}

Papers:
{papers_text}

Return a JSON array of objects with "arxiv_id", "score" (0-10 integer), and "reasoning" (1 sentence).
Only return the JSON array, no other text."""


class PaperRanker:
    def __init__(self, model: str | None = None):
        self._llm = ChatAnthropic(
            model=model or LLM_MODEL,
            temperature=0.0,
            api_key=ANTHROPIC_API_KEY,
        )

    def _format_papers(self, papers: list[Paper]) -> str:
        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. [{p.arxiv_id}] {p.title}")
            lines.append(f"   Abstract: {p.abstract[:300]}...")
            lines.append(f"   Categories: {', '.join(p.categories)}")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(self, response_text: str, papers: list[Paper]) -> list[dict]:
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from markdown code block or surrounding text
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return all papers with score 5
        return [{"arxiv_id": p.arxiv_id, "score": 5, "reasoning": "Could not parse LLM response"} for p in papers]

    def _rank_batch(self, query: str, papers: list[Paper]) -> list[dict]:
        papers_text = self._format_papers(papers)
        prompt = RANKING_PROMPT.format(query=query, papers_text=papers_text)

        response = self._llm.invoke(prompt)
        return self._parse_response(response.content, papers)

    def rank_papers(self, query: str, papers: list[Paper], top_k: int = RANKING_TOP_K) -> list[RankedPaper]:
        if not papers:
            return []

        paper_map = {p.arxiv_id: p for p in papers}

        # Batch papers if there are too many
        if len(papers) <= RANKING_BATCH_SIZE:
            all_scores = self._rank_batch(query, papers)
        else:
            # Process in batches, collect top candidates
            candidates: list[dict] = []
            for i in range(0, len(papers), RANKING_BATCH_SIZE):
                batch = papers[i : i + RANKING_BATCH_SIZE]
                batch_scores = self._rank_batch(query, batch)
                # Keep top from each batch
                batch_scores.sort(key=lambda x: x.get("score", 0), reverse=True)
                candidates.extend(batch_scores[: top_k * 2])

            # Final ranking pass on candidates
            candidate_papers = [paper_map[s["arxiv_id"]] for s in candidates if s["arxiv_id"] in paper_map]
            if len(candidate_papers) > RANKING_BATCH_SIZE:
                all_scores = self._rank_batch(query, candidate_papers[:RANKING_BATCH_SIZE])
            else:
                all_scores = self._rank_batch(query, candidate_papers) if candidate_papers else candidates

        # Sort by score descending
        all_scores.sort(key=lambda x: x.get("score", 0), reverse=True)

        ranked = []
        for score_data in all_scores[:top_k]:
            arxiv_id = score_data.get("arxiv_id", "")
            if arxiv_id in paper_map:
                ranked.append(
                    RankedPaper(
                        paper=paper_map[arxiv_id],
                        score=float(score_data.get("score", 0)),
                        reasoning=score_data.get("reasoning", ""),
                    )
                )
        return ranked
