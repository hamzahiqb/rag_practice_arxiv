import argparse
import sys

from rag_practice_arxiv.agent import run_search
from rag_practice_arxiv.database import PaperDatabase
from rag_practice_arxiv.models import RankedPaper
from rag_practice_arxiv.vectorstore import PaperVectorStore


def _print_ranked_papers(ranked_papers: list[RankedPaper]) -> None:
    if not ranked_papers:
        print("No papers to display.")
        return

    print(f"\nTop {len(ranked_papers)} papers:\n")
    for i, rp in enumerate(ranked_papers, 1):
        print(f"  {i}. [{rp.score:.0f}/10] {rp.paper.title}")
        print(f"     ID: {rp.paper.arxiv_id}")
        print(f"     Authors: {', '.join(rp.paper.authors[:3])}")
        print(f"     Categories: {', '.join(rp.paper.categories)}")
        print(f"     Reasoning: {rp.reasoning}")
        print(f"     PDF: {rp.paper.pdf_url}")
        print()


def cmd_search(args: argparse.Namespace) -> None:
    print(f"Searching for: {args.query}")
    print(f"Frequency: {args.frequency}")
    print()

    result = run_search(args.query, frequency=args.frequency)

    papers = result.get("papers", [])
    print(f"Found {len(papers)} papers")

    papers_stored = result.get("papers_stored", 0)
    vectors_stored = result.get("vectors_stored", 0)
    if papers:
        print(f"Stored: {papers_stored} in DB, {vectors_stored} new vectors")

    ranked = result.get("ranked_papers", [])
    _print_ranked_papers(ranked)


def cmd_stats(args: argparse.Namespace) -> None:
    db = PaperDatabase()
    print(f"SQLite papers: {db.count()}")

    try:
        vs = PaperVectorStore()
        print(f"ChromaDB vectors: {vs.count()}")
    except Exception as e:
        print(f"ChromaDB: unavailable ({e})")


def cmd_query(args: argparse.Namespace) -> None:
    try:
        vs = PaperVectorStore()
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        sys.exit(1)

    results = vs.query(args.query, top_k=args.top_k)

    if not results:
        print("No matching papers found.")
        return

    print(f"\nTop {len(results)} semantic matches for: '{args.query}'\n")
    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {})
        print(f"  {i}. {metadata.get('title', 'Unknown')}")
        print(f"     ID: {r['arxiv_id']}")
        print(f"     Distance: {r['distance']:.4f}")
        print(f"     Categories: {metadata.get('categories', '')}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="arXiv RAG Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search
    search_parser = subparsers.add_parser("search", help="Search arXiv for papers")
    search_parser.add_argument("query", help="Search query, category, or arXiv ID")
    search_parser.add_argument("-f", "--frequency", default="weekly", choices=["daily", "weekly"])
    search_parser.set_defaults(func=cmd_search)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # query
    query_parser = subparsers.add_parser("query", help="Semantic search over stored papers")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args(argv)
    args.func(args)
