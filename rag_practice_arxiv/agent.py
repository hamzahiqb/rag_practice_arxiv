from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag_practice_arxiv.arxiv_client import classify_input, fetch_papers
from rag_practice_arxiv.database import PaperDatabase
from rag_practice_arxiv.models import Frequency, Paper, RankedPaper, SearchRequest
from rag_practice_arxiv.ranking import PaperRanker
from rag_practice_arxiv.vectorstore import PaperVectorStore


class GraphState(TypedDict, total=False):
    user_input: str
    frequency: str
    input_type: str
    search_request: SearchRequest | None
    papers: list[Paper]
    ranked_papers: list[RankedPaper]
    status: str
    papers_stored: int
    vectors_stored: int


def classify_input_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    input_type = classify_input(user_input)
    return {"input_type": input_type.value}


def build_request_node(state: GraphState) -> GraphState:
    from rag_practice_arxiv.models import InputType

    request = SearchRequest(
        query=state["user_input"],
        input_type=InputType(state["input_type"]),
        frequency=Frequency(state.get("frequency", "weekly")),
    )
    return {"search_request": request}


def fetch_papers_node(state: GraphState) -> GraphState:
    request = state.get("search_request")
    if request is None:
        return {"papers": [], "status": "No search request"}

    papers = fetch_papers(request)
    return {"papers": papers, "status": f"Fetched {len(papers)} papers"}


def has_papers(state: GraphState) -> str:
    papers = state.get("papers", [])
    return "store" if papers else "end"


def store_papers_node(state: GraphState) -> GraphState:
    papers = state.get("papers", [])
    if not papers:
        return {"papers_stored": 0, "vectors_stored": 0}

    db = PaperDatabase()
    papers_stored = db.upsert_papers(papers)

    vs = PaperVectorStore()
    vectors_stored = vs.add_papers(papers)

    return {
        "papers_stored": papers_stored,
        "vectors_stored": vectors_stored,
        "status": f"Stored {papers_stored} papers, {vectors_stored} new vectors",
    }


def rank_papers_node(state: GraphState) -> GraphState:
    papers = state.get("papers", [])
    user_input = state.get("user_input", "")

    if not papers:
        return {"ranked_papers": []}

    ranker = PaperRanker()
    ranked = ranker.rank_papers(user_input, papers)
    return {"ranked_papers": ranked, "status": f"Ranked top {len(ranked)} papers"}


def build_agent() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_input", classify_input_node)
    workflow.add_node("build_request", build_request_node)
    workflow.add_node("fetch_papers", fetch_papers_node)
    workflow.add_node("store_papers", store_papers_node)
    workflow.add_node("rank_papers", rank_papers_node)

    workflow.set_entry_point("classify_input")
    workflow.add_edge("classify_input", "build_request")
    workflow.add_edge("build_request", "fetch_papers")
    workflow.add_conditional_edges("fetch_papers", has_papers, {"store": "store_papers", "end": END})
    workflow.add_edge("store_papers", "rank_papers")
    workflow.add_edge("rank_papers", END)

    return workflow.compile()


def run_search(user_input: str, frequency: str = "weekly") -> GraphState:
    agent = build_agent()
    result = agent.invoke({"user_input": user_input, "frequency": frequency})
    return result
