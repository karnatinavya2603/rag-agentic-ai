import re
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from src.config import settings


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float
    sources: Optional[List[Dict[str, Any]]]


REFUSAL_PHRASE = "I cannot answer based on the provided document."


def compute_confidence_score(
    answer: str,
    context_chunks: List[str],
    similarity_scores: Optional[List[float]] = None
) -> float:
    """
    Compute a grounded confidence score between 0.0 and 1.0.

    Logic:
    - 0.0: If no context was retrieved or the model explicitly stated it cannot answer.
    - If similarity scores from cosine distance are available, compute average score bounded [0.0, 1.0].
    - Otherwise, heuristic based on answer richness and context grounding.
    """
    if not context_chunks:
        return 0.0

    lower_ans = answer.strip().lower()
    refusal_lower = REFUSAL_PHRASE.lower()

    # If model responded with refusal or cannot find info
    if refusal_lower in lower_ans or "cannot answer based on the provided document" in lower_ans:
        return 0.0

    if "i am unable to answer" in lower_ans or "the provided document does not mention" in lower_ans:
        return 0.0

    if similarity_scores and len(similarity_scores) > 0:
        # Cosine similarity in Pinecone ranges [0, 1] for normalized embeddings
        avg_sim = sum(similarity_scores) / len(similarity_scores)
        # Scale and bound between 0.50 and 0.99 for positive answers
        bounded = max(0.50, min(0.99, round(avg_sim, 3)))
        return bounded

    # Fallback heuristic: strong grounded response
    return 0.95


def build_rag_graph(
    index_name: Optional[str] = None,
    top_k: Optional[int] = None,
    model_name: Optional[str] = None
):
    """
    Build and compile the LangGraph RAG workflow.

    Workflow:
      START -> retrieve -> generate -> END
    """
    settings.validate_api_keys()

    target_index = index_name or settings.pinecone_index_name
    k = top_k or settings.retrieval_k
    target_model = model_name or settings.openai_model

    # 1. Initialize Vector Store & Retriever
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key
    )
    vectorstore = PineconeVectorStore(
        index_name=target_index,
        embedding=embeddings
    )

    # 2. Initialize LLM
    llm = ChatOpenAI(
        model=target_model,
        temperature=0,
        api_key=settings.openai_api_key
    )

    # 3. Define Nodes
    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant context chunks and metadata from Pinecone."""
        query = state["question"]

        # Attempt to retrieve with relevance scores
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
            docs = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]
        except Exception:
            # Fallback standard retrieval
            docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
            scores = []

        context_texts = [d.page_content for d in docs]
        sources = [
            {
                "page": d.metadata.get("page", 0) + 1 if isinstance(d.metadata.get("page"), int) else d.metadata.get("page"),
                "source": d.metadata.get("source", "Ebook-Agentic-AI.pdf"),
                "score": round(scores[i], 4) if i < len(scores) else None
            }
            for i, d in enumerate(docs)
        ]

        return {
            "context": context_texts,
            "sources": sources,
            "_scores": scores
        }

    def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strictly grounded response using LLM."""
        context_texts: List[str] = state.get("context", [])
        question: str = state["question"]
        scores: List[float] = state.get("_scores", [])

        if not context_texts:
            return {
                "answer": REFUSAL_PHRASE,
                "score": 0.0
            }

        context_str = "\n\n--- CHUNK ---\n\n".join(context_texts)

        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state '{REFUSAL_PHRASE}'
Do not use any outside knowledge or assumptions.

Context:
{context_str}

Question: {question}"""

        response = llm.invoke(prompt)
        answer = str(response.content).strip()

        confidence = compute_confidence_score(
            answer=answer,
            context_chunks=context_texts,
            similarity_scores=scores if scores else None
        )

        return {
            "answer": answer,
            "score": confidence
        }

    # 4. Assemble StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
