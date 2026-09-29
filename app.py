import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import build_rag_graph

# Global compiled graph holder
rag_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler: initializes the RAG workflow graph on startup if keys are present."""
    global rag_graph
    try:
        settings.validate_api_keys()
        print(f"[FastAPI] Initializing LangGraph RAG with index '{settings.pinecone_index_name}'...")
        rag_graph = build_rag_graph()
        print("[FastAPI] LangGraph RAG initialized successfully.")
    except Exception as e:
        print(f"[FastAPI Startup Warning] Could not pre-initialize RAG graph: {e}")
        print("[FastAPI Startup Warning] Ensure OPENAI_API_KEY and PINECONE_API_KEY are configured in .env.")
    yield


app = FastAPI(
    title="Agentic AI RAG API",
    description=(
        "Retrieval-Augmented Generation (RAG) API answering strictly from the Agentic AI eBook "
        "using LangGraph, Pinecone, and OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for external frontends and Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The question to ask based on the Agentic AI eBook",
        example="What is Agentic AI according to the eBook?"
    )


class QueryResponse(BaseModel):
    final_answer: str = Field(
        ...,
        description="The strictly grounded answer generated from retrieved context"
    )
    retrieved_context: List[str] = Field(
        ...,
        description="List of text chunks retrieved from Pinecone used to formulate the answer"
    )
    confidence_score: float = Field(
        ...,
        description="Confidence / relevance score between 0.0 and 1.0"
    )
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Source metadata (page numbers and chunk scores)"
    )


class HealthResponse(BaseModel):
    status: str
    pinecone_index: str
    openai_model: str
    api_keys_configured: bool


@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": "Agentic AI RAG Chatbot API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint to inspect configuration and status."""
    keys_configured = bool(
        settings.openai_api_key
        and settings.pinecone_api_key
        and settings.openai_api_key != "your_openai_api_key_here"
        and settings.pinecone_api_key != "your_pinecone_api_key_here"
    )
    return HealthResponse(
        status="healthy" if keys_configured else "misconfigured",
        pinecone_index=settings.pinecone_index_name,
        openai_model=settings.openai_model,
        api_keys_configured=keys_configured
    )


@app.post("/chat", response_model=QueryResponse, tags=["RAG"])
async def chat_endpoint(request: QueryRequest):
    """
    Main RAG endpoint:
    1. Executes LangGraph workflow (retrieve -> generate).
    2. Strictly answers questions grounded on retrieved document context.
    3. Returns final answer, retrieved context chunks, and confidence score.
    """
    global rag_graph

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty."
        )

    # Initialize graph if not already cached
    if rag_graph is None:
        try:
            rag_graph = build_rag_graph()
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize RAG graph: {str(err)}"
            )

    try:
        initial_state = {
            "question": request.query.strip(),
            "context": [],
            "answer": "",
            "score": 0.0,
            "sources": []
        }
        result = rag_graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result.get("answer", ""),
            retrieved_context=result.get("context", []),
            confidence_score=result.get("score", 0.0),
            sources=result.get("sources")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG workflow: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
