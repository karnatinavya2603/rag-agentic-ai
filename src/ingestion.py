import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import settings


GOOGLE_DRIVE_EBOOK_URL = "https://drive.google.com/uc?export=download&id=15VLphKcY23_fpYxN62UEQRri_psRVfP9"


def ensure_pdf_exists(pdf_path: Optional[str] = None) -> Path:
    """Ensure the target PDF exists locally, downloading if necessary."""
    target_path = Path(pdf_path) if pdf_path else settings.default_pdf_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists() or target_path.stat().st_size == 0:
        print(f"[Ingestion] Source PDF not found at {target_path}. Downloading from Google Drive...")
        response = requests.get(GOOGLE_DRIVE_EBOOK_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[Ingestion] Downloaded eBook to {target_path} ({target_path.stat().st_size} bytes)")
    else:
        print(f"[Ingestion] Found existing PDF at {target_path} ({target_path.stat().st_size} bytes)")

    return target_path


def ensure_pinecone_index(
    index_name: str,
    dimension: int = 1536,
    metric: str = "cosine"
) -> None:
    """Ensure the Pinecone index exists; create it with serverless spec if missing."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"[Ingestion] Index '{index_name}' does not exist. Creating serverless index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region
            )
        )
        # Wait for index ready
        print(f"[Ingestion] Waiting for index '{index_name}' to become ready...")
        while not pc.describe_index(index_name).status.get("ready", False):
            time.sleep(2)
        print(f"[Ingestion] Index '{index_name}' is ready.")
    else:
        print(f"[Ingestion] Index '{index_name}' already exists.")


def run_ingestion(
    pdf_path: Optional[str] = None,
    index_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    batch_size: int = 100
) -> PineconeVectorStore:
    """Run full ETL ingestion pipeline: Load -> Chunk -> Embed -> Upsert to Pinecone."""
    settings.validate_api_keys()

    target_pdf = ensure_pdf_exists(pdf_path)
    target_index = index_name or settings.pinecone_index_name
    target_chunk_size = chunk_size or settings.chunk_size
    target_chunk_overlap = chunk_overlap or settings.chunk_overlap

    # 1. Load document pages using PyPDFLoader
    print(f"[Ingestion] Loading PDF from {target_pdf}...")
    loader = PyPDFLoader(str(target_pdf))
    docs = loader.load()
    print(f"[Ingestion] Successfully loaded {len(docs)} pages.")

    # 2. Text Chunking
    print(f"[Ingestion] Splitting documents (chunk_size={target_chunk_size}, chunk_overlap={target_chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=target_chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[Ingestion] Generated {len(chunks)} text chunks.")

    # 3. Pinecone Index Setup
    ensure_pinecone_index(
        index_name=target_index,
        dimension=settings.embedding_dimension,
        metric="cosine"
    )

    # 4. Create embeddings & upsert to Pinecone
    print(f"[Ingestion] Initializing OpenAI embeddings ({settings.openai_embedding_model})...")
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key
    )

    print(f"[Ingestion] Upserting {len(chunks)} chunks into Pinecone index '{target_index}'...")
    # Upsert in batches to avoid network payload limits
    vector_store = PineconeVectorStore(index_name=target_index, embedding=embeddings)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"[Ingestion] Uploaded batch {i // batch_size + 1} / {(len(chunks) + batch_size - 1) // batch_size} ({len(batch)} chunks)")

    print(f"[Ingestion] Complete! Total {len(chunks)} chunks indexed in '{target_index}'.")
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Agentic AI eBook PDF into Pinecone vector index.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    parser.add_argument("--index-name", type=str, default=None, help="Pinecone index name")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters")

    args = parser.parse_args()
    try:
        run_ingestion(
            pdf_path=args.pdf,
            index_name=args.index_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    except Exception as e:
        print(f"[Ingestion Error] {e}", file=sys.stderr)
        sys.exit(1)
