import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Resolve project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass
class Settings:
    # OpenAI Settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = 1536

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Document Chunking Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # RAG Retrieval Settings
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "3"))

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    default_pdf_path: Path = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"

    def has_api_keys(self) -> bool:
        """Check if both OpenAI and Pinecone API keys are set."""
        return bool(
            self.openai_api_key
            and self.pinecone_api_key
            and self.openai_api_key.strip() not in ("", "your_openai_api_key_here")
            and self.pinecone_api_key.strip() not in ("", "your_pinecone_api_key_here")
        )

    def validate_api_keys(self) -> None:
        """Validate required API keys and raise informative errors if missing."""
        missing = []
        if not self.openai_api_key or self.openai_api_key.strip() == "your_openai_api_key_here":
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key or self.pinecone_api_key.strip() == "your_pinecone_api_key_here":
            missing.append("PINECONE_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Please configure them in your .env file or environment."
            )


settings = Settings()
