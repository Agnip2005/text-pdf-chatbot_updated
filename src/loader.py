"""
Generic document loader.

Unlike the original version (which only ever read one hard-coded
data/data_tnu.txt file), this loader can load *any* PDF or TXT file the
user provides at request time. This is what allows the chatbot to answer
questions from a different document every time, instead of being
permanently trained on a single file.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _load_txt(path: Path) -> List[Document]:
    """Load a plain-text file into Documents, splitting on optional page breaks."""
    raw_text = path.read_text(encoding="utf-8", errors="ignore")

    documents = []
    pages = raw_text.split("===== PAGE BREAK =====")

    for i, page_text in enumerate(pages):
        content = page_text.strip()
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "page_number": i + 1,
                },
            )
        )

    return documents


def _load_pdf(path: Path) -> List[Document]:
    """Load a PDF file into Documents, one Document per page."""
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. Add `pypdf` to requirements.txt to enable PDF support."
        )

    reader = PdfReader(str(path))
    documents = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page_number": i + 1,
                },
            )
        )

    return documents


def load_document(file_path: str) -> List[Document]:
    """
    Load a single file (PDF or TXT) from disk into a list of LangChain
    Documents. This is the entry point used for user-uploaded files.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix == ".txt":
        return _load_txt(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )


def load_documents_from_paths(file_paths: List[str]) -> List[Document]:
    """Load and concatenate documents from multiple uploaded files."""
    all_documents: List[Document] = []
    for file_path in file_paths:
        all_documents.extend(load_document(file_path))
    return all_documents


# ---------------------------------------------------------------------------
# Backwards-compatible default loader (kept for the old sample dataset so the
# original build_index.py / CLI flow still works if someone runs it).
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "data_tnu.txt"


def load_documents() -> List[Document]:
    """Load the legacy default TNU dataset (kept for backwards compatibility)."""
    return _load_txt(DATA_FILE)
