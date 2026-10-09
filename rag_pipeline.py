"""Core retrieval and grounded answer generation for the Bookxpert RAG assignment."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # Load local .env values when present; environment variables still take precedence.

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "faiss_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_documents(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load PDF/TXT/DOCX files, preserving filename and page/section metadata."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Document folder not found: {data_dir}")
    paths = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise ValueError(f"No PDF, TXT, or DOCX files found in {data_dir}")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pages = PyPDFLoader(str(path)).load()
                for page_doc in pages:
                    text = page_doc.page_content.strip()
                    if text:
                        records.append({
                            "text": text,
                            "source": path.name,
                            "page": int(page_doc.metadata.get("page", 0)) + 1,
                        })
            elif suffix == ".txt":
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    records.append({"text": text, "source": path.name, "page": None})
            elif suffix == ".docx":
                for doc in Docx2txtLoader(str(path)).load():
                    text = doc.page_content.strip()
                    if text:
                        records.append({"text": text, "source": path.name, "page": None})
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not records:
        detail = "; ".join(errors) if errors else "No extractable text was found."
        raise ValueError(f"No usable text was extracted. Details: {detail}")
    return records


def split_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split extracted text using fixed-size recursive character chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[dict[str, Any]] = []
    for record in records:
        for chunk in splitter.split_text(record["text"]):
            clean = chunk.strip()
            if len(clean) < 40:
                continue
            chunks.append({
                "text": clean,
                "source": record["source"],
                "page": record.get("page"),
            })
    if not chunks:
        raise ValueError("Document text was extracted, but no useful chunks were created.")
    return chunks


def _embed_in_batches(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list in batches; never call the model once per individual chunk."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> dict[str, int]:
    records = load_documents(data_dir)
    chunks = split_records(records)
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = _embed_in_batches(model, [c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "documents.faiss"))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": chunks,
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"documents": len({r["source"] for r in records}), "chunks": len(chunks)}


def load_index(index_dir: Path = INDEX_DIR) -> tuple[Any, list[dict[str, Any]], SentenceTransformer]:
    index_path = index_dir / "documents.faiss"
    metadata_path = index_dir / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Persistent FAISS index not found. Run `python index_documents.py` first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError(
            f"Index uses {metadata.get('embedding_model')}, but current configuration uses {EMBEDDING_MODEL}. "
            "Rebuild the index with `python index_documents.py`."
        )
    index = faiss.read_index(str(index_path))
    chunks = metadata.get("chunks", [])
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata are out of sync. Rebuild the index.")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, chunks, model


def retrieve(question: str, top_k: int = DEFAULT_TOP_K, index_dir: Path = INDEX_DIR) -> list[dict[str, Any]]:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    index, chunks, model = load_index(index_dir)
    query_vector = _embed_in_batches(model, [question], batch_size=1)
    scores, positions = index.search(query_vector, min(top_k, index.ntotal))
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        item = dict(chunks[int(position)])
        item["score"] = float(score)
        results.append(item)
    return results


def generate_answer(question: str, retrieved: list[dict[str, Any]]) -> str:
    """Generate a context-grounded answer with Groq; sources are rendered by the app."""
    if not retrieved:
        return "I could not find relevant information in the indexed documents, so I cannot answer this from the available sources."
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to your environment before asking questions.")
    from groq import Groq
    context_parts = []
    for i, item in enumerate(retrieved, start=1):
        page = f", page {item['page']}" if item.get("page") else ""
        context_parts.append(f"[Source {i}: {item['source']}{page}]\n{item['text']}")
    context = "\n\n".join(context_parts)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. Answer using only the supplied source excerpts. "
                    "If the excerpts do not contain enough evidence, say: 'The available documents do not contain enough "
                    "information to answer this question.' Do not use outside knowledge. Do not invent facts, filenames, "
                    "page numbers, or citations. Keep the answer concise and explain uncertainty when appropriate."
                ),
            },
            {"role": "user", "content": f"Question:\n{question}\n\nSource excerpts:\n{context}\n\nAnswer from these excerpts only."},
        ],
    )
    return response.choices[0].message.content or "The model returned an empty response."


def answer_question(
    question: str, top_k: int = DEFAULT_TOP_K, index_dir: Path = INDEX_DIR
) -> tuple[str, list[dict[str, Any]]]:
    """Answer from a chosen persistent collection index (default: assignment index)."""
    sources = retrieve(question, top_k=top_k, index_dir=index_dir)
    answer = generate_answer(question, sources)
    return answer, sources
