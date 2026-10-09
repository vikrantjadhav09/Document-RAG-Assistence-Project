"""Lightweight unit tests for deterministic pipeline helpers."""
from rag_pipeline import split_records

def test_split_records_preserves_source_metadata():
    records = [{
        "text": ("Retrieval augmented generation connects a language model to external documents. " * 30),
        "source": "sample.txt",
        "page": None,
    }]
    chunks = split_records(records)
    assert chunks
    assert all(chunk["source"] == "sample.txt" for chunk in chunks)
    assert all("text" in chunk and len(chunk["text"]) >= 40 for chunk in chunks)

def test_page_metadata_is_preserved():
    records = [{
        "text": ("FAISS can search dense vectors and store a searchable index. " * 25),
        "source": "guide.pdf",
        "page": 3,
    }]
    chunks = split_records(records)
    assert chunks
    assert all(chunk["page"] == 3 for chunk in chunks)
