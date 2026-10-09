"""Build and persist the FAISS index from files in ./data."""
from rag_pipeline import build_index

if __name__ == "__main__":
    try:
        summary = build_index()
        print("\nIndexing complete.")
        print(f"Documents indexed: {summary['documents']}")
        print(f"Chunks stored: {summary['chunks']}")
        print("Persistent index: faiss_db/documents.faiss")
        print("Metadata: faiss_db/metadata.json")
    except Exception as exc:
        print(f"Indexing failed: {exc}")
        raise SystemExit(1)
