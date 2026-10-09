"""Friendly upload-first Streamlit UI for a document-grounded RAG assistant."""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import streamlit as st
from rag_pipeline import build_index, answer_question, DEFAULT_TOP_K, INDEX_DIR

st.set_page_config(page_title="DocuMind · Document Q&A", page_icon="📚", layout="wide")

ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = ROOT / "user_documents"
USER_INDEX_ROOT = INDEX_DIR / "user_collections"
SUPPORTED_TYPES = ["pdf", "txt", "docx"]
MAX_FILES = 5
MAX_FILE_MB = 20

if "collection_id" not in st.session_state:
    st.session_state.collection_id = uuid.uuid4().hex[:12]
if "collection_ready" not in st.session_state:
    st.session_state.collection_ready = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0
if "index_seconds" not in st.session_state:
    st.session_state.index_seconds = None

collection_dir = UPLOAD_ROOT / st.session_state.collection_id
index_dir = USER_INDEX_ROOT / st.session_state.collection_id

# Light styling: readable, approachable, and responsive.
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}
.hero {padding: 1.25rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg, rgba(76,110,245,.17), rgba(32,201,151,.10)); border: 1px solid rgba(128,128,128,.20); margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .82;}
.step-card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 1rem; height: 100%;}
.small-muted {font-size: .9rem; opacity: .75;}
div[data-testid="stExpander"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📚 DocuMind — Ask your documents</h1>
  <p>Upload up to 5 files. We'll extract the text, build a searchable index, and answer questions using your documents with source references.</p>
</div>
""", unsafe_allow_html=True)

step1, step2, step3 = st.columns(3)
with step1:
    st.markdown('<div class="step-card"><b>① Upload</b><br><span class="small-muted">PDF, DOCX or TXT · up to 5 files</span></div>', unsafe_allow_html=True)
with step2:
    st.markdown('<div class="step-card"><b>② Prepare</b><br><span class="small-muted">Text extraction, chunking, batched embeddings, FAISS</span></div>', unsafe_allow_html=True)
with step3:
    st.markdown('<div class="step-card"><b>③ Ask</b><br><span class="small-muted">Grounded answers with filename and page citations</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Answer settings")
    top_k = st.slider("Source passages to retrieve", min_value=1, max_value=8, value=DEFAULT_TOP_K)
    st.caption("Higher values give the model more context, but may include less relevant passages.")
    st.divider()
    st.markdown("**Privacy & storage**")
    st.caption("Uploaded files and their FAISS index are stored locally for this browser session's collection. Use ‘Clear collection’ on a trusted local deployment to remove them.")
    st.caption("Supported formats: PDF, DOCX, TXT. Maximum 20 MB per file.")

st.subheader("1. Add your documents")
uploaded_files = st.file_uploader(
    "Choose 1–5 documents",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="For best results, use text-based PDFs or documents with selectable text. Scanned/image-only PDFs may need OCR first.",
)

if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.error(f"You selected {len(uploaded_files)} files. Please select no more than {MAX_FILES} documents.")
    else:
        too_large = [f.name for f in uploaded_files if f.size > MAX_FILE_MB * 1024 * 1024]
        duplicate_names = len({f.name.lower() for f in uploaded_files}) != len(uploaded_files)
        if too_large:
            st.error(f"These files exceed {MAX_FILE_MB} MB: {', '.join(too_large)}. Please upload smaller files.")
        elif duplicate_names:
            st.error("Two or more files have the same filename. Rename them so source citations remain unambiguous.")
        else:
            st.success(f"{len(uploaded_files)} file(s) selected · total size {sum(f.size for f in uploaded_files) / (1024 * 1024):.2f} MB")
            for f in uploaded_files:
                st.write(f"📄 **{f.name}** · {f.size / 1024:.1f} KB")

            if st.button("✨ Process documents & create searchable knowledge base", type="primary", use_container_width=True):
                # Validate the API key before spending time building the index.
                import os
                from dotenv import load_dotenv
                load_dotenv()
                if not os.getenv("GROQ_API_KEY", "").strip():
                    st.error("GROQ_API_KEY is missing. Add it to your .env file, save it, then restart Streamlit.")
                    st.stop()

                with st.status("Preparing your documents…", expanded=True) as status:
                    started = time.perf_counter()
                    try:
                        if collection_dir.exists():
                            shutil.rmtree(collection_dir)
                        if index_dir.exists():
                            shutil.rmtree(index_dir)
                        collection_dir.mkdir(parents=True, exist_ok=True)

                        st.write("**Step 1/3** · Saving uploaded files…")
                        for file in uploaded_files:
                            safe_name = Path(file.name).name
                            (collection_dir / safe_name).write_bytes(file.getvalue())

                        st.write("**Step 2/3** · Extracting text, splitting into overlapping chunks, and generating batched embeddings…")
                        st.write("The first run may take longer while the embedding model is downloaded and loaded.")
                        stats = build_index(data_dir=collection_dir, index_dir=index_dir)

                        st.write("**Step 3/3** · Saving the FAISS index and source metadata to disk…")
                        elapsed = time.perf_counter() - started
                        st.session_state.collection_ready = True
                        st.session_state.collection_name = ", ".join(Path(f.name).name for f in uploaded_files)
                        st.session_state.indexed_count = stats["documents"]
                        st.session_state.index_seconds = elapsed
                        status.update(label="Your document collection is ready!", state="complete", expanded=False)
                        st.success(f"Ready in {elapsed:.1f} seconds · {stats['documents']} document(s) · {stats['chunks']} searchable chunks")
                    except Exception as exc:
                        st.session_state.collection_ready = False
                        status.update(label="We couldn't process these documents", state="error", expanded=True)
                        message = str(exc).strip() or exc.__class__.__name__
                        st.error(f"Processing failed: {message}")
                        st.info("Try a text-based PDF/DOCX/TXT file, check that it is not password-protected or corrupted, and try again. If the issue mentions memory or dependencies, restart the app after installing requirements.")

st.divider()
st.subheader("2. Ask a question")
if st.session_state.collection_ready and index_dir.exists():
    st.success(f"Knowledge base ready · {st.session_state.indexed_count} document(s) indexed" + (f" · prepared in {st.session_state.index_seconds:.1f}s" if st.session_state.index_seconds is not None else ""))
    with st.expander("Documents in this collection", expanded=False):
        for name in sorted(p.name for p in collection_dir.iterdir() if p.is_file()):
            st.write(f"📄 {name}")
    question = st.text_area("Your question", placeholder="e.g. Summarize the main points and tell me which page supports them.", height=90)
    if st.button("🔎 Find answer in my documents", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please type a question before searching.")
        else:
            answer_slot = st.container()
            started = time.perf_counter()
            try:
                with st.spinner("Searching your document index and composing a source-grounded answer…"):
                    answer, sources = answer_question(question, top_k=top_k, index_dir=index_dir)
                elapsed = time.perf_counter() - started
                with answer_slot:
                    st.subheader("💡 Answer")
                    st.write(answer)
                    st.caption(f"Completed in {elapsed:.2f} seconds · retrieved {len(sources)} source passage(s)")
                    st.subheader("📌 Sources used")
                    if not sources:
                        st.warning("No useful passages were found. Try rephrasing the question or check that the document contains the topic.")
                    for i, source in enumerate(sources, start=1):
                        page = f" · Page {source['page']}" if source.get("page") else " · Page not available for this file type"
                        with st.expander(f"[{i}] {source['source']}{page} · relevance {source['score']:.3f}", expanded=(i == 1)):
                            st.write(source["text"])
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.info("Upload and process your documents again to create the searchable index.")
            except Exception as exc:
                msg = str(exc).strip() or exc.__class__.__name__
                st.error(f"Couldn't answer this question: {msg}")
                st.info("Check your internet connection and GROQ_API_KEY. If the issue persists, process the documents again or restart the app.")
else:
    st.info("Upload at least one document and click **Process documents**. The question box will become available when indexing finishes.")

if st.session_state.collection_ready:
    st.divider()
    if st.button("🗑️ Clear my collection", help="Deletes this session's uploaded files and its saved FAISS index."):
        try:
            shutil.rmtree(collection_dir, ignore_errors=True)
            shutil.rmtree(index_dir, ignore_errors=True)
            st.session_state.collection_ready = False
            st.session_state.collection_name = ""
            st.session_state.indexed_count = 0
            st.session_state.index_seconds = None
            st.success("Collection and saved index removed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not clear collection: {exc}")

st.caption("Answers are generated from retrieved excerpts, not guaranteed to be complete or correct. Always verify important claims against the displayed source passages.")
