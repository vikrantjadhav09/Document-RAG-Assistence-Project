# Basic Document Q&A Bot (RAG Pipeline)

A document question-answering application. It loads a small collection of PDF, TXT, or DOCX files, splits extracted text into overlapping chunks, embeds chunks in batches, stores vectors in a persistent FAISS index, retrieves relevant passages for a question, and uses Groq to generate an answer grounded in those passages. The interface displays the retrieved source text and application-generated filename/page labels.

## Features

- PDF, TXT, and DOCX ingestion from `data/`
- Fixed-size recursive character chunking with overlap
- Batched semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Persistent FAISS inner-product index with normalized vectors
- Separate indexing and querying workflows
- Configurable top-k retrieval
- Groq answer generation with instructions to abstain when evidence is insufficient
- Source excerpts with filename and PDF page metadata
- Streamlit UI and interactive command-line loop

## Architecture

```text
PDF / TXT / DOCX files in data/
          |
          v
Document loaders (PDF page metadata preserved)
          |
          v
Recursive character chunking (900 characters, 150 overlap)
          |
          v
SentenceTransformer embeddings (batched)
          |
          v
FAISS index + metadata.json saved under faiss_db/
          |
          |  User question
          v
Query embedding -> top-k FAISS search
          |
          v
Retrieved excerpts + source metadata
          |
          v
Groq LLM (context-only answer instructions)
          |
          v
Answer + retrieved source passages/citations
```

## Tech stack

- Python 3.11+
- Streamlit 1.50.0
- LangChain Community 0.3.29 (document loaders)
- LangChain Text Splitters 0.3.11
- pypdf 6.1.1
- Sentence Transformers 5.1.1
- FAISS CPU 1.12.0
- NumPy 2.2.6
- Groq Python SDK 0.31.0
- python-dotenv 1.1.1
- pytest 8.4.2 (tests)
- python-docx extraction through `docx2txt` 0.9

Dependency pins are the intended baseline. If a package version is unavailable for your Python/platform, install a compatible version and update the pins after testing.

## Documents

The `data/` directory contains five original educational documents about RAG fundamentals, embedding and vector search, document ingestion and metadata, grounded answers and evaluation, and practical RAG implementation. Four are TXT files and one is a PDF. Check the included documents' word/page counts before submission if the evaluator applies the assignment's minimum length literally.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vikrantjadhav09/PDF-RAG-ChatBot.git
cd PDF-RAG-ChatBot
```

### 2. Create and activate a virtual environment

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Groq

Copy `.env.example` to `.env` and set `GROQ_API_KEY` to your own key. The application loads `.env` using `python-dotenv`. Never commit your real key.

PowerShell for the current terminal session:

```powershell
$env:GROQ_API_KEY="your_key_here"
```

macOS/Linux:

```bash
export GROQ_API_KEY="your_key_here"
```

Optional environment variables: `GROQ_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K`.

### 5. Build the index once

```bash
python index_documents.py
```

This creates `faiss_db/documents.faiss` and `faiss_db/metadata.json`. Re-run indexing whenever documents or embedding configuration changes. The index is deliberately excluded from Git and can be rebuilt from the committed `data/` files.

### 6. Run the web UI

```bash
streamlit run app.py
```

### 7. Or run the command-line loop

```bash
python cli.py
```

Type `exit` to quit. Both interfaces load the saved index; they do not rebuild it for every question.

### 8. Run unit tests

```bash
pytest -q
```



## Design decisions

- **Chunking:** Recursive character chunks of 900 characters with 150 characters of overlap provide a simple baseline and preserve nearby context.
- **Embeddings:** `all-MiniLM-L6-v2` is a lightweight semantic embedding model suitable for a small demo. Embeddings are produced in batches.
- **Vector database:** FAISS `IndexFlatIP` with normalized vectors supports exact inner-product search for a small collection. The index and chunk metadata are persisted separately.
- **Metadata:** PDF page numbers are normalized to one-based numbering; TXT and DOCX sources are cited by filename.
- **Indexing/query separation:** `index_documents.py` builds the store; `app.py` and `cli.py` load it for questions.

## Known limitations

- The system depends on Groq availability and a valid API key for generated answers.
- Scanned/image-only PDFs may not yield usable text because OCR is not implemented.
- TXT and DOCX citations use filenames; page-level citations are available for PDFs only.
- Similarity search may retrieve irrelevant chunks. Scores are not calibrated answer-confidence probabilities.
- The prompt encourages abstention but cannot guarantee that the LLM never makes an unsupported claim. Inspect the displayed source passages.
- Tables, complex layouts, conflicting sources, and very long documents may require more specialized parsing and evaluation.
- The index must be rebuilt after changing the document collection or embedding model.

## Security

Never commit `.env`, API keys, or other secrets. Only load FAISS index files generated by a trusted indexing process. The supplied sample documents are educational material created for this demo.



## Upload-first UI workflow

1. Start the app with `streamlit run app.py`.
2. Upload 1–5 PDF, TXT, or DOCX files (up to 20 MB each).
3. Click **Process documents & create searchable knowledge base**. The app saves the files to a session-specific local folder, extracts text, chunks it with overlap, generates embeddings in batches, and writes a persistent FAISS index plus metadata.
4. Ask questions in the same UI. Each response shows the answer, processing time, and retrieved source passages with filename and PDF page where available.
5. Use **Clear my collection** to remove the session's uploaded files and index.

The first indexing run can take longer because the embedding model may need to download. Actual processing time depends on file size, page count, device resources, and network/model availability; the UI displays the measured elapsed time rather than promising a fixed duration. Uploads and per-collection indexes are stored locally on the machine running Streamlit. This is intended for a local assignment/demo, not a multi-user production deployment without authentication, quotas, and isolated storage.
