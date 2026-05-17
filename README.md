# 📊 Sales Analysis RAG — Powered by Grok AI

A Retrieval-Augmented Generation (RAG) application for sales data analysis. Upload your sales documents and ask natural language questions — the app retrieves relevant context and uses **Grok (xAI)** to generate actionable insights.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | Excel (`.xlsx`/`.xls`), CSV, PDF, Word (`.docx`/`.doc`) |
| **Smart chunking** | Configurable chunk size & overlap |
| **TF-IDF retrieval** | Fast, offline semantic search — no external vector DB needed |
| **Grok-3-mini** | Low-latency reasoning with full chat history support |
| **Streamlit UI** | One-click file upload, suggested queries, source citations |
| **Sales-aware prompts** | System prompt tuned for sales KPIs, trends, and recommendations |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                     │
│  File Upload → Document Processor → Vector Store (TFIDF) │
│                       ↓                                  │
│              User Query → Retriever → Grok API           │
│                       ↓                                  │
│              Answer + Sources → Chat UI                  │
└──────────────────────────────────────────────────────────┘
```

### File structure

```
sales-rag-grok/
├── app.py                  # Streamlit entry point
├── src/
│   ├── document_processor.py  # Excel/CSV/PDF/Word ingestion & chunking
│   ├── vector_store.py        # TF-IDF vector store
│   ├── rag_engine.py          # Grok API integration + RAG pipeline
│   └── utils.py               # Helper functions
├── tests/
│   ├── test_document_processor.py
│   ├── test_vector_store.py
│   └── test_rag_engine.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/sales-rag-grok.git
cd sales-rag-grok
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your Grok API key

```bash
cp .env.example .env
# Edit .env and add your key:
# GROK_API_KEY=xai-your-key-here
```

> Get your key at **[console.x.ai](https://console.x.ai)**

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROK_API_KEY` | Yes | xAI Grok API key |

You can also enter the key directly in the Streamlit sidebar — no `.env` file needed.

---

## 📋 Supported File Types

| Format | Extension | Notes |
|---|---|---|
| Excel | `.xlsx`, `.xls` | All sheets processed; numeric summaries auto-generated |
| CSV | `.csv` | UTF-8, Latin-1, CP1252 encodings supported |
| PDF | `.pdf` | Text-based PDFs; scanned PDFs may need OCR preprocessing |
| Word | `.docx`, `.doc` | Paragraphs + tables extracted |

---

## 💬 Example Queries

- *"What are the top 5 products by revenue?"*
- *"Show the monthly sales trend for Q1 2024"*
- *"Which sales rep has the highest win rate?"*
- *"Compare region-wise performance and identify underperformers"*
- *"What is the average deal size and how has it changed?"*
- *"Forecast next quarter sales based on current trends"*

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Chunk Size | 500 tokens | Words per chunk |
| Chunk Overlap | 50 tokens | Overlap between consecutive chunks |
| Top-K Retrieval | 5 | Number of chunks sent to Grok |

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 🛠️ Development

### Adding a new file type

1. Add a handler method `_process_<ext>` in `src/document_processor.py`
2. Register the extension in the `process_file` dispatcher
3. Add the extension to `st.file_uploader(type=[...])` in `app.py`

### Swapping the vector store

Replace `VectorStore` in `src/vector_store.py` with any store that exposes:
```python
def add_documents(chunks: List[DocumentChunk]) -> None: ...
def similarity_search(query: str, k: int) -> List[Tuple[DocumentChunk, float]]: ...
```

Compatible with: **ChromaDB**, **FAISS**, **Pinecone**, **Qdrant**.

---

## 📄 License

MIT © 2024 — see [LICENSE](LICENSE) for details.
