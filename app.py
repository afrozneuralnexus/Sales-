import streamlit as st
import os
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.utils import format_sources, get_file_icon

st.set_page_config(
    page_title="Sales Analysis RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d7377 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0d7377; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

    .source-chip {
        display: inline-block;
        background: #e8f4f8;
        border: 1px solid #b3d9e8;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        color: #1e3a5f;
    }

    .chat-container { max-height: 500px; overflow-y: auto; }

    .stChatMessage { border-radius: 10px; }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #0d7377;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vector_store": None,
    "doc_processor": None,
    "rag_engine": None,
    "uploaded_files_info": [],
    "total_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Sales Analysis RAG Assistant</h1>
    <p>Upload your sales data (Excel, PDF, Word, CSV) and ask anything about it — powered by Grok AI</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Grok API Key",
        type="password",
        placeholder="xai-...",
        help="Get your API key from console.x.ai",
    )

    st.divider()

    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your sales files here",
        type=["xlsx", "xls", "csv", "pdf", "docx", "doc"],
        accept_multiple_files=True,
        help="Supported: Excel (.xlsx/.xls), CSV, PDF, Word (.docx/.doc)",
    )

    chunk_size = st.slider("Chunk Size (tokens)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top-K Retrieval", 1, 10, 5)

    process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)

    st.divider()

    if st.session_state.uploaded_files_info:
        st.header("📂 Loaded Files")
        for info in st.session_state.uploaded_files_info:
            icon = get_file_icon(info["type"])
            st.markdown(f"{icon} **{info['name']}**  \n`{info['chunks']} chunks · {info['size']}`")

    if st.button("🗑️ Clear All", use_container_width=True):
        for k in ["messages", "vector_store", "doc_processor", "rag_engine", "uploaded_files_info", "total_chunks"]:
            st.session_state[k] = [] if k in ["messages", "uploaded_files_info"] else None if k != "total_chunks" else 0
        st.rerun()

# ── Process documents ─────────────────────────────────────────────────────
if process_btn:
    if not api_key:
        st.error("⚠️ Please enter your Grok API key in the sidebar.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one file.")
    else:
        with st.spinner("Processing documents…"):
            try:
                processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks = []
                files_info = []

                progress = st.progress(0)
                for i, f in enumerate(uploaded_files):
                    chunks = processor.process_file(f)
                    all_chunks.extend(chunks)
                    files_info.append({
                        "name": f.name,
                        "type": f.name.split(".")[-1].lower(),
                        "chunks": len(chunks),
                        "size": f"{f.size / 1024:.1f} KB",
                    })
                    progress.progress((i + 1) / len(uploaded_files))

                vs = VectorStore()
                vs.add_documents(all_chunks)

                st.session_state.vector_store = vs
                st.session_state.doc_processor = processor
                st.session_state.rag_engine = RAGEngine(api_key=api_key, vector_store=vs, top_k=top_k)
                st.session_state.uploaded_files_info = files_info
                st.session_state.total_chunks = len(all_chunks)

                st.success(f"✅ Processed {len(uploaded_files)} file(s) → {len(all_chunks)} chunks indexed!")
            except Exception as e:
                st.error(f"❌ Error processing files: {e}")

# ── Metrics row ───────────────────────────────────────────────────────────
if st.session_state.uploaded_files_info:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(st.session_state.uploaded_files_info)}</div>
            <div class="label">Files Loaded</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{st.session_state.total_chunks}</div>
            <div class="label">Total Chunks</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(st.session_state.messages) // 2}</div>
            <div class="label">Queries Asked</div></div>""", unsafe_allow_html=True)
    with c4:
        model_name = "grok-3-mini" if st.session_state.rag_engine else "—"
        st.markdown(f"""<div class="metric-card">
            <div class="value" style="font-size:1.1rem">🤖</div>
            <div class="label">{model_name}</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Suggested queries ─────────────────────────────────────────────────────
if st.session_state.rag_engine and not st.session_state.messages:
    st.subheader("💡 Suggested Queries")
    suggestions = [
        "What are the top 5 products by revenue?",
        "Show monthly sales trend and identify peak months",
        "Which region has the highest growth rate?",
        "What is the average deal size and win rate?",
        "Identify underperforming sales reps",
        "Forecast next quarter sales based on current trends",
    ]
    cols = st.columns(3)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 3]:
            if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ── Chat interface ────────────────────────────────────────────────────────
st.subheader("💬 Chat with Your Sales Data")

if not st.session_state.rag_engine:
    st.info("👆 Upload documents and click **Process Documents** to start chatting with your sales data.")
else:
    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources used"):
                    for src in msg["sources"]:
                        st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

    # New input
    if prompt := st.chat_input("Ask about your sales data…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your sales data…"):
                try:
                    result = st.session_state.rag_engine.query(
                        prompt,
                        chat_history=st.session_state.messages[:-1],
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Sources used"):
                            for src in sources:
                                st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
