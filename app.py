import os
import json
import numpy as np
import streamlit as st
import shutil
import re
from datetime import datetime

from embedder import Embedder
from chunker import DocumentChunker
from graph_builder import GraphBuilder
from retrieval.bm25_index import BM25Retriever
from retrieval.vector_search import VectorSearch
from retrieval.graph_retriever import GraphRetriever
from retrieval.fusion import ResultFusion
from retrieval.reranker import Reranker
from llm import LLMGenerator

# -----------------------------------------------------------------------------
# 2. PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="GraphRAG", layout="wide")


# -----------------------------------------------------------------------------
# 3. HELPER: get_existing_docs()
# -----------------------------------------------------------------------------
def get_existing_docs() -> list:
    """Scans doc_store/ for ready docs and returns their doc_names."""
    if not os.path.exists("doc_store"):
        os.makedirs("doc_store", exist_ok=True)
    ready_docs = []
    for doc_name in os.listdir("doc_store"):
        meta_path = f"doc_store/{doc_name}/meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("status") == "ready":
                    ready_docs.append(doc_name)
            except Exception:
                pass
    return ready_docs


# -----------------------------------------------------------------------------
# 4. HELPER: ingest_pdf(uploaded_file)
# -----------------------------------------------------------------------------
def ingest_pdf(uploaded_file) -> dict:
    """
    Runs the full ingestion pipeline on an uploaded PDF.
    Saves all indexes to doc_store/<doc_name>/.
    Returns a metadata dict on success.
    Raises an exception on failure.
    """
    filename = uploaded_file.name
    # 1. Derive doc_name
    doc_name = re.sub(r'[^a-zA-Z0-9]', '_', filename.replace(".pdf", "")).lower()
    doc_path = f"doc_store/{doc_name}"

    # 2. Create directory
    os.makedirs(doc_path, exist_ok=True)
    
    try:
        # 3. Save uploaded bytes
        source_path = f"{doc_path}/source.pdf"
        with open(source_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 4. Run chunker
        chunker = DocumentChunker(chunk_size_words=200, overlap_words=50)
        chunks = chunker.process_pdf(source_path)
        with open(f"{doc_path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        # 5. Run embedder
        embedder = get_shared_models()["embedder"]
        embeddings = embedder.generate_embeddings(chunks)
        np.save(f"{doc_path}/embeddings.npy", embeddings)
        
        # 6. Build FAISS index
        vs = VectorSearch()
        vs.build_faiss_index(chunks, embeddings)
        vs.save_index(index_path=f"{doc_path}/faiss.index", chunks_path=f"{doc_path}/faiss_chunks.pkl")

        # 7. Build BM25 index
        bm25 = BM25Retriever()
        bm25.build_bm25_index(chunks)
        bm25.save_index(output_path=f"{doc_path}/bm25_index.pkl")

        # 8. Build knowledge graph
        gb = GraphBuilder()
        gb.build_graph(chunks, embeddings)
        gb.save_graph(output_path=f"{doc_path}/graph.pkl")

        # 9. Write meta.json
        meta = {
            "filename": filename,
            "doc_name": doc_name,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "page_count": len(set([c.get("page_number", 0) for c in chunks])),
            "chunk_count": len(chunks),
            "status": "ready"
        }
        with open(f"{doc_path}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return meta
    except Exception as e:
        shutil.rmtree(doc_path, ignore_errors=True)
        raise e


# -----------------------------------------------------------------------------
# 5. CACHED LOADER: get_shared_models()
# -----------------------------------------------------------------------------
@st.cache_resource
def get_shared_models() -> dict:
    return {
        "embedder": Embedder(),
        "llm": LLMGenerator(),
        "reranker": Reranker()
    }


# -----------------------------------------------------------------------------
# 6. CACHED LOADER: get_doc_system(doc_name)
# -----------------------------------------------------------------------------
def load_doc_systems(doc_name: str) -> dict:
    """Loads all indexes for a single document into memory."""
    doc_path = f"doc_store/{doc_name}"
    
    with open(f"{doc_path}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    embeddings = np.load(f"{doc_path}/embeddings.npy")

    bm25 = BM25Retriever()
    bm25.load_index(input_path=f"{doc_path}/bm25_index.pkl")

    vs = VectorSearch()
    vs.load_index(index_path=f"{doc_path}/faiss.index", chunks_path=f"{doc_path}/faiss_chunks.pkl")

    gb = GraphBuilder()
    gb.load_graph(input_path=f"{doc_path}/graph.pkl")

    gr = GraphRetriever(gb)
    fusion = ResultFusion()
    
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "bm25": bm25,
        "vs": vs,
        "gb": gb,
        "gr": gr,
        "fusion": fusion
    }

@st.cache_resource
def get_doc_system(doc_name: str) -> dict:
    return load_doc_systems(doc_name)


# -----------------------------------------------------------------------------
# 7. HELPER: query_all_docs()
# -----------------------------------------------------------------------------
def query_all_docs(query: str, doc_names: list, top_k: int = 10, graph_depth: int = 1) -> dict:
    shared = get_shared_models()
    embedder = shared["embedder"]
    reranker = shared["reranker"]
    llm = shared["llm"]
    
    q_emb = embedder.generate_query_embedding(query)
    
    merged_candidates = []
    
    for doc_name in doc_names:
        sys = get_doc_system(doc_name)
        
        # 1. Lexical Search
        bm25_res = sys["bm25"].bm25_search(query, top_k=20)
        
        # 2. Vector Search
        vec_res = sys["vs"].vector_search(q_emb, top_k=20)
        
        # 3. Graph Expansion
        graph_res = []
        if graph_depth > 0:
            seeds = [res[0] for res in vec_res[:3]]
            graph_res = sys["gr"].graph_based_retrieval(seeds, max_depth=graph_depth, use_bfs=True)
            
        # 4. Fusion
        fused_res = sys["fusion"].fuse_results(bm25_res, vec_res, graph_res, top_k=20)
        merged_candidates.extend(fused_res)
        
    # Sort merged candidates by initial fusion score to keep top pool
    merged_candidates.sort(key=lambda x: x[1], reverse=True)
    # Take top 50 across all docs for reranking to save time
    merged_candidates = merged_candidates[:50]
    
    # 5. Reranking
    final_res = reranker.rerank(query, merged_candidates, top_k=top_k)
    
    # 6. LLM Generation
    answer = llm.generate_answer(query, final_res)

    sources = []
    for chunk, score in final_res:
        sources.append({
            "filename": chunk.get("source_filename", "Unknown"),
            "page": chunk.get("page_number", 0),
            "score": score,
            "text": chunk.get("text", "")
        })

    return {
        "answer": answer,
        "sources": sources
    }


# -----------------------------------------------------------------------------
# 8. HELPER: delete_doc()
# -----------------------------------------------------------------------------
def delete_doc(doc_name: str):
    shutil.rmtree(f"doc_store/{doc_name}", ignore_errors=True)
    st.session_state.uploaded_docs = [d for d in st.session_state.uploaded_docs if d != doc_name]
    get_doc_system.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# 9. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = get_existing_docs()

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------------------------------------------------
# 10. SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.title("GraphRAG")
st.sidebar.caption("Zero hallucination document Q&A")

st.sidebar.markdown("── Your Documents ──")

for doc_name in list(st.session_state.uploaded_docs):
    meta_path = f"doc_store/{doc_name}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"📄 **{meta['filename']}**")
        col1.caption(f"{meta.get('chunk_count', 0)} chunks · {meta.get('page_count', 0)} pages")
        if col2.button("🗑", key=f"del_{doc_name}"):
            delete_doc(doc_name)
    else:
        # cleanup missing
        st.session_state.uploaded_docs.remove(doc_name)

st.sidebar.markdown("── ── ── ── ── ──")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    already_uploaded = [
        json.load(open(f"doc_store/{d}/meta.json"))["filename"]
        for d in st.session_state.uploaded_docs
        if os.path.exists(f"doc_store/{d}/meta.json")
    ]
    if uploaded_file.name not in already_uploaded:
        with st.sidebar.status("Processing PDF...", expanded=True) as status:
            st.write("Extracting text...")
            st.write("Building indexes...")
            st.write("Building knowledge graph...")
            try:
                meta = ingest_pdf(uploaded_file)
                st.session_state.uploaded_docs.append(meta["doc_name"])
                status.update(label="✅ Ready!", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label=f"❌ Failed: {e}", state="error")
                st.error(f"Failed to process PDF: {e}")

st.sidebar.markdown("── Settings ──")
with st.sidebar.expander("⚙️ Settings"):
    top_k = st.slider("Results per document", 3, 20, 5)
    graph_depth = st.slider("Graph depth", 0, 3, 1)


# -----------------------------------------------------------------------------
# 11. MAIN AREA UI
# -----------------------------------------------------------------------------
if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ GROQ_API_KEY not set in .env file or environment variables. Answers will fail.")

if not st.session_state.uploaded_docs:
    st.title("GraphRAG")
    st.markdown("### Upload a PDF to get started")
    st.markdown("Your documents are processed locally. "
                "Answers are grounded in your documents only — no hallucinations.")
    st.info("👈 Upload a PDF using the sidebar to begin")
    st.stop()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['filename']}** · Page {src['page']} · "
                        f"Score: {src['score']:.2f}"
                    )
                    st.caption(src["text"][:300] + "...")
                    st.divider()

if prompt := st.chat_input(
    "Ask anything about your documents...",
    disabled=len(st.session_state.uploaded_docs) == 0
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = query_all_docs(
                    prompt,
                    st.session_state.uploaded_docs,
                    top_k=top_k,
                    graph_depth=graph_depth
                )
                st.write(result["answer"])
                if result["sources"]:
                    with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                        for src in result["sources"]:
                            st.markdown(
                                f"**{src['filename']}** · Page {src['page']} · "
                                f"Score: {src['score']:.2f}"
                            )
                            st.caption(src["text"][:300] + "...")
                            st.divider()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"Query failed: {e}")
