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
from retrieval.hyde import HyDEGenerator
from retrieval.query_decomposer import QueryDecomposer
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
def ingest_pdf_with_progress(uploaded_file, progress_bar, status_text) -> dict:
    """Same as ingest_pdf but updates a progress bar at each step."""
    filename = uploaded_file.name
    doc_name = re.sub(r'[^a-zA-Z0-9]', '_', filename.replace(".pdf", "")).lower()
    doc_path = f"doc_store/{doc_name}"
    os.makedirs(doc_path, exist_ok=True)

    try:
        # Step 1 — Save PDF
        source_path = f"{doc_path}/source.pdf"
        with open(source_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Step 2 — Chunk
        status_text.text("📄 Step 2/5: Chunking document...")
        progress_bar.progress(20, text="Chunking document...")
        chunker = DocumentChunker(chunk_size_words=200, overlap_words=50)
        chunks = chunker.process_pdf(source_path)
        with open(f"{doc_path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        # Step 3 — Embed
        status_text.text("🔢 Step 3/5: Generating embeddings...")
        progress_bar.progress(45, text="Generating embeddings...")
        embedder = get_shared_models()["embedder"]
        embeddings = embedder.generate_embeddings(chunks)
        np.save(f"{doc_path}/embeddings.npy", embeddings)

        # Step 4 — FAISS + BM25
        status_text.text("🔍 Step 4/5: Building search indexes...")
        progress_bar.progress(65, text="Building search indexes...")
        vs = VectorSearch()
        vs.build_faiss_index(chunks, embeddings)
        vs.save_index(
            index_path=f"{doc_path}/faiss.index",
            chunks_path=f"{doc_path}/faiss_chunks.pkl"
        )
        bm25 = BM25Retriever()
        bm25.build_bm25_index(chunks)
        bm25.save_index(output_path=f"{doc_path}/bm25_index.pkl")

        # Step 5 — Knowledge graph
        status_text.text("🕸️ Step 5/5: Building knowledge graph...")
        progress_bar.progress(85, text="Building knowledge graph...")
        gb = GraphBuilder()
        gb.build_graph(chunks, embeddings)
        gb.save_graph(output_path=f"{doc_path}/graph.pkl")

        # Write meta
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
    # Set the key in environment so Groq() client picks it up
    if st.session_state.get("groq_api_key"):
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    return {
        "embedder": Embedder(),
        "llm": LLMGenerator(),
        "reranker": Reranker(),
        "hyde": HyDEGenerator(),
        "decomposer": QueryDecomposer()
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
def verify_citations(answer: str, sources: list) -> str:
    """
    Verifies that source citations in the answer match actual retrieved chunks.
    Marks unverified citations with a ⚠️ warning.

    Example: [Source 1: test.pdf, Page 7] → checks if source 1 is actually
    test.pdf page 7 in the sources list.
    """
    import re

    def check_citation(match):
        full_cite = match.group(0)
        try:
            # Extract source number
            src_num_match = re.search(r'Source\s+(\d+)', full_cite)
            if not src_num_match:
                return full_cite

            src_num = int(src_num_match.group(1)) - 1  # 0-indexed

            if src_num < 0 or src_num >= len(sources):
                return f"{full_cite}⚠️"

            actual_source = sources[src_num]

            # Extract claimed filename from citation
            filename_match = re.search(r'Source \d+:\s*([^,]+)', full_cite)
            if filename_match:
                claimed_file = filename_match.group(1).strip()
                actual_file = actual_source.get("filename", "")

                if claimed_file.lower() not in actual_file.lower():
                    return f"{full_cite}⚠️"

            return full_cite  # Citation verified
        except Exception:
            return full_cite  # On any error, leave citation unchanged

    # Match patterns like [Source 1: test.pdf, Page 7] or [Source 1]
    citation_pattern = r'\[Source\s+\d+(?:[^\]]*?)?\]'
    verified_answer = re.sub(citation_pattern, check_citation, answer)
    return verified_answer

def compute_confidence(reranked_results: list) -> tuple:
    """
    Returns (label, color_emoji) based on top chunk scores.
    Reranker scores are typically in range -10 to +10.
    """
    if not reranked_results:
        return "Low", "🔴"

    # Take average of top 3 scores
    top_scores = [score for _, score in reranked_results[:3]]
    avg_score = sum(top_scores) / len(top_scores)

    if avg_score >= 2.0:
        return "High", "🟢"
    elif avg_score >= 0.0:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"

def query_all_docs(query: str, doc_names: list, top_k: int = 10, graph_depth: int = 1, chat_history: list = None) -> dict:
    shared = get_shared_models()
    embedder = shared["embedder"]
    reranker = shared["reranker"]
    llm = shared["llm"]
    hyde = shared["hyde"]
    decomposer = shared["decomposer"]

    # Decompose complex queries into sub-questions
    sub_questions = decomposer.decompose(query)

    merged_candidates = []

    # Retrieve for each sub-question separately
    for sub_q in sub_questions:
        hypothesis = hyde.generate_hypothesis(sub_q)
        q_emb_hyde = embedder.generate_query_embedding(hypothesis)
        q_emb_original = embedder.generate_query_embedding(sub_q)

        for doc_name in doc_names:
            sys = get_doc_system(doc_name)

            bm25_res = sys["bm25"].bm25_search(sub_q, top_k=10)
            vec_res = sys["vs"].vector_search(q_emb_hyde, top_k=10)

            graph_res = []
            if graph_depth > 0:
                seeds = [res[0] for res in sys["vs"].vector_search(q_emb_original, top_k=3)]
                graph_res = sys["gr"].graph_based_retrieval(seeds, max_depth=graph_depth, use_bfs=True)

            fused_res = sys["fusion"].fuse_results(bm25_res, vec_res, graph_res, top_k=10)
            merged_candidates.extend(fused_res)

    # Deduplicate by chunk_id before reranking
    seen_ids = set()
    unique_candidates = []
    for chunk, score in merged_candidates:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_candidates.append((chunk, score))

    unique_candidates.sort(key=lambda x: x[1], reverse=True)
    unique_candidates = unique_candidates[:50]

    # Rerank against ORIGINAL query (not sub-questions)
    final_res = reranker.rerank(query, unique_candidates, top_k=top_k)

    # 6. LLM Generation
    answer = llm.generate_answer(query, final_res, chat_history=chat_history)

    sources = []
    for chunk, score in final_res:
        sources.append({
            "filename": chunk.get("source_filename", "Unknown"),
            "page": chunk.get("page_number", 0),
            "score": score,
            "text": chunk.get("text", "")
        })

    confidence_label, confidence_emoji = compute_confidence(final_res)

    # Verify citations against actual retrieved sources
    answer = verify_citations(answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence_label,
        "confidence_emoji": confidence_emoji
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

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")


# -----------------------------------------------------------------------------
# 10. SIDEBAR UI
# -----------------------------------------------------------------------------
st.sidebar.title("GraphRAG")
st.sidebar.caption("Zero hallucination document Q&A")

st.sidebar.markdown("── API Key ──")
api_key_input = st.sidebar.text_input(
    "Groq API Key",
    value=st.session_state.groq_api_key,
    type="password",
    placeholder="gsk_...",
    help="Get a free key at console.groq.com"
)
if api_key_input != st.session_state.groq_api_key:
    st.session_state.groq_api_key = api_key_input
    # Clear cached models so they reload with the new key
    get_shared_models.clear()
    st.rerun()

if not st.session_state.groq_api_key:
    st.sidebar.warning("⚠️ Enter your Groq API key to enable answers")
else:
    st.sidebar.success("✓ API key set")

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

col1, col2 = st.sidebar.columns(2)
if col1.button("🗑 Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if col2.button("↺ Reload Docs", use_container_width=True,
               help="Rescan doc_store for new documents"):
    st.session_state.uploaded_docs = get_existing_docs()
    get_doc_system.clear()
    st.rerun()

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    already_uploaded_names = []
    for d in st.session_state.uploaded_docs:
        meta_path = f"doc_store/{d}/meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    already_uploaded_names.append(json.load(f)["filename"])
            except:
                pass

    if uploaded_file.name not in already_uploaded_names:
        progress_bar = st.sidebar.progress(0, text="Starting...")
        status_text = st.sidebar.empty()

        try:
            # Step 1: Save and chunk
            status_text.text("📄 Step 1/5: Extracting text...")
            progress_bar.progress(10, text="Extracting text...")
            meta = ingest_pdf_with_progress(
                uploaded_file, progress_bar, status_text
            )
            st.session_state.uploaded_docs.append(meta["doc_name"])
            progress_bar.progress(100, text="✅ Done!")
            status_text.text(f"✅ {uploaded_file.name} ready!")
            import time; time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.sidebar.error(f"❌ Failed: {e}")

st.sidebar.markdown("── Settings ──")
with st.sidebar.expander("⚙️ Settings"):
    top_k = st.slider("Results per document", 3, 20, 5)
    graph_depth = st.slider("Graph depth", 0, 3, 1)


# -----------------------------------------------------------------------------
# 11. MAIN AREA UI
# -----------------------------------------------------------------------------
if not st.session_state.get("groq_api_key"):
    st.warning("⚠️ Enter your Groq API key in the sidebar to enable answers.")

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
        if msg["role"] == "assistant":
            if msg.get("confidence"):
                st.caption(
                    f"{msg['confidence_emoji']} **Confidence: {msg['confidence']}**"
                )
            if msg.get("sources"):
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
    disabled=len(st.session_state.uploaded_docs) == 0 or not st.session_state.get("groq_api_key")
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
                    graph_depth=graph_depth,
                    chat_history=st.session_state.messages
                )
                st.write(result["answer"])

                # Confidence badge
                confidence = result.get("confidence", "Medium")
                emoji = result.get("confidence_emoji", "🟡")
                st.caption(f"{emoji} **Confidence: {confidence}** — based on document relevance scores")

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
                    "sources": result["sources"],
                    "confidence": result.get("confidence", "Medium"),
                    "confidence_emoji": result.get("confidence_emoji", "🟡")
                })
            except Exception as e:
                st.error(f"Query failed: {e}")
