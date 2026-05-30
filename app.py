import os
import json
import numpy as np
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from embedder import Embedder
from graph_builder import GraphBuilder
from retrieval.bm25_index import BM25Retriever
from retrieval.vector_search import VectorSearch
from retrieval.graph_retriever import GraphRetriever
from retrieval.fusion import ResultFusion
from retrieval.reranker import Reranker
from llm import LLMGenerator

# --- App Config ---
st.set_page_config(page_title="Adaptive GraphRAG", layout="wide")
st.title("🕸️ Adaptive GraphRAG")

# --- Session State Initialization ---
@st.cache_resource
def load_system():
    """Loads all models and indices into memory once."""
    print("Loading system components...")
    
    # 1. Load Data
    try:
        with open("graph/chunk_store.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load("embeddings/embeddings.npy")
    except Exception as e:
        return None, f"Data not found. Please run the ingestion pipeline first. ({e})"
        
    embedder = Embedder()
    
    gb = GraphBuilder()
    try:
        gb.load_graph()
    except:
        return None, "Graph not found. Run ingestion first."
        
    bm25 = BM25Retriever()
    try:
        bm25.load_index()
    except:
        return None, "BM25 index not found. Run ingestion first."
        
    vs = VectorSearch()
    try:
        vs.load_index()
    except:
        return None, "FAISS index not found. Run ingestion first."
        
    gr = GraphRetriever(gb)
    fusion = ResultFusion()
    reranker = Reranker()
    llm = LLMGenerator()
    
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "embedder": embedder,
        "gb": gb,
        "bm25": bm25,
        "vs": vs,
        "gr": gr,
        "fusion": fusion,
        "reranker": reranker,
        "llm": llm
    }, None

systems, error = load_system()

if error:
    st.error(error)
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Retrieval Parameters")
top_k_initial = st.sidebar.slider("Initial Top-K (BM25 & Vector)", 5, 50, 10)
graph_depth = st.sidebar.slider("Graph Traversal Depth", 0, 3, 1)
use_bfs = st.sidebar.checkbox("Use BFS for Graph (Uncheck for DFS)", value=True)
top_k_final = st.sidebar.slider("Final Top-K (After Reranking)", 1, 10, 5)

# --- Helper function for PyVis ---
def generate_graph_html(chunks_with_scores, graph: nx.Graph):
    """Generates a PyVis HTML visualization of the retrieved neighborhood."""
    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add retrieved nodes
    retrieved_ids = [c["chunk_id"] for c, _ in chunks_with_scores]
    
    # We want to pull a small subgraph for visualization
    subgraph_nodes = set(retrieved_ids)
    
    # Add 1-hop neighbors for visual context
    for node_id in retrieved_ids:
        if node_id in graph:
            subgraph_nodes.update(graph.neighbors(node_id))
            
    subgraph = graph.subgraph(subgraph_nodes)
    
    for node in subgraph.nodes:
        # Color retrieved nodes red, others blue
        color = "#ff4b4b" if node in retrieved_ids else "#1f77b4"
        title = subgraph.nodes[node].get("text", "Unknown text")[:100] + "..."
        net.add_node(node, label=node[:8], title=title, color=color)
        
    for u, v in subgraph.edges:
        net.add_edge(u, v)
        
    html = net.generate_html()
    return html

# --- Main Interface ---
query = st.text_input("Ask a question based on your documents:", placeholder="e.g., What are the key features of the project?")

if st.button("Search") and query:
    if not os.getenv("GROQ_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Generating answer will fail.")
        
    with st.spinner("Executing Hybrid Retrieval Pipeline..."):
        # 1. Lexical Search
        bm25_res = systems["bm25"].bm25_search(query, top_k=top_k_initial)
        
        # 2. Vector Search
        q_emb = systems["embedder"].generate_query_embedding(query)
        vec_res = systems["vs"].vector_search(q_emb, top_k=top_k_initial)
        
        # 3. Graph Expansion
        graph_res = []
        if graph_depth > 0:
            seeds = [res[0] for res in vec_res[:3]] # Use top 3 vector hits as seeds
            graph_res = systems["gr"].graph_based_retrieval(seeds, max_depth=graph_depth, use_bfs=use_bfs)
            
        # 4. Fusion
        fused_res = systems["fusion"].fuse_results(bm25_res, vec_res, graph_res, top_k=20)
        
        # 5. Reranking
        final_res = systems["reranker"].rerank(query, fused_res, top_k=top_k_final)
        
        # 6. LLM Generation
        answer = systems["llm"].generate_answer(query, final_res)
        
    st.subheader("🤖 Generated Answer")
    st.write(answer)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Retrieved Context (Top Chunks)")
        for i, (chunk, score) in enumerate(final_res):
            with st.expander(f"[{i+1}] Source: {chunk.get('source_filename')} (Page {chunk.get('page_number')}) | Score: {score:.4f}"):
                st.write(chunk.get("text"))
                
    with col2:
        st.subheader("🕸️ Local Graph Neighborhood")
        if final_res and systems["gb"].graph:
            try:
                graph_html = generate_graph_html(final_res, systems["gb"].graph)
                components.html(graph_html, height=410)
            except Exception as e:
                st.error(f"Failed to generate graph visualization: {e}")
        else:
            st.info("Graph visualization not available.")