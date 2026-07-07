# -*- coding: utf-8 -*-
import sys
# Mask tensorflow to prevent import conflicts with streamlit's protobuf requirements
sys.modules['tensorflow'] = None

import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import numpy as np
import streamlit as st
import shutil
import re
import pickle
import time
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
from retrieval.query_classifier import QueryClassifier
from llm import LLMGenerator

# Phase 4 Answer Intelligence imports
from citation_verifier import CitationVerifier
from claim_extractor import ClaimExtractor
from claim_verifier import ClaimVerifier

# Workspace database & providers
from storage import Database
from providers import MultiProviderManager, get_provider_health_stats

# -----------------------------------------------------------------------------
# 2. PAGE CONFIG
# -----------------------------------------------------------------------------
import base64

st.set_page_config(
    page_title="GraphRAG Workspace",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load base64 logo for HTML rendering
def load_base64_logo():
    try:
        with open("logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

base64_logo = load_base64_logo()

# -----------------------------------------------------------------------------
# 3. CUSTOM CSS - ChatGPT-like Dark Theme with Glassmorphism
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0d0d0d;
    --bg-sidebar: #171717;
    --bg-surface: #212121;
    --bg-hover: #2f2f2f;
    --bg-input: #2f2f2f;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-light: rgba(255, 255, 255, 0.05);
    --text-primary: #ececec;
    --text-secondary: #b4b4b4;
    --text-muted: #8e8e8e;
    --text-dim: #666666;
    --accent: #10a37f;
}

/* ─── Body ─── */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp > header { background: transparent !important; }
div[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1rem !important; max-width: 900px !important; }

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-light) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}
/* Sidebar action buttons in column rows: subtle by default, visible on hover */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button {
    min-height: 0 !important;
    padding: 0.15rem 0.3rem !important;
    font-size: 0.72rem !important;
    line-height: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stPopover"] {
    display: flex !important;
    justify-content: flex-end !important;
}
section[data-testid="stSidebar"] div[data-testid="stPopover"] button {
    background: transparent !important;
    border: none !important;
    color: var(--text-dim) !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 28px !important;
    height: 28px !important;
    line-height: 28px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
section[data-testid="stSidebar"] div[data-testid="stPopover"] button svg {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stPopover"] button:hover {
    color: var(--text-primary) !important;
    background: var(--bg-hover) !important;
}
div[data-testid="stPopoverBody"] {
    background-color: #212121 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    padding: 4px !important;
    min-width: 140px !important;
}
div[data-testid="stPopoverBody"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    color: #ececec !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    position: relative !important;
}
div[data-testid="stPopoverBody"] button:hover {
    background: #2f2f2f !important;
    color: white !important;
}
/* Vector SVG mask icons matching ChatGPT style */
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] > div:nth-child(1) button::before {
    content: "" !important;
    position: absolute !important;
    left: 12px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 14px !important;
    height: 14px !important;
    background-color: var(--text-secondary) !important;
    -webkit-mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'><path d='M12 20h9'/><path d='M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z'/></svg>") no-repeat center !important;
    mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'><path d='M12 20h9'/><path d='M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z'/></svg>") no-repeat center !important;
    -webkit-mask-size: contain !important;
    mask-size: contain !important;
}
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] > div:nth-child(1) button:hover::before {
    background-color: white !important;
}
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button {
    color: #f87171 !important;
}
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button::before {
    content: "" !important;
    position: absolute !important;
    left: 12px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 14px !important;
    height: 14px !important;
    background-color: #f87171 !important;
    -webkit-mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'><path d='M3 6h18'/><path d='M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6'/><path d='M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2'/><line x1='10' x2='10' y1='11' y2='17'/><line x1='14' x2='14' y1='11' y2='17'/></svg>") no-repeat center !important;
    mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'><path d='M3 6h18'/><path d='M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6'/><path d='M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2'/><line x1='10' x2='10' y1='11' y2='17'/><line x1='14' x2='14' y1='11' y2='17'/></svg>") no-repeat center !important;
    -webkit-mask-size: contain !important;
    mask-size: contain !important;
}
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button:hover {
    background: rgba(248, 113, 113, 0.08) !important;
    color: #fca5a5 !important;
}
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button:hover::before {
    background-color: #fca5a5 !important;
}

/* ─── Buttons ─── */
div.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.85rem !important;
    transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
}
div.stButton > button[kind="primary"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--bg-hover) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
    transform: none !important;
}
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-muted) !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background-color: transparent !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.25rem !important;
}
.streamlit-expanderContent {
    background-color: transparent !important;
    border-left: 1px solid var(--border-light) !important;
    border-right: 1px solid var(--border-light) !important;
    border-bottom: 1px solid var(--border-light) !important;
    border-radius: 0 0 8px 8px !important;
    padding: 0.75rem 1rem !important;
    margin-top: -0.35rem !important;
}

/* ─── Chat Messages ─── */
div[data-testid="stChatMessageAvatar"] {
    display: none !important;
}
div[data-testid="stChatMessage"] {
    padding: 0.5rem 0 !important;
    background: transparent !important;
    border: none !important;
    gap: 0 !important;
}
div[data-testid="stChatMessageContent"] {
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    max-width: 100% !important;
    box-shadow: none !important;
}
/* User messages: subtle pill */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has(img[alt="user"]) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) div[data-testid="stChatMessageContent"] {
    background: var(--bg-hover) !important;
    border-radius: 18px !important;
    padding: 0.85rem 1.25rem !important;
    border: none !important;
    max-width: 82% !important;
    margin-left: auto !important;
}
/* Assistant messages: transparent, no bubble */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has(img[alt="assistant"]) div[data-testid="stChatMessageContent"],
div[data-testid="stChatMessage"]:has(svg[aria-label="assistant"]) div[data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.5rem 0 !important;
    margin-left: 0 !important;
    max-width: 100% !important;
}
/* User message row: right-align the bubble */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
div[data-testid="stChatMessage"]:has(img[alt="user"]),
div[data-testid="stChatMessage"]:has(svg[aria-label="user"]) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}
/* Assistant message row: left-align */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
div[data-testid="stChatMessage"]:has(img[alt="assistant"]),
div[data-testid="stChatMessage"]:has(svg[aria-label="assistant"]) {
    flex-direction: row !important;
}

/* ─── Chat Input: Premium Surface Treatment ─── */
.stChatInput {
    max-width: 800px !important;
    margin: 0 auto !important;
}
.stChatInput > div {
    background: linear-gradient(180deg, #1e2028 0%, #1a1b20 50%, #17181d 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 26px !important;
    overflow: hidden !important;
    box-shadow:
        inset 0 1px 0 0 rgba(255, 255, 255, 0.04),
        0 1px 3px 0 rgba(0, 0, 0, 0.25),
        0 4px 12px -2px rgba(0, 0, 0, 0.15) !important;
    transition: border-color 200ms ease-in-out, box-shadow 200ms ease-in-out !important;
}
.stChatInput > div:hover {
    border-color: rgba(255, 255, 255, 0.14) !important;
}
.stChatInput > div:focus-within {
    border-color: rgba(255, 255, 255, 0.18) !important;
    box-shadow:
        inset 0 1px 0 0 rgba(255, 255, 255, 0.05),
        0 0 0 2px rgba(59, 130, 246, 0.08),
        0 1px 3px 0 rgba(0, 0, 0, 0.25) !important;
}
/* Reset ALL inner elements: strip Streamlit default backgrounds, borders, outlines */
.stChatInput div,
.stChatInput form,
.stChatInput [data-baseweb],
.stChatInput [data-baseweb] > div {
    background: transparent !important;
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}
.stChatInput textarea,
.stChatInput input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    padding-left: 10px !important;
    caret-color: rgba(59, 130, 246, 0.7) !important;
}
.stChatInput textarea:focus,
.stChatInput input:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
.stChatInput textarea::placeholder {
    color: rgba(255, 255, 255, 0.28) !important;
}
/* Send button – embedded inside the bar */
.stChatInput button[kind="primary"],
.stChatInput button[data-testid="stChatInputSubmitButton"] {
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    margin: 4px 8px 4px 2px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #e5e5e5 !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    transition: transform 200ms ease-in-out, background 200ms ease-in-out, box-shadow 200ms ease-in-out !important;
    flex-shrink: 0 !important;
}
.stChatInput button[kind="primary"]:hover,
.stChatInput button[data-testid="stChatInputSubmitButton"]:hover {
    background: white !important;
    transform: scale(1.02) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
}
.stChatInput button svg { color: #0d0d0d !important; }

/* ─── File Uploader ─── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255, 255, 255, 0.18) !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border-light) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-dim) !important;
    padding: 0.6rem 1.25rem !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.12s ease !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--text-primary) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-secondary) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--text-primary) !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ─── Thinking Indicator ─── */
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
.thinking-indicator {
    display: flex; align-items: center; gap: 8px;
    padding: 0.6rem 0; color: var(--text-muted);
    font-size: 0.82rem; font-weight: 400;
}
.thinking-dots { display: flex; gap: 4px; }
.thinking-dots span {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--text-muted);
    animation: dotPulse 1.4s infinite ease-in-out both;
}
.thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
.thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
.thinking-dots span:nth-child(3) { animation-delay: 0s; }

/* ─── Response Actions ─── */
.response-actions {
    display: flex; align-items: center; gap: 2px;
    margin-top: 0.4rem; padding-top: 0.2rem;
}
.response-action-btn {
    background: transparent; border: none; color: var(--text-dim);
    cursor: pointer; padding: 4px 8px; border-radius: 6px;
    font-size: 0.75rem; display: inline-flex; align-items: center; gap: 4px;
    transition: color 0.12s ease, background 0.12s ease;
}
.response-action-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
.response-action-btn svg { width: 14px; height: 14px; }
.copy-toast {
    position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
    background: var(--bg-surface); color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 8px 16px; border-radius: 8px; font-size: 0.8rem; font-weight: 500;
    z-index: 9999; animation: toastFade 2s ease forwards;
}
@keyframes toastFade {
    0% { opacity: 0; transform: translateX(-50%) translateY(10px); }
    15% { opacity: 1; transform: translateX(-50%) translateY(0); }
    70% { opacity: 1; }
    100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
}

/* ─── Welcome Hero ─── */
.hero-container {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 3rem 1.5rem; text-align: center; max-width: 600px; margin: 3rem auto;
    background: transparent !important; border: none !important;
}
.hero-title {
    font-size: 1.5rem; font-weight: 600; margin-bottom: 0.4rem;
    color: var(--text-primary) !important;
}
.hero-subtitle {
    color: var(--text-muted); font-size: 0.88rem; line-height: 1.6;
    margin-bottom: 1.5rem; max-width: 460px;
}
.onboarding-steps {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 12px; width: 100%; margin-top: 0.5rem;
}
.onboarding-step-card {
    background: var(--bg-surface); border: 1px solid var(--border-light);
    border-radius: 12px; padding: 1.25rem 1rem; text-align: center;
    transition: border-color 0.2s ease;
}
.onboarding-step-card:hover { border-color: rgba(255,255,255,0.12); }
.onboarding-step-icon { font-size: 1.2rem; margin-bottom: 0.5rem; color: var(--text-muted); }
.onboarding-step-title { font-weight: 600; color: var(--text-primary); font-size: 0.82rem; margin-bottom: 0.2rem; }
.onboarding-step-desc { font-size: 0.72rem; color: var(--text-muted); line-height: 1.4; }

/* ─── Document cards ─── */
.document-card {
    background: var(--bg-surface); border: 1px solid var(--border-light);
    border-radius: 8px; padding: 0.4rem 0.6rem; margin-bottom: 0.35rem;
    display: flex; align-items: center; justify-content: space-between;
}
.document-title {
    font-size: 0.78rem; font-weight: 500; color: var(--text-primary);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 170px;
}
.document-meta { font-size: 0.68rem; color: var(--text-dim); }

/* ─── Header badges (simplified) ─── */
.header-badge-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.2rem; }
.header-badge {
    background: transparent !important; color: var(--text-dim) !important;
    border: none !important; padding: 0 !important;
    font-size: 0.72rem !important; font-weight: 400 !important;
}

/* ─── KPI cards ─── */
.kpi-card {
    background: var(--bg-surface); border: 1px solid var(--border-light);
    border-radius: 12px; padding: 1.25rem; text-align: center;
}
.kpi-val { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.15rem; }
.kpi-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.03em; }

/* Badges */
.badge-fast {
    background: rgba(16, 185, 129, 0.06) !important; color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.12) !important;
    padding: 2px 8px !important; border-radius: 6px !important;
    font-size: 0.7rem !important; font-weight: 500 !important;
}
.badge-verified {
    background: rgba(14, 165, 233, 0.06) !important; color: #38bdf8 !important;
    border: 1px solid rgba(14, 165, 233, 0.12) !important;
    padding: 2px 8px !important; border-radius: 6px !important;
    font-size: 0.7rem !important; font-weight: 500 !important;
}

/* Input indicator bar */
.input-indicator-bar {
    display: flex; align-items: center; gap: 6px;
    max-width: 800px; margin: 0.3rem auto; justify-content: center;
    font-size: 0.72rem; color: var(--text-dim);
}

/* ─── Select boxes & inputs ─── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}
</style>

""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. DATABASE INITIALIZATION
# -----------------------------------------------------------------------------
db = Database.get_db()


# -----------------------------------------------------------------------------
# 5. CACHED LOADER: get_shared_models()
# -----------------------------------------------------------------------------
@st.cache_resource
def get_shared_models() -> dict:
    return {
        "embedder": Embedder(),
        "llm": LLMGenerator(),
        "reranker": Reranker(),
        "hyde": HyDEGenerator(),
        "decomposer": QueryDecomposer(),
        "classifier": QueryClassifier(),
        "citation_verifier": CitationVerifier(),
        "claim_extractor": ClaimExtractor(),
        "claim_verifier": ClaimVerifier()
    }


# -----------------------------------------------------------------------------
# 6. HELPER: load_doc_systems / get_doc_system (Chat-Isolated)
# -----------------------------------------------------------------------------
def load_doc_systems(user_id: str, chat_id: str, doc_name: str) -> dict:
    """Loads all indexes for a single document into memory."""
    doc_path = f"doc_store/users/{user_id}/chats/{chat_id}/{doc_name}"

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
def get_doc_system(user_id: str, chat_id: str, doc_name: str) -> dict:
    return load_doc_systems(user_id, chat_id, doc_name)


# -----------------------------------------------------------------------------
# 7. API KEY ENVIRONMENT INJECTION HELPER
# -----------------------------------------------------------------------------
def inject_user_api_keys(user_id: str) -> dict:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    keys = db.list_api_keys(user_id)
    original_keys = {}
    for k in keys:
        if k["is_active"] != 1:
            continue
        decrypted = k.get("decrypted_key", "").strip()
        if not decrypted:
            continue
        p = k["provider"]
        
        env_names = []
        if p == "groq":
            env_names = ["GROQ_API_KEY"]
        elif p == "openai":
            env_names = ["OPENAI_API_KEY"]
        elif p == "gemini":
            env_names = ["GEMINI_API_KEY"]
        elif p == "claude":
            env_names = ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
            
        for env_name in env_names:
            original_keys[env_name] = os.environ.get(env_name)
            os.environ[env_name] = decrypted
            
    return original_keys

def restore_api_keys(original_keys: dict):
    for env_name, original_val in original_keys.items():
        if original_val is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = original_val


# -----------------------------------------------------------------------------
# 8. INGESTION WITH PROGRESS (Chat-Isolated)
# -----------------------------------------------------------------------------
def ingest_pdf_with_progress(uploaded_file, progress_bar, status_text, user_id: str, chat_id: str) -> dict:
    """Ingests a PDF file to the isolated chat folder and generates indexes."""
    filename = uploaded_file.name
    doc_name = re.sub(r'[^a-zA-Z0-9]', '_', filename.replace(".pdf", "").replace(".PDF", "")).lower()
    doc_path = f"doc_store/users/{user_id}/chats/{chat_id}/{doc_name}"
    os.makedirs(doc_path, exist_ok=True)

    try:
        # Step 1 - Save PDF
        source_path = f"{doc_path}/source.pdf"
        with open(source_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Step 2 - Chunk
        status_text.text("Chunking document...")
        progress_bar.progress(20, text="Chunking...")
        chunker = DocumentChunker(chunk_size_words=200, overlap_words=50)
        chunks = chunker.process_pdf(source_path)
        for chunk in chunks:
            chunk["source_filename"] = filename
        with open(f"{doc_path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        # Step 3 - Embed
        status_text.text("Generating embeddings...")
        progress_bar.progress(45, text="Embedding...")
        embedder = get_shared_models()["embedder"]
        embeddings = embedder.generate_embeddings(chunks)
        np.save(f"{doc_path}/embeddings.npy", embeddings)

        # Step 4 - FAISS + BM25
        status_text.text("Building search indexes...")
        progress_bar.progress(65, text="Indexing...")
        vs = VectorSearch()
        vs.build_faiss_index(chunks, embeddings)
        vs.save_index(
            index_path=f"{doc_path}/faiss.index",
            chunks_path=f"{doc_path}/faiss_chunks.pkl"
        )
        bm25 = BM25Retriever()
        bm25.build_bm25_index(chunks)
        bm25.save_index(output_path=f"{doc_path}/bm25_index.pkl")

        # Step 5 - Knowledge graph
        status_text.text("Building knowledge graph...")
        progress_bar.progress(85, text="Knowledge Graph...")
        gb = GraphBuilder()
        gb.build_graph(chunks, embeddings)
        gb.save_graph(output_path=f"{doc_path}/graph.pkl")

        # Step 6 - Document Summary Generation
        status_text.text("Summarizing document...")
        progress_bar.progress(90, text="Summarizing...")
        doc_text = "\n\n".join([c.get("text", "") for c in chunks])
        llm = get_shared_models()["llm"]
        embedder = get_shared_models()["embedder"]
        doc_summary = llm.summarize_document(doc_text)
        summary_emb = embedder.generate_query_embedding(doc_summary).tolist()

        # Write meta
        meta = {
            "filename": filename,
            "doc_name": doc_name,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "page_count": len(set([c.get("page_number", 0) for c in chunks])),
            "chunk_count": len(chunks),
            "status": "ready",
            "document_summary": doc_summary,
            "summary_embedding": summary_emb
        }
        with open(f"{doc_path}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return meta

    except Exception as e:
        shutil.rmtree(doc_path, ignore_errors=True)
        raise e


# -----------------------------------------------------------------------------
# 9. CORE: query_all_docs()
# -----------------------------------------------------------------------------
def verify_citations(answer: str, sources: list) -> str:
    """
    Verifies that source citations in the answer match actual retrieved chunks.
    Marks unverified citations with a warning.
    """
    retrieved_chunks = []
    for s in sources:
        chunk_dict = {
            "source_filename": s.get("filename", ""),
            "page_number": s.get("page", 0),
            "text": s.get("text", "")
        }
        retrieved_chunks.append((chunk_dict, s.get("score", 0.0)))
        
    shared = get_shared_models()
    cit_verifier = shared["citation_verifier"]
    cit_res = cit_verifier.verify_citations(answer, retrieved_chunks)
    
    # Format markers for UI
    v_answer = cit_res["verified_answer"]
    v_answer = v_answer.replace(" [Verified]", " <span style='color:#10b981; font-weight:600; font-size: 0.82em;'>[Verified]</span>")
    v_answer = v_answer.replace(" [Unverified]", " <span style='color:#ef4444; font-weight:600; font-size: 0.82em;'>[Unverified]</span>")
    return v_answer

def clean_answer_formatting(text: str) -> str:
    """Convert • bullet characters to markdown - bullets Streamlit can render."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('•'):
            indent = len(line) - len(stripped)
            line = ' ' * indent + '- ' + stripped[1:].lstrip()
        cleaned.append(line)
    return '\n'.join(cleaned)

def parse_llm_response(raw: str) -> tuple:
    """
    Splits LLM response into (clean_answer, confidence_label, confidence_emoji).
    LLM self-reports confidence — accurate regardless of query typos.
    """
    confidence_map = {
        "HIGH":   ("High",   ""),
        "MEDIUM": ("Medium", ""),
        "LOW":    ("Low",    ""),
    }
    label, emoji = "Medium", ""
    answer = raw.strip()

    for key, (l, e) in confidence_map.items():
        tag = f"CONFIDENCE: {key}"
        if tag in raw:
            label, emoji = l, e
            answer = raw.replace(tag, "").strip()
            break

    answer = clean_answer_formatting(answer)
    return answer, label, emoji

def query_all_docs(query: str, user_id: str, chat_id: str, doc_names: list, top_k: int = 10, graph_depth: int = 1, chat_history: list = None, mode: str = "FAST", active_provider: str = "groq", active_model: str = None) -> dict:
    import time
    from performance_tracker import PerformanceTracker
    tracker = PerformanceTracker()
    tracker.reset_call_count()
    
    t_total_start = time.perf_counter()

    shared = get_shared_models()
    embedder = shared["embedder"]
    reranker = shared["reranker"]
    llm = shared["llm"]
    hyde = shared["hyde"]
    decomposer = shared["decomposer"]
    classifier = shared["classifier"]
    claim_extractor = shared["claim_extractor"]
    claim_verifier = shared["claim_verifier"]

    # 1. Timing: Classification & Planning
    t_start = time.perf_counter()
    query_type = classifier.classify(query)

    traversal = "BFS"
    use_decomposition = False
    
    if query_type == "SIMPLE":
        graph_depth = 1
        traversal = "BFS"
        use_decomposition = False
    elif query_type == "MODERATE":
        graph_depth = 2
        traversal = "BFS"
        use_decomposition = False
    elif query_type == "COMPLEX":
        graph_depth = 3
        traversal = "PPR"
        use_decomposition = True
    elif query_type == "ALGORITHM":
        graph_depth = 2
        traversal = "BFS"
        use_decomposition = False
    elif query_type == "RESEARCH":
        graph_depth = 3
        traversal = "HYBRID"
        use_decomposition = True
    classification_ms = (time.perf_counter() - t_start) * 1000

    # 2. Timing: Query Decomposition
    t_start = time.perf_counter()
    if use_decomposition and mode != "FAST":
        sub_questions = decomposer.decompose(query)
    else:
        sub_questions = [query]
    decomposition_ms = (time.perf_counter() - t_start) * 1000

    hyde_ms = 0.0
    retrieval_ms = 0.0
    graph_expansion_ms = 0.0

    merged_candidates = []

    # 3. Timing: Retrieve candidates per sub-question
    for sub_q in sub_questions:
        # HyDE - skip in FAST mode to avoid expensive LLM call
        t_sub_start = time.perf_counter()
        if mode == "FAST":
            q_emb_search = embedder.generate_query_embedding(sub_q)
            q_emb_original = q_emb_search
        else:
            hypothesis = hyde.generate_hypothesis(sub_q)
            q_emb_search = embedder.generate_query_embedding(hypothesis)
            q_emb_original = embedder.generate_query_embedding(sub_q)
        hyde_ms += (time.perf_counter() - t_sub_start) * 1000

        # Retrieval (BM25 + Vector)
        for doc_name in doc_names:
            t_ret_start = time.perf_counter()
            sys = get_doc_system(user_id, chat_id, doc_name)

            bm25_res = sys["bm25"].bm25_search(sub_q, top_k=10)
            vec_res = sys["vs"].vector_search(q_emb_search, top_k=10)
            retrieval_ms += (time.perf_counter() - t_ret_start) * 1000

            t_graph_start = time.perf_counter()
            graph_res = []
            if graph_depth > 0:
                seeds = [res[0] for res in sys["vs"].vector_search(q_emb_original, top_k=3)]
                graph_res = sys["gr"].graph_based_retrieval(
                    seeds, 
                    max_depth=graph_depth, 
                    use_bfs=True,
                    strategy=traversal
                )
            graph_expansion_ms += (time.perf_counter() - t_graph_start) * 1000

            fused_res = sys["fusion"].fuse_results(bm25_res, vec_res, graph_res, top_k=10)
            merged_candidates.extend(fused_res)

    # 4. Global Deduplication
    seen_ids = set()
    unique_candidates = []
    for chunk, score in merged_candidates:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_candidates.append((chunk, score))

    unique_candidates.sort(key=lambda x: x[1], reverse=True)
    unique_candidates = unique_candidates[:50]

    # Reranking
    t_start = time.perf_counter()
    final_res = reranker.rerank(query, unique_candidates, top_k=top_k, query_type=query_type)
    rerank_ms = (time.perf_counter() - t_start) * 1000

    # Load active document summaries
    document_summaries = []
    for doc_name in doc_names:
        meta_path = f"doc_store/users/{user_id}/chats/{chat_id}/{doc_name}/meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "document_summary" in meta:
                    document_summaries.append(f"Document: {meta.get('filename', doc_name)}\nSummary: {meta['document_summary']}")
            except:
                pass

    # Context construction
    context = llm.build_context(final_res, document_summaries)

    system_prompt = """You are a precise Q&A assistant. Answer ONLY from the provided Context.

STRICT RULES:
- Only use information from the Context. If not found, say "I cannot answer this based on the provided documents."
- Cite sources inline using the source filename and page number like [filename.pdf, Page X].
- NEVER repeat a sentence, phrase, or conclusion you have already written.
- NEVER write filler like "In conclusion", "Overall", "In summary" more than once.
- NEVER pad the answer. Each sentence must add NEW information.
- If the user asks for a specific length, cover MORE topics and details — never repeat points.
- If the user asks a follow-up question, use the conversation history to understand what they are referring to.
- Stop writing the moment you have no new information to add.

FORMAT RULES:
- Always answer using markdown formatting.
- Start with a one-line definition as plain text.
- Then use markdown bullet points (start each point with "- ").
- Each bullet = one distinct fact with its source citation.
- If the user asks for code or an algorithm, present it in a clean code block (```).
  Use the pseudocode/algo notation from the source — do not convert to Python.
  If the OCR text looks garbled or corrupted, use your understanding of the algorithm
  to present a clean readable version in the same pseudocode style.
- Keep each bullet concise — one idea per bullet, max 2 lines.

After your answer, on a NEW LINE, output exactly one of these — nothing else on that line:
CONFIDENCE: HIGH
CONFIDENCE: MEDIUM
CONFIDENCE: LOW

Use HIGH if the context directly and fully answers the question.
Use MEDIUM if the context partially answers or required some inference.
Use LOW if the answer is not clearly in the context or you said you cannot answer."""

    formatted_history = []
    if chat_history:
        for m in chat_history[-4:]:
            if m["role"] in ("user", "assistant"):
                formatted_history.append({"role": m["role"], "content": m.get("content", "")})

    user_content = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # 5. LLM Answer Generation with Provider Routing
    t_start = time.perf_counter()
    provider_manager = MultiProviderManager(db.list_api_keys(user_id))
    
    raw_answer, final_p, final_m = provider_manager.generate(
        prompt=user_content,
        system_prompt=system_prompt,
        chat_history=formatted_history,
        target_provider=active_provider,
        model=active_model
    )
    answer, confidence_label, confidence_emoji = parse_llm_response(raw_answer)
    generation_ms = (time.perf_counter() - t_start) * 1000
    tracker.increment_llm_calls()

    sources = []
    for chunk, score in final_res:
        sources.append({
            "filename": chunk.get("source_filename", "Unknown"),
            "page": chunk.get("page_number", 0),
            "score": score,
            "text": chunk.get("text", "")
        })

    # 6. Verification Stages
    claims_list = []
    grounding_score = 100.0
    trust_level = "VERIFIED"
    hallucination_risk = "LOW"
    claim_extraction_ms = 0.0
    claim_verification_ms = 0.0
    citation_verification_ms = 0.0
    grounding_ms = 0.0

    if mode == "VERIFIED":
        # Claim Extraction
        t_start = time.perf_counter()
        claims = claim_extractor.extract_claims(answer)
        claim_extraction_ms = (time.perf_counter() - t_start) * 1000
        
        # Claim Verification
        t_start = time.perf_counter()
        verify_res = claim_verifier.verify_claims(claims, final_res)
        claims_list = verify_res["claims"]
        grounding_score = verify_res["grounding_score"]
        trust_level = verify_res["trust_level"]
        hallucination_risk = verify_res["hallucination_risk"]
        claim_verification_ms = (time.perf_counter() - t_start) * 1000

        # Citation Verification
        t_start = time.perf_counter()
        answer = verify_citations(answer, sources)
        citation_verification_ms = (time.perf_counter() - t_start) * 1000

    total_ms = (time.perf_counter() - t_total_start) * 1000

    stages_timings = {
        "classification": classification_ms,
        "hyde": hyde_ms,
        "decomposition": decomposition_ms,
        "retrieval": retrieval_ms,
        "graph_expansion": graph_expansion_ms,
        "rerank": rerank_ms,
        "generation": generation_ms,
        "claim_extraction": claim_extraction_ms,
        "claim_verification": claim_verification_ms,
        "citation_verification": citation_verification_ms,
        "grounding": grounding_ms
    }
    slowest_stage = max(stages_timings, key=stages_timings.get)
    slowest_stage_ms = stages_timings[slowest_stage]

    perf_trace = {
        "total_ms": total_ms,
        "slowest_stage": slowest_stage,
        "slowest_stage_ms": round(slowest_stage_ms, 1),
        "classification_ms": round(classification_ms, 1),
        "hyde_ms": round(hyde_ms, 1),
        "decomposition_ms": round(decomposition_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "graph_expansion_ms": round(graph_expansion_ms, 1),
        "rerank_ms": round(rerank_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "claim_extraction_ms": round(claim_extraction_ms, 1),
        "claim_verification_ms": round(claim_verification_ms, 1),
        "citation_verification_ms": round(citation_verification_ms, 1),
        "grounding_ms": round(grounding_ms, 1),
        "claims_count": len(claims_list),
        "avg_per_claim_ms": round((claim_verification_ms / len(claims_list)) if claims_list else 0.0, 1)
    }
    tracker.log_trace(perf_trace)

    trace = {
        "query_type": query_type,
        "sub_questions": sub_questions,
        "traversal_used": traversal,
        "graph_depth": graph_depth,
        "retrieval_strategy": f"BM25 + Vector + Graph ({traversal})",
        "candidate_pool_size": len(unique_candidates),
        "performance": perf_trace
    }

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence_label,
        "confidence_emoji": confidence_emoji,
        "grounding_score": grounding_score,
        "trust_level": trust_level,
        "hallucination_risk": hallucination_risk,
        "claims": claims_list,
        "trace": trace,
        "performance": perf_trace
    }


# -----------------------------------------------------------------------------
# 10. HELPER: delete_doc() - Database & File Sync
# -----------------------------------------------------------------------------
def delete_doc(user_id: str, chat_id: str, document_id: str, doc_name: str):
    doc_path = f"doc_store/users/{user_id}/chats/{chat_id}/{doc_name}"
    shutil.rmtree(doc_path, ignore_errors=True)
    db.delete_document(document_id)
    get_doc_system.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# 10.1 HELPERS: resolve_unique_title() & generate_workspace_title()
# -----------------------------------------------------------------------------
def resolve_unique_title(user_id: str, base_title: str) -> str:
    chats = db.list_chats(user_id)
    existing_titles = {c["title"].lower() for c in chats}
    
    if base_title.lower() not in existing_titles:
        return base_title
        
    suffix = 1
    while True:
        candidate = f"{base_title} ({suffix})"
        if candidate.lower() not in existing_titles:
            return candidate
        suffix += 1


def generate_workspace_title(user_id: str, query: str, active_chat: dict) -> str:
    # 1. Fallback text title (first 5 words)
    words = query.strip().split()
    fallback_title = " ".join(words[:5]) if len(words) > 5 else " ".join(words)
    if len(fallback_title) > 30:
        fallback_title = fallback_title[:27] + "..."
    
    if active_chat:
        provider = active_chat["model_provider"]
        model = active_chat["model_name"]
    else:
        provider = "groq"
        model = "llama-3.3-70b-versatile"
        
    try:
        from providers import MultiProviderManager
        provider_manager = MultiProviderManager(db.list_api_keys(user_id))
        
        prompt = (
            "Summarize the following user query into an extremely short, concise title (maximum 3 to 4 words). "
            "Do not include quotes, markdown formatting, or any introductory text.\n\n"
            f"Query: {query}\n\nTitle:"
        )
        
        raw_title, _, _ = provider_manager.generate(
            prompt=prompt,
            target_provider=provider,
            model=model,
            max_tokens=15,
            temperature=0.1
        )
        
        # Clean up the generated title
        clean_title = raw_title.strip().replace('"', '').replace("'", "").replace(".", "").split("\n")[0].strip()
        if clean_title.lower().startswith("title:"):
            clean_title = clean_title[6:].strip()
            
        if clean_title and len(clean_title) <= 40:
            return clean_title
    except Exception:
        pass
        
    return fallback_title


# -----------------------------------------------------------------------------
# 11. HELPER: get_combined_graph()
# -----------------------------------------------------------------------------
def get_combined_graph(user_id: str, chat_id: str, docs: list) -> GraphBuilder:
    import networkx as nx
    combined = nx.Graph()
    for doc in docs:
        doc_path = f"doc_store/users/{user_id}/chats/{chat_id}/{doc['doc_name']}/graph.pkl"
        if os.path.exists(doc_path):
            try:
                with open(doc_path, "rb") as f:
                    g = pickle.load(f)
                combined = nx.compose(combined, g)
            except:
                pass
    builder = GraphBuilder()
    builder.graph = combined
    return builder


# -----------------------------------------------------------------------------
# 12. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "user" not in st.session_state or st.session_state.user is None:
    # Auto-initialize user from DB or create if missing
    dev_user_id = "b1710dc5-73f5-469d-afd6-f4fdb2a9bfbc"
    user_record = db.get_user(dev_user_id)
    if not user_record:
        with db._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, google_id, email, display_name, role, created_at, last_login) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dev_user_id, "mock-dev-id-1234", "dev-revanth@example.com", "Beta Version", "user", datetime.now().isoformat(), datetime.now().isoformat())
            )
            conn.commit()
        user_record = db.get_user(dev_user_id)
    st.session_state.user = user_record

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# chat_alignment default layout used directly

if "messages" not in st.session_state:
    st.session_state.messages = []

if "answering_mode" not in st.session_state:
    st.session_state.answering_mode = "FAST"

if "settings_top_k" not in st.session_state:
    st.session_state.settings_top_k = 10

if "settings_depth" not in st.session_state:
    st.session_state.settings_depth = 1

if "last_query_metrics" not in st.session_state:
    st.session_state.last_query_metrics = None

if "selected_node_id" not in st.session_state:
    st.session_state.selected_node_id = None

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "regenerate_query" not in st.session_state:
    st.session_state.regenerate_query = None

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


# =============================================================================
# 14. LOGGED IN CONTEXT
# =============================================================================
user = st.session_state.user
user_id = user["user_id"]

# Fetch chats list
chats = db.list_chats(user_id)

# Set active chat if none active
if not st.session_state.active_chat_id and chats:
    st.session_state.active_chat_id = chats[0]["chat_id"]
elif st.session_state.active_chat_id and not any(c["chat_id"] == st.session_state.active_chat_id for c in chats):
    st.session_state.active_chat_id = chats[0]["chat_id"] if chats else None

active_chat_id = st.session_state.active_chat_id
active_chat = next((c for c in chats if c["chat_id"] == active_chat_id), None) if active_chat_id else None

# Sync chat_title in session state
if active_chat:
    if "chat_title" not in st.session_state or st.session_state.get("last_chat_id") != active_chat_id:
        st.session_state.chat_title = active_chat["title"]
        st.session_state.last_chat_id = active_chat_id
else:
    st.session_state.chat_title = None
    st.session_state.last_chat_id = None


# =============================================================================
# 15. LEFT SIDEBAR UI
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; padding: 0.6rem 0; margin-bottom: 0.75rem;">
        <img src="data:image/png;base64,{base64_logo}" style="width: 24px; height: 24px; border-radius: 4px; object-fit: contain;"/>
        <div style="font-size:0.95rem; font-weight:600; color:#ececec; letter-spacing: -0.01em;">Adaptive Workspace</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. New Chat Button (Proper creation & Auto-Cleanup)
    if st.button("＋  New chat", use_container_width=True, type="primary", key="btn_new_chat"):
        # Auto-delete active workspace if it has 0 messages (no questions asked)
        if active_chat_id:
            history = db.get_chat_history(active_chat_id)
            if not history:
                db.delete_chat(active_chat_id)
                chat_dir = f"doc_store/users/{user_id}/chats/{active_chat_id}"
                shutil.rmtree(chat_dir, ignore_errors=True)
                get_doc_system.clear()

        if active_chat:
            provider = active_chat["model_provider"]
            model = active_chat["model_name"]
        else:
            provider = "groq"
            model = "llama-3.3-70b-versatile"
        
        new_chat = db.create_chat(user_id, "New Chat", provider, model)
        st.session_state.active_chat_id = new_chat["chat_id"]
        st.session_state.chat_title = new_chat["title"]
        st.session_state.messages = []
        st.rerun()

    # 2. Documents Section
    st.markdown("<div style='margin-top: 1rem; margin-bottom: 0.4rem; font-size: 0.68rem; font-weight: 600; color: #666; letter-spacing: 0.06em; text-transform: uppercase;'>Documents</div>", unsafe_allow_html=True)
    if not active_chat_id:
        st.caption("No active workspace.")
        active_docs = []
    else:
        active_docs = db.list_documents(active_chat_id)
        if not active_docs:
            st.caption("No documents in this workspace.")
        else:
            for d in active_docs:
                page_cnt = d.get("page_count")
                page_str = f"{page_cnt} pages" if page_cnt else "ready"
                
                col_doc, col_del = st.columns([5, 1])
                with col_doc:
                    svg_doc_icon = """<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 6px; vertical-align: middle; color: #3b82f6;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>"""
                    st.markdown(f"""
                    <div style="font-size: 0.78rem; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 1px 0;" title="{d['filename']} ({page_str})">
                        {svg_doc_icon}{d['filename']}
                    </div>
                    """, unsafe_allow_html=True)
                with col_del:
                    if st.button("X", key=f"del_doc_sidebar_{d['document_id']}", use_container_width=True, help="Remove document"):
                        delete_doc(user_id, active_chat_id, d["document_id"], d["doc_name"])
                        st.rerun()

        # Upload PDFs button in sidebar
        if st.button("Upload PDFs", use_container_width=True, key="sidebar_upload_btn"):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()
        
        if st.session_state.show_uploader:
            uploaded_files = st.file_uploader(
                "Upload PDF documents", 
                type=["pdf"], 
                accept_multiple_files=True,
                key=f"sidebar_uploader_{active_chat_id}"
            )
            
            if uploaded_files:
                existing_filenames = {d["filename"] for d in active_docs}
                new_files = [f for f in uploaded_files if f.name not in existing_filenames]
                
                if new_files:
                    for f in new_files:
                        prog_bar = st.progress(0, text=f"Ingesting {f.name}...")
                        status_txt = st.empty()
                        try:
                            orig_env = inject_user_api_keys(user_id)
                            meta = ingest_pdf_with_progress(f, prog_bar, status_txt, user_id, active_chat_id)
                            restore_api_keys(orig_env)
                            
                            db.add_document(
                                chat_id=active_chat_id,
                                user_id=user_id,
                                filename=f.name,
                                doc_name=meta["doc_name"],
                                page_count=meta["page_count"],
                                chunk_count=meta["chunk_count"],
                                status="ready",
                                document_summary=meta.get("document_summary", ""),
                                summary_embedding=meta.get("summary_embedding"),
                                metadata=None
                            )
                        except Exception as e:
                            prog_bar.empty()
                            err_msg = str(e)
                            if "Failover error" in err_msg or "429" in err_msg or "Rate limit" in err_msg or "Too Many Requests" in err_msg:
                                status_txt.error("Ingestion failed: API rate limit exhausted. Check Settings.")
                            else:
                                status_txt.error(f"Ingestion failed: {e}")
                    
                    st.session_state.show_uploader = False
                    st.rerun()

    # 4. Chronological Chats switching (Bottom of sidebar) - Show all chats with inline Rename and Delete
    if chats:
        st.markdown("<div style='margin-top: 1.25rem; margin-bottom: 0.4rem; font-size: 0.68rem; font-weight: 600; color: #666; letter-spacing: 0.06em; text-transform: uppercase;'>Chats</div>", unsafe_allow_html=True)
        for c in chats:
            is_active = (c["chat_id"] == active_chat_id)
            btn_type = "primary" if is_active else "secondary"
            title = c["title"]
            if len(title) > 20:
                title = title[:18] + "..."
            
            rename_key = f"rename_active_{c['chat_id']}"
            if rename_key in st.session_state and st.session_state[rename_key]:
                col_input, col_save, col_cancel = st.columns([7, 1.5, 1.5])
                with col_input:
                    new_title_val = st.text_input(
                        "Rename", 
                        value=c["title"], 
                        key=f"chat_rename_input_{c['chat_id']}", 
                        label_visibility="collapsed"
                    )
                with col_save:
                    if st.button("💾", key=f"chat_save_btn_{c['chat_id']}", use_container_width=True, help="Save"):
                        if new_title_val.strip():
                            db.rename_chat(c["chat_id"], new_title_val.strip())
                            if is_active:
                                st.session_state.chat_title = new_title_val.strip()
                        st.session_state[rename_key] = False
                        st.rerun()
                with col_cancel:
                    if st.button("❌", key=f"chat_cancel_btn_{c['chat_id']}", use_container_width=True, help="Cancel"):
                        st.session_state[rename_key] = False
                        st.rerun()
            else:
                col_chat, col_menu = st.columns([8.2, 1.8])
                with col_chat:
                    if st.button(f"{title}", key=f"chat_nav_sidebar_{c['chat_id']}", use_container_width=True, type=btn_type):
                        # Auto-delete previous active workspace if switching away and it has 0 messages
                        if active_chat_id and active_chat_id != c["chat_id"]:
                            history = db.get_chat_history(active_chat_id)
                            if not history:
                                db.delete_chat(active_chat_id)
                                chat_dir = f"doc_store/users/{user_id}/chats/{active_chat_id}"
                                shutil.rmtree(chat_dir, ignore_errors=True)
                                get_doc_system.clear()

                        st.session_state.active_chat_id = c["chat_id"]
                        st.session_state.chat_title = c["title"]
                        st.session_state.messages = []
                        st.rerun()
                with col_menu:
                    with st.popover("…", use_container_width=True, help="Chat actions"):
                        if st.button("\u00a0\u00a0\u00a0\u00a0Rename", key=f"chat_ren_sidebar_{c['chat_id']}", use_container_width=True):
                            st.session_state[rename_key] = True
                            st.rerun()
                        if st.button("\u00a0\u00a0\u00a0\u00a0Delete", key=f"chat_del_sidebar_{c['chat_id']}", use_container_width=True):
                            # Delete from database
                            db.delete_chat(c["chat_id"])
                            
                            # Clean up local workspace files
                            chat_dir = f"doc_store/users/{user_id}/chats/{c['chat_id']}"
                            shutil.rmtree(chat_dir, ignore_errors=True)
                            get_doc_system.clear()
                            
                            # Handle redirect if the deleted chat was the active one
                            if c["chat_id"] == active_chat_id:
                                remaining_chats = db.list_chats(user_id)
                                st.session_state.active_chat_id = remaining_chats[0]["chat_id"] if remaining_chats else None
                                st.session_state.chat_title = remaining_chats[0]["title"] if remaining_chats else None
                            st.rerun()

    # 5. User Profile Footer & Logout
    st.markdown("<div style='border-top: 1px solid rgba(255,255,255,0.05); margin-top: 1rem;'></div>", unsafe_allow_html=True)
    prof_col1, prof_col2 = st.columns([1, 5])
    with prof_col1:
        initials = user['display_name'][0].upper() if user.get('display_name') else 'U'
        if user.get("profile_picture"):
            st.image(user["profile_picture"], width=28)
        else:
            st.markdown(f"<div style='width:28px; height:28px; border-radius:50%; background:#2f2f2f; border: 1px solid rgba(255,255,255,0.1); text-align:center; line-height:28px; color:#b4b4b4; font-weight:600; font-size: 0.72rem;'>{initials}</div>", unsafe_allow_html=True)
    with prof_col2:
        st.markdown(f"<div style='font-weight:500; color:#b4b4b4; font-size:0.78rem; line-height:28px;'>{user['display_name']}</div>", unsafe_allow_html=True)


# =============================================================================
# 16. MAIN AREA UI
# =============================================================================
# Load messages history for active chat
if active_chat_id:
    st.session_state.messages = db.get_chat_history(active_chat_id)
else:
    st.session_state.messages = []

# Top Bar Badges Rendering
if active_chat:
    docs_count = len(db.list_documents(active_chat_id))
    chat_title = st.session_state.get("chat_title", active_chat["title"])
    prov_upper = active_chat['model_provider'].upper()
    model_name = active_chat['model_name']
    st.markdown(f"""
    <div style="margin-bottom:0.75rem; padding-bottom:0.5rem;">
        <h2 style="margin:0; font-size:1.25rem; font-weight:600; color:var(--text-primary);">{chat_title}</h2>
        <div style="font-size:0.72rem; color:#666; margin-top:0.2rem;">{docs_count} docs · {model_name} · {prov_upper}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
        <img src="data:image/png;base64,{base64_logo}" style="width: 28px; height: 28px; object-fit: contain; opacity: 0.8;"/>
        <h2 style="margin:0; font-size:1.25rem; font-weight:600; color:var(--text-primary);">Welcome to Adaptive Workspace</h2>
    </div>
    """, unsafe_allow_html=True)

# Horizontal Tabs Layout
tab_chat, tab_settings = st.tabs(["Chat Workspace", "Settings"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Chat Workspace
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    if not active_chat_id:
        st.markdown(f"""
        <div class="hero-container" style="text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 1.5rem;">
            <img src="data:image/png;base64,{base64_logo}" style="width: 80px; height: 80px; object-fit: contain; margin-bottom: 1.5rem; filter: drop-shadow(0 4px 10px rgba(59, 130, 246, 0.25));"/>
            <div class="hero-title">Adaptive Workspace</div>
            <div class="hero-subtitle" style="margin-bottom: 2rem;">
                Select or create a workspace in the sidebar to start chatting with your PDF documents.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        _, btn_col, _ = st.columns([3, 2, 3])
        with btn_col:
            if st.button("➕ Create New Workspace", use_container_width=True, type="primary", key="empty_state_create_workspace"):
                provider = "groq"
                model = "llama-3.3-70b-versatile"
                new_chat = db.create_chat(user_id, "New Chat", provider, model)
                st.session_state.active_chat_id = new_chat["chat_id"]
                st.session_state.chat_title = new_chat["title"]
                st.session_state.messages = []
                st.rerun()
    else:
        # Show step-by-step onboarding flow if no messages are present
        if not st.session_state.messages:
            st.markdown(f"""
            <div class="hero-container" style="text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2.5rem 1.5rem;">
                <img src="data:image/png;base64,{base64_logo}" style="width: 70px; height: 70px; object-fit: contain; margin-bottom: 1.25rem; filter: drop-shadow(0 4px 10px rgba(59, 130, 246, 0.2));"/>
                <div class="hero-title">Welcome to Adaptive Workspace</div>
                <div class="hero-subtitle" style="margin-bottom: 2rem;">
                    Chat with your documents using advanced retrieval and your preferred AI models.
                </div>
                <div class="onboarding-steps" style="display: flex; gap: 1.5rem; justify-content: center; width: 100%; max-width: 800px; text-align: left;">
                    <div class="onboarding-step-card" style="flex: 1;">
                        <div class="onboarding-step-icon">1</div>
                        <div class="onboarding-step-title">Upload Documents</div>
                        <div class="onboarding-step-desc">Upload document PDFs in the uploader section below</div>
                    </div>
                    <div class="onboarding-step-card" style="flex: 1;">
                        <div class="onboarding-step-icon">2</div>
                        <div class="onboarding-step-title">Configure Mode</div>
                        <div class="onboarding-step-desc">Configure your models and keys in Settings if needed</div>
                    </div>
                    <div class="onboarding-step-card" style="flex: 1;">
                        <div class="onboarding-step-icon">3</div>
                        <div class="onboarding-step-title">Start Chatting</div>
                        <div class="onboarding-step-desc">Type a prompt to begin your retrieval chat</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Chat message alignment is handled by main CSS block

        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                
                # Show assistant dashboard (citations + latency logs)
                if msg["role"] == "assistant":
                    # Copy & Regenerate action bar
                    msg_content_escaped = msg["content"].replace("\\", "\\\\").replace("`", "\\`").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
                    copy_btn_id = f"copy_btn_{msg_idx}"
                    st.markdown(f"""
                    <div class="response-actions">
                        <button class="response-action-btn" id="{copy_btn_id}" onclick="
                            navigator.clipboard.writeText(`{msg_content_escaped}`).then(function() {{
                                var toast = document.createElement('div');
                                toast.className = 'copy-toast';
                                toast.textContent = 'Copied to clipboard';
                                document.body.appendChild(toast);
                                setTimeout(function() {{ toast.remove(); }}, 2200);
                            }});
                        ">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                            Copy
                        </button>
                    </div>
                    """, unsafe_allow_html=True)

                    # Regenerate button (Streamlit button - triggers rerun)
                    # Only show on the very last assistant message
                    if msg_idx == len(st.session_state.messages) - 1:
                        if st.button("Regenerate", key=f"regen_{msg_idx}", type="secondary"):
                            # Find the last user query
                            last_user_query = None
                            for prev_msg in reversed(st.session_state.messages):
                                if prev_msg["role"] == "user":
                                    last_user_query = prev_msg["content"]
                                    break
                            if last_user_query:
                                # Delete the last assistant message from DB
                                msg_id = msg.get("message_id")
                                if msg_id:
                                    db.delete_message(msg_id)
                                st.session_state.regenerate_query = last_user_query
                                st.rerun()

                    # 1. Sources Used (collapsed cards grouped by filename)
                    if msg.get("sources"):
                        sources_by_file = {}
                        for src in msg["sources"]:
                            fname = src.get("filename", "Unknown")
                            if fname not in sources_by_file:
                                sources_by_file[fname] = []
                            sources_by_file[fname].append(src)
                        
                        with st.expander(f"Sources Used ({len(msg['sources'])} chunks)", expanded=False):
                            for fname, chunks in sources_by_file.items():
                                st.markdown(f"**{fname}**")
                                for ch in chunks:
                                    page_val = ch.get("page", 0)
                                    score_val = ch.get("score", 0.0)
                                    text_val = ch.get("text", "")
                                    st.markdown(f"- **Page {page_val}** (Score: {score_val:.2f})")
                                    st.caption(text_val)

                    # 2. Technical Details Expander (collapsed by default)
                    with st.expander("Technical Details", expanded=False):
                        # Trust and Claim metrics hidden to simplify UX
                                    
                        if msg.get("trace"):
                            t = msg["trace"]
                            st.write(f"**Query Intent / Type**: {t.get('query_type')}")
                            st.write(f"**Traversal Strategy**: {t.get('traversal_used')}")
                            st.write(f"**Graph Expansion Depth**: {t.get('graph_depth')}")
                            
                            if len(t.get("sub_questions", [])) > 1:
                                st.write("**Decomposed Sub-Questions:**")
                                for sq in t.get("sub_questions", []):
                                    st.markdown(f"- *{sq}*")
                                    
                            perf = t.get("performance")
                            if perf:
                                st.markdown("**Component execution latency breakdown (ms):**")
                                for k, name in [
                                    ("classification_ms", "Intent Classification"),
                                    ("decomposition_ms", "Query Decomposition"),
                                    ("hyde_ms", "HyDE Generation"),
                                    ("retrieval_ms", "Fusion Retrieval"),
                                    ("graph_expansion_ms", "KG Traversal"),
                                    ("rerank_ms", "Reranker"),
                                    ("generation_ms", "LLM Answer Generation")
                                ]:
                                    dur = perf.get(k, 0.0)
                                    if dur > 0.0:
                                        st.markdown(f"- *{name}*: {dur:.1f} ms")

        # Chat Input Container
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        
        # Display config metrics bar + Attach PDFs on the same line
        if active_chat:
            doc_count = len(active_docs)
            prov_name = active_chat["model_provider"].upper()
            model_name = active_chat["model_name"]
            st.markdown(f"""
            <div class="input-indicator-bar">
                <span><strong>{doc_count}</strong> documents active</span>
                <span>&middot;</span>
                <span>model: <strong>{model_name}</strong> ({prov_name})</span>
            </div>
            """, unsafe_allow_html=True)

        # Chat message trigger
        active_docs = db.list_documents(active_chat_id)
        input_disabled = (len(active_docs) == 0)
        
        prompt_placeholder = "Ask anything about your documents..." if not input_disabled else "Please upload at least one PDF document to enable chat."
        
        # Handle regeneration trigger
        regen_query = st.session_state.get("regenerate_query")
        if regen_query:
            st.session_state.regenerate_query = None
            prompt = regen_query
            # Don't re-save the user message, it already exists
        elif prompt := st.chat_input(prompt_placeholder, disabled=input_disabled, key="chat_query"):
            # Append & Save User Message
            db.save_message(active_chat_id, "user", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Auto-rename if the chat is currently titled "New Chat" (case-insensitive) and this is the first message
            is_new_chat = (active_chat and active_chat["title"].lower() == "new chat") or (st.session_state.get("chat_title") and st.session_state.get("chat_title").lower() == "new chat")
            if is_new_chat and len(st.session_state.messages) == 1:
                # Generate a short clean title using LLM or fallback
                base_title = generate_workspace_title(user_id, prompt, active_chat)
                new_title = resolve_unique_title(user_id, base_title)
                
                db.rename_chat(active_chat_id, new_title)
                st.session_state.chat_title = new_title
        else:
            prompt = None

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Show thinking indicator
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="thinking-indicator">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
                Thinking...
            </div>
            """, unsafe_allow_html=True)
            
            # Stop button (displayed during generation)
            stop_col1, stop_col2, stop_col3 = st.columns([4, 2, 4])
            with stop_col2:
                if st.button("Stop generating", key="stop_gen_btn", use_container_width=True, type="secondary"):
                    st.session_state.stop_requested = True

            try:
                # 1. Inject API keys for the run
                orig_env = inject_user_api_keys(user_id)
                
                # 2. Get active doc names
                doc_names = [d["doc_name"] for d in active_docs]
                
                # settings traversal parameters
                settings_top_k = st.session_state.get("settings_top_k", 10)
                settings_depth = st.session_state.get("settings_depth", 1)
                
                # Check for stop request
                if st.session_state.stop_requested:
                    st.session_state.stop_requested = False
                    thinking_placeholder.empty()
                    restore_api_keys(orig_env)
                    st.info("Generation stopped.")
                    st.rerun()
                
                # 3. Query Execution
                result = query_all_docs(
                    query=prompt,
                    user_id=user_id,
                    chat_id=active_chat_id,
                    doc_names=doc_names,
                    top_k=settings_top_k,
                    graph_depth=settings_depth,
                    chat_history=st.session_state.messages,
                    mode=st.session_state.answering_mode,
                    active_provider=active_chat["model_provider"],
                    active_model=active_chat["model_name"]
                )
                
                # 4. Restore API keys
                restore_api_keys(orig_env)
            except Exception as e:
                thinking_placeholder.empty()
                try:
                    restore_api_keys(orig_env)
                except Exception:
                    pass
                err_msg = str(e)
                if "Failover error" in err_msg or "429" in err_msg or "Rate limit" in err_msg or "Too Many Requests" in err_msg:
                    st.error("""
                    ### API Tokens / Rate Limit Exhausted
                    
                    It looks like all available API keys or provider connections failed to generate a response. This usually happens when:
                    
                    1. **Rate Limits (429 Too Many Requests)**: You have hit the request/token rate limits for your active provider (e.g., Groq).
                    2. **Exhausted Credits/Token Balance**: Your provider account (e.g., OpenAI, Claude) has run out of credits.
                    3. **Invalid or Unconfigured API Keys**: Custom API keys are missing, inactive, or invalid.
                    
                    **Action Steps:**
                    - Switch to the **Settings & Analytics** tab to check and register active API keys.
                    - Select a different LLM Provider (e.g., Gemini or OpenAI) in the sidebar.
                    - Check the **Provider API Status Monitor** at the bottom of the Settings page.
                    - Verify your token limits and billing status directly on your provider's developer console.
                    """)
                else:
                    st.error(f"Workspace generation failed: {e}")
                st.stop()

            # Clear thinking indicator
            thinking_placeholder.empty()

            with st.chat_message("assistant"):
                try:
                    # 5. Typewriter streaming effect
                    answer_text = result["answer"]
                    response_placeholder = st.empty()
                    
                    # Stream words with a subtle typewriter effect
                    words = answer_text.split(" ")
                    displayed = ""
                    chunk_size = 3  # Stream 3 words at a time for smooth effect
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        displayed += (" " if displayed else "") + chunk
                        response_placeholder.markdown(displayed + " |", unsafe_allow_html=True)
                        time.sleep(0.03)
                    
                    # Final render without cursor
                    response_placeholder.markdown(answer_text, unsafe_allow_html=True)
                    
                    # 6. Save Answer
                    db.save_message(
                        chat_id=active_chat_id,
                        role="assistant",
                        content=result["answer"],
                        sources=result["sources"],
                        confidence=result["confidence"],
                        confidence_emoji=result["confidence_emoji"],
                        grounding_score=result["grounding_score"],
                        trust_level=result["trust_level"],
                        hallucination_risk=result["hallucination_risk"],
                        claims=result["claims"],
                        trace=result["trace"],
                        performance=result["performance"]
                    )
                    
                    st.session_state.last_query_metrics = result["performance"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Workspace generation failed: {e}")


with tab_settings:
    st.markdown("### Settings & Configuration")
    
    # 1. Model Configuration
    st.markdown("#### Model Configuration")
    if active_chat:
        providers_list = ["groq", "openai", "gemini", "claude"]
        active_prov = active_chat["model_provider"]
        if active_prov not in providers_list:
            providers_list.append(active_prov)
        prov_idx = providers_list.index(active_prov)
        
        selected_prov = st.selectbox("LLM Provider", providers_list, index=prov_idx, key="settings_provider_select")
        
        from providers import PROVIDER_MODEL_FALLBACKS
        models_list = PROVIDER_MODEL_FALLBACKS.get(selected_prov, []).copy()
        active_model = active_chat["model_name"]
        if active_model not in models_list:
            models_list.append(active_model)
        
        model_idx = models_list.index(active_model) if active_model in models_list else 0
        selected_model = st.selectbox("Active Model", models_list, index=model_idx, key="settings_model_select")
        
        if selected_prov != active_prov or selected_model != active_model:
            db.update_chat_model(active_chat_id, selected_prov, selected_model)
            st.success(f"Model updated to {selected_model} ({selected_prov.upper()})")
            st.rerun()
    else:
        st.info("No active chat to configure.")

    st.divider()

    # 3. API Key Management
    with st.expander("API Key Management"):
        keys = db.list_api_keys(user_id)
        if keys:
            st.markdown("##### Registered API Keys")
            for k in keys:
                col_k_details, col_k_status, col_k_del = st.columns([4, 2, 1])
                with col_k_details:
                    st.markdown(f"**{k['provider'].upper()}** ({k['nickname']}) · `{k['display_key']}`")
                with col_k_status:
                    is_active = st.checkbox("Active", value=(k["is_active"] == 1), key=f"key_active_{k['key_id']}")
                    if is_active != (k["is_active"] == 1):
                        db.toggle_api_key(k["key_id"], is_active)
                        st.rerun()
                with col_k_del:
                    if st.button("Delete", key=f"key_del_{k['key_id']}", use_container_width=True):
                        db.delete_api_key(k["key_id"])
                        st.rerun()
        else:
            st.caption("No custom API keys registered. Falling back to environment variables.")
            
        st.markdown("##### Register New API Key")
        add_prov = st.selectbox("Key Provider", ["groq", "openai", "gemini", "claude"], key="settings_add_prov_select")
        add_nick = st.text_input("Nickname (e.g. Work Groq)", key="settings_add_nick_input")
        add_val = st.text_input("API Key Value", type="password", key="settings_add_val_input")
        
        test_col, save_col = st.columns(2)
        with test_col:
            if st.button("Test Connection", key="btn_test_conn_action"):
                if not add_val:
                    st.error("Please enter a key value.")
                else:
                    with st.spinner("Testing API connection..."):
                        try:
                            if add_prov == "groq":
                                from providers import GroqProvider
                                p = GroqProvider("groq", add_val)
                                res = p.generate("Hello", max_tokens=5)
                            elif add_prov == "openai":
                                from providers import OpenAIProvider
                                p = OpenAIProvider("openai", add_val)
                                res = p.generate("Hello", max_tokens=5)
                            elif add_prov == "gemini":
                                from providers import GeminiProvider
                                p = GeminiProvider("gemini", add_val)
                                res = p.generate("Hello", max_tokens=5)
                            elif add_prov == "claude":
                                from providers import ClaudeProvider
                                p = ClaudeProvider("claude", add_val)
                                res = p.generate("Hello", max_tokens=5)
                            st.success("Connection successful!")
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
        with save_col:
            if st.button("Save Key", type="primary", key="btn_save_key_action"):
                if not add_nick or not add_val:
                    st.error("Nickname and Key Value are required.")
                else:
                    db.add_api_key(user_id, add_prov, add_val, add_nick)
                    st.success("API Key successfully registered!")
                    st.rerun()

    # 4. Search & Retrieval Settings
    with st.expander("Search & Retrieval Parameters"):
        top_k_val = st.slider("Results per document", 3, 20, value=st.session_state.settings_top_k, key="settings_slider_top_k")
        depth_val = st.slider("Graph search depth", 0, 3, value=st.session_state.settings_depth, key="settings_slider_depth")
        st.session_state.settings_top_k = top_k_val
        st.session_state.settings_depth = depth_val

    # 5. Workspace Controls
    with st.expander("Workspace Controls"):
        if active_chat:
            st.markdown("##### Rename Workspace")
            new_title = st.text_input("Workspace Title", value=st.session_state.chat_title, key="settings_rename_input_field")
            if new_title != st.session_state.chat_title and new_title.strip():
                db.rename_chat(active_chat_id, new_title.strip())
                st.session_state.chat_title = new_title.strip()
                st.rerun()
                
            st.markdown("##### Danger Zone")
            if st.button("Delete Workspace", type="primary", use_container_width=True, key="settings_del_workspace_action"):
                db.delete_chat(active_chat_id)
                chat_dir = f"doc_store/users/{user_id}/chats/{active_chat_id}"
                shutil.rmtree(chat_dir, ignore_errors=True)
                get_doc_system.clear()
                chats = db.list_chats(user_id)
                st.session_state.active_chat_id = chats[0]["chat_id"] if chats else None
                st.rerun()
        else:
            st.info("No active workspace.")

    # 6. Knowledge Graph Explorer (Developer Tools)
    with st.expander("Developer Tools: Knowledge Graph Explorer"):
        if not active_chat_id:
            st.info("No active chat session. Create a chat to inspect its knowledge graphs.")
        else:
            active_docs = db.list_documents(active_chat_id)
            if not active_docs:
                st.info("No documents uploaded to this chat. Ingest a document first to see graph statistics.")
            else:
                with st.spinner("Loading combined knowledge graphs..."):
                    builder = get_combined_graph(user_id, active_chat_id, active_docs)
                    graph = builder.graph
                    num_nodes = graph.number_of_nodes()
                    num_edges = graph.number_of_edges()
                    
                    if num_nodes > 0:
                        import networkx as nx
                        avg_degree = sum(dict(graph.degree()).values()) / num_nodes
                        density = nx.density(graph)
                        components = nx.number_connected_components(graph)
                    else:
                        avg_degree = 0.0
                        density = 0.0
                        components = 0
                        
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-val">{num_nodes}</div>
                    <div class="kpi-label">Total Nodes</div>
                </div>
                """, unsafe_allow_html=True)
                stat_col2.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-val">{num_edges}</div>
                    <div class="kpi-label">Total Edges</div>
                </div>
                """, unsafe_allow_html=True)
                stat_col3.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-val">{avg_degree:.2f}</div>
                    <div class="kpi-label">Average Node Degree</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Semantic Node Inspector
                st.markdown("##### Semantic Node Inspector")
                if num_nodes == 0:
                    st.caption("No nodes present in graph database to inspect.")
                else:
                    nodes_list = list(graph.nodes(data=True))
                    options = {}
                    for node_id, data in nodes_list:
                        text_preview = data.get("text", "")[:40].replace("\n", " ")
                        options[node_id] = f"{node_id[:8]}... - {data.get('source', '')} (Page {data.get('page', '')}) - {text_preview}"
                    
                    # Check session state selected node
                    if st.session_state.selected_node_id not in options:
                        st.session_state.selected_node_id = list(options.keys())[0]
                       
                    selected_key = st.selectbox(
                        "Select chunk node to inspect details",
                        options=list(options.keys()),
                        format_func=lambda x: options[x],
                        index=list(options.keys()).index(st.session_state.selected_node_id),
                        key="settings_inspect_node_select"
                    )
                    st.session_state.selected_node_id = selected_key
                    
                    node_data = graph.nodes[selected_key]
                    st.markdown("**Chunk Content Preview**")
                    st.text_area("Chunk Content", value=node_data.get("text", ""), height=150, disabled=True, label_visibility="collapsed", key="settings_chunk_preview_textarea")
                    
                    det_col1, det_col2, det_col3 = st.columns(3)
                    det_col1.metric("Source PDF File", node_data.get("source", "Unknown"))
                    det_col2.metric("Page Number", node_data.get("page", "Unknown"))
                    det_col3.metric("Semantic Type", node_data.get("chunk_type", "text"))
                    
                    ent_col1, ent_col2 = st.columns(2)
                    with ent_col1:
                        st.markdown("**Extracted Entities**")
                        ents = node_data.get("entities", [])
                        if ents:
                            st.write(", ".join(ents))
                        else:
                            st.caption("No entities extracted from this node.")
                    with ent_col2:
                        st.markdown("**Keywords**")
                        kws = node_data.get("keywords", [])
                        if kws:
                            st.write(", ".join(kws))
                        else:
                            st.caption("No keywords extracted.")
                    
                    # Neighbors inspector
                    st.markdown("##### Neighboring Linked Chunks")
                    neighbors = list(graph.neighbors(selected_key))
                    if not neighbors:
                        st.caption("No semantic edges found for this chunk node.")
                    else:
                        for idx, nbr in enumerate(neighbors):
                            edge = graph[selected_key][nbr]
                            weight = edge.get("weight", 0.0)
                            types = edge.get("types", [])
                            types_str = ", ".join(types)
                            
                            nbr_text = graph.nodes[nbr].get("text", "")[:60].replace("\n", " ")
                            btn_label = f"Jump to Neighbor Node ({weight:.2f} sim | Types: {types_str}): {nbr[:8]}... - {nbr_text}..."
                            
                            if st.button(btn_label, key=f"settings_inspect_jump_btn_{idx}_{nbr}", use_container_width=True):
                                st.session_state.selected_node_id = nbr
                                st.rerun()

    # 7. Workspace Analytics & Observability
    with st.expander("Observability & Usage Metrics"):
        total_docs_count = len(db.list_documents(active_chat_id)) if active_chat_id else 0
        total_queries = 0
        all_query_latencies = []
        if active_chat_id:
            hist = db.get_chat_history(active_chat_id)
            for m in hist:
                if m["role"] == "user":
                    total_queries += 1
                elif m["role"] == "assistant":
                    p = m.get("performance")
                    if p and isinstance(p, dict) and "total_ms" in p:
                        all_query_latencies.append(p["total_ms"])
        avg_latency = np.mean(all_query_latencies) if all_query_latencies else 0.0
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{total_docs_count}</div>
            <div class="kpi-label">Workspace Documents</div>
        </div>
        """, unsafe_allow_html=True)
        stat_col2.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{total_queries}</div>
            <div class="kpi-label">Workspace Queries</div>
        </div>
        """, unsafe_allow_html=True)
        stat_col3.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{avg_latency / 1000.0:.2f}s</div>
            <div class="kpi-label">Avg Response Time</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Provider status monitor
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Provider API Status Monitor")
        health_stats = get_provider_health_stats()
        h_cols = st.columns(4)
        for idx, hs in enumerate(health_stats):
            with h_cols[idx]:
                if hs["failures"] == 0 and hs["successes"] > 0:
                    status_text = "<span style='color: #10b981; font-weight: 600;'>Active</span>"
                elif hs["success_rate"] == 0.0 and hs["failures"] > 0:
                    status_text = "<span style='color: #ef4444; font-weight: 600;'>Offline</span>"
                elif hs["failures"] > 0:
                    status_text = "<span style='color: #f59e0b; font-weight: 600;'>Degraded</span>"
                else:
                    status_text = "<span style='color: #6b7280; font-weight: 500;'>Inactive</span>"
                    
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.04); padding: 10px; border-radius: 8px; font-size:0.8rem; display:flex; flex-direction:column;">
                    <div><strong style="font-size:0.85rem;">{hs['provider'].upper()}</strong></div>
                    <div style="margin-top:4px; font-size:0.75rem; color:var(--text-secondary);">Status: {status_text}</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary);">Success Rate: {hs['success_rate']}%</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary);">Avg Latency: {hs['avg_latency']:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
