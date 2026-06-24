"""DataSmith — Synthetic Dataset Generator.

Phase 1 MVP: Domain-based generation with imperfection profiles.
"""

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DataSmith",
    page_icon="⚒️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Runtime setup — ensure writable DB ─────────────────────────────────
_DATA_DIR = Path(os.getenv("DATASMITH_DATA", str(tempfile.gettempdir() / "datasmith")))
_SEED_DB = Path(__file__).parent / "data" / "datasmith.db"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "datasmith.db"

# If a seed DB exists in the repo, copy it to the writable location
if not _DB_PATH.exists() and _SEED_DB.exists():
    shutil.copy2(_SEED_DB, _DB_PATH)

# Set env var so engine.py picks it up
os.environ["DATASMITH_DB_PATH"] = str(_DB_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<h2 style="text-align: center; margin-bottom: 0;">⚒️ DataSmith</h2>'
        '<p style="text-align: center; color: #888; font-size: 0.9em;">'
        "Synthetic Dataset Generator</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/01_Generate.py", label="Generate", icon="⚒️")
    st.page_link("pages/02_About.py", label="About", icon="ℹ️")
    st.divider()
    st.caption("DataSmith v0.2.0 — AGPL v3")

# ── Main content ──────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align: center;'>⚒️ DataSmith</h1>"
    "<p style='text-align: center; color: #888; font-size: 1.1em;'>"
    "Generate realistic synthetic datasets for development, testing, and demos.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        "<div style='text-align: center; padding: 1em;'>"
        "<h3>📊 Schema</h3>"
        "<p style='color: #888; font-size: 0.9em;'>"
        "Domain-tuned column schemas from our Knowledge Graph</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        "<div style='text-align: center; padding: 1em;'>"
        "<h3>🌊 Imperfections</h3>"
        "<p style='color: #888; font-size: 0.9em;'>"
        "Realistic nulls, outliers, and noise patterns</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        "<div style='text-align: center; padding: 1em;'>"
        "<h3>📥 Export</h3>"
        "<p style='color: #888; font-size: 0.9em;'>"
        "Download as CSV or JSON, ready to use</p></div>",
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("### Quick Start")
st.markdown(
    "1. Go to **Generate** in the sidebar\n"
    "2. Select a domain (e-commerce, healthcare, etc.)\n"
    "3. Edit the schema to your needs\n"
    "4. Choose row count and generate\n"
    "5. Download your dataset"
)
