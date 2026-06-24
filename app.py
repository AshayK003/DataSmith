"""DataSmith — Synthetic Dataset Generator.

Phase 1 MVP: Domain-based generation with imperfection profiles.
"""

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

_SVGS = {
    "brand": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'
        "</svg>"
    ),
    "home": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<polyline points="9 22 9 12 15 12 15 22"/>'
        "</svg>"
    ),
    "info": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M12 16v-4"/>'
        '<path d="M12 8h.01"/>'
        "</svg>"
    ),
    "database": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
        "</svg>"
    ),
    "waves": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
        "</svg>"
    ),
    "download": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>'
        "</svg>"
    ),
}

st.set_page_config(
    page_title="DataSmith",
    page_icon="⚒️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Runtime setup — ensure writable DB ─────────────────────────────────
_DATA_DIR = Path(os.getenv("DATASMITH_DATA", str(Path(tempfile.gettempdir()) / "datasmith")))
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
        f'<h2 style="text-align: center; margin-bottom: 0;">{_SVGS["brand"]} DataSmith</h2>'
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
    f"<h1 style='text-align: center;'>{_SVGS['brand']} DataSmith</h1>"
    "<p style='text-align: center; color: #888; font-size: 1.1em;'>"
    "Realistic synthetic data for dev, testing, and demos. "
    "No training. No GPU. No cloud calls.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"<div style='text-align: center; padding: 1em;'>"
        f"<h3>{_SVGS['database']} Schema</h3>"
        "<p style='color: #888; font-size: 0.9em;'>"
        "Domain-tuned column schemas from our Knowledge Graph</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div style='text-align: center; padding: 1em;'>"
        f"<h3>{_SVGS['waves']} Imperfections</h3>"
        "<p style='color: #888; font-size: 0.9em;'>"
        "Realistic nulls, outliers, and noise patterns</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div style='text-align: center; padding: 1em;'>"
        f"<h3>{_SVGS['download']} Export</h3>"
        "<p style='color: #888; font-size: 0.9em;'>"
        "Download as CSV or JSON, ready to use</p></div>",
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("### Quick Start")
st.markdown(
    "1. Go to **Generate** in the sidebar\n"
    "2. Select a domain (e-commerce, healthcare, etc.)\n"
    "3. Tweak columns, types, and ranges\n"
    "4. Choose row count and generate\n"
    "5. Download your dataset"
)

st.divider()

# Centered footer with Chai4Me badge
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    st.markdown(
        '<div style="text-align: center;">'
        '<a href="https://chai4.me/ashaykushwaha003" target="_blank" '
        'title="Support ashaykushwaha003 on Chai4Me" '
        'style="display:inline-flex;flex-direction:column;align-items:center;'
        'justify-content:center;background:#ffffff;padding:8px 32px;'
        'border-radius:16px;text-decoration:none;'
        'border:1px solid #e5e7eb;'
        'box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), '
        '0 2px 4px -2px rgba(0,0,0,0.05);transition:transform 0.2s;">'
        '<img src="https://chai4.me/icons/wordmark.png" alt="Chai4Me" '
        'style="height:32px;object-fit:contain;"/></a>'
        '<p style="color: #888; font-size: 0.8em; margin-top: 8px;">'
        "If DataSmith helps your project, consider supporting</p></div>",
        unsafe_allow_html=True,
    )
