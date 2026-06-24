"""DataSmith — Synthetic Dataset Generator."""
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from datasmith.ui import icons

# ── CSS injection ────────────────────────────────────────────────────────


@st.cache_resource
def _load_css() -> str:
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ds-accent: #2563eb;
    --ds-accent-dim: #1d4ed8;
    --ds-bg: #ffffff;
    --ds-surface: #f8fafc;
    --ds-border: #e2e8f0;
    --ds-text: #0f172a;
    --ds-text-muted: #64748b;
    --ds-radius: 8px;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
.stAppToolbar {display: none;}

/* Global font */
.stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* Component overrides */
.stButton button {
    border-radius: var(--ds-radius) !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.stButton button[kind="primary"] {
    background: var(--ds-accent) !important;
    border-color: var(--ds-accent) !important;
}
.stButton button[kind="primary"]:hover {
    background: var(--ds-accent-dim) !important;
    border-color: var(--ds-accent-dim) !important;
}

.stTextInput input {
    border-radius: var(--ds-radius) !important;
}
.stTextInput input:focus {
    border-color: var(--ds-accent) !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
}

.stSelectbox [data-baseweb="select"] {
    border-radius: var(--ds-radius) !important;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 500 !important;
}

.stExpander {
    border-radius: var(--ds-radius) !important;
    border: 1px solid var(--ds-border) !important;
}

[data-testid="stMetric"] {
    background: var(--ds-surface);
    border: 1px solid var(--ds-border);
    border-radius: var(--ds-radius);
    padding: 1rem;
}

.stDownloadButton button {
    border-radius: var(--ds-radius) !important;
}

/* Section spacing */
section + section { margin-top: 1.5rem; }

/* Responsive */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .stButton button { font-size: 0.85rem; }
    [data-testid="stMetric"] { padding: 0.75rem; }
}

/* Focus visible for keyboard nav */
:focus-visible {
    outline: 2px solid var(--ds-accent);
    outline-offset: 2px;
}
</style>"""


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataSmith",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%232563eb' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z'/></svg>",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(_load_css(), unsafe_allow_html=True)

# ── Runtime setup ────────────────────────────────────────────────────────

_DATA_DIR = Path(os.getenv("DATASMITH_DATA", str(Path(tempfile.gettempdir()) / "datasmith")))
_SEED_DB = Path(__file__).parent / "data" / "datasmith.db"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "datasmith.db"

if not _DB_PATH.exists() and _SEED_DB.exists():
    shutil.copy2(_SEED_DB, _DB_PATH)

os.environ["DATASMITH_DB_PATH"] = str(_DB_PATH)

# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<h3 style="text-align:center;margin-bottom:0;">{icons.BRAND} DataSmith</h3>'
        '<p style="text-align:center;color:var(--ds-text-muted);font-size:0.85rem;margin-top:0.25rem;">'
        "Synthetic Dataset Generator</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/01_Generate.py", label="Generate", icon="⚡")
    st.page_link("pages/02_About.py", label="About", icon="ℹ️")
    st.divider()
    st.caption("DataSmith v0.3.1 — AGPL v3")

# ── Main content ─────────────────────────────────────────────────────────

st.markdown(
    f"<h1 style='text-align:center;'>{icons.BRAND} DataSmith</h1>"
    "<p style='text-align:center;color:var(--ds-text-muted);font-size:1.05rem;'>"
    "Realistic synthetic data for dev, testing, and demos. "
    "No training. No GPU. No cloud calls.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center;'>{icons.DATABASE}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Schema**", help="Domain-tuned column schemas from our Knowledge Graph")
        st.caption("Domain-tuned schemas from the Knowledge Graph")
with col2:
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center;'>{icons.WAVES}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Imperfections**", help="Realistic nulls, outliers, and noise patterns")
        st.caption("Realistic nulls, outliers, and noise")
with col3:
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center;'>{icons.DOWNLOAD}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Export**", help="Download as CSV or JSON, ready to use")
        st.caption("CSV or JSON, ready to use")

st.divider()

# ── Footer ───────────────────────────────────────────────────────────────

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    st.markdown(
        '<div style="text-align:center;">'
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
        '<p style="color:var(--ds-text-muted);font-size:0.8em;margin-top:8px;">'
        "If DataSmith helps your project, consider supporting</p></div>",
        unsafe_allow_html=True,
    )
