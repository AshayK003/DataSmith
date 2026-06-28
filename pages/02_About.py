"""About Page -- project info and credits."""
import streamlit as st

from datasmith.ui import icons
from datasmith.ui.components import render_header

st.set_page_config(page_title="About — DataSmith", layout="centered")

render_header("about")

st.markdown(f"<h1 style='text-align:center;'>{icons.INFO} About DataSmith</h1>",
            unsafe_allow_html=True)

st.markdown(
    "DataSmith generates realistic synthetic data by learning from real-world "
    "dataset schemas. Its **Schema Knowledge Graph** (crawled from Kaggle, UCI, "
    "and HuggingFace) and **Domain Imperfection Fingerprints** (statistical "
    "patterns extracted from real data) produce datasets that look, feel, "
    "and break like the real thing."
)

st.divider()

st.markdown("### Key Features")
st.markdown(
    "- **Domain-tuned schemas** -- 10 domains sourced from Kaggle, UCI, and HuggingFace\n"
    "- **Realistic imperfections** -- nulls, outliers, and noise patterns extracted from real datasets\n"
    "- **Customizable** -- edit column names, types, and parameters before generation\n"
    "- **Lightweight** -- pure Python, no GPU, no cloud, no training time\n"
    "- **Open Source** -- AGPL v3 licensed"
)

st.divider()

st.markdown("### Stack")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Backend**\n- Python 3.11+\n- NumPy/SciPy\n- Pandas\n- SQLite (FTS5)")
with col2:
    st.markdown("**Data Sources**\n- KaggleHub\n- HuggingFace\n- UCI Archive\n- Frictionless")
with col3:
    st.markdown("**Delivery**\n- CSV / JSON\n- Streamlit Cloud\n- AGPL v3")

st.divider()

st.markdown(
    "<p style='text-align: center; color: var(--ds-text-muted);'>"
    "DataSmith v0.4.0 -- Built by Ashay Kushwaha<br>"
    "License: AGPL v3 -- <a href='https://github.com/AshayK003/DataSmith'>GitHub</a></p>",
    unsafe_allow_html=True,
)
