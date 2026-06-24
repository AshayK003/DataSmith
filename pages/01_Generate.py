"""Generate Page — the core of DataSmith.

Natural language → schema → editor → generate → preview → download.
"""

import io
import os
from pathlib import Path

import streamlit as st

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.llm.client import is_available as llm_available
from datasmith.llm.discovery import discover_schema
from datasmith.schema.crawler import SEED_DOMAINS
from datasmith.generation.engine import generate_dataset, schema_from_kg, _generic_schema

st.set_page_config(page_title="Generate — DataSmith", layout="centered")

st.markdown("<h1 style='text-align: center;'>⚒️ Generate Dataset</h1>",
            unsafe_allow_html=True)

# ── Init KG ───────────────────────────────────────────────────────────────

@st.cache_resource
def _get_kg():
    db_path = os.environ.get("DATASMITH_DB_PATH", "data/datasmith.db")
    db = Database(db_path)
    return KnowledgeGraph(db)

kg = _get_kg()

# ── Track active domain name ──────────────────────────────────────────────

if "active_domain" not in st.session_state:
    st.session_state["active_domain"] = "custom"

# ── Schema Discovery ──────────────────────────────────────────────────────

st.markdown("### Describe Your Dataset")
st.caption(
    "Tell DataSmith what kind of data you need in plain English. "
    "Or pick a domain below."
)

# Two tabs: NL input or domain selector
discover_tab, browse_tab = st.tabs(["🗣️ Describe", "📂 Browse Domains"])

with discover_tab:
    nl_input = st.text_input(
        "Dataset description",
        placeholder="e.g., e-commerce transactions with customer demographics and order history",
        label_visibility="collapsed",
    )
    use_nl = st.button("🔍 Discover Schema", type="primary", use_container_width=True)

with browse_tab:
    domain_name = st.selectbox(
        "Domain",
        list(SEED_DOMAINS.keys()),
        format_func=lambda d: d.replace("-", " ").title(),
        label_visibility="collapsed",
    )
    use_domain = st.button("Load Domain", use_container_width=True)

# ── Resolve schema ────────────────────────────────────────────────────────

resolved_schema = None
schema_source = None

if use_nl and nl_input:
    if not llm_available():
        st.warning(
            "No LLM API key configured. "
            "Set `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `LLM_API_KEY` in "
            "Streamlit Cloud secrets for NL discovery. "
            "Using domain browser instead."
        )
    else:
        with st.spinner("Analyzing your description..."):
            resolved_schema = discover_schema(kg, nl_input)
            if resolved_schema:
                schema_source = f"Description: _{nl_input}_"
                st.session_state["active_domain"] = nl_input[:40]
            else:
                st.info(
                    "Could not discover a schema from that description. "
                    "Try being more specific or pick a domain below."
                )

elif use_domain:
    with st.spinner("Loading domain..."):
        resolved_schema = schema_from_kg(kg, domain_name)
        if not resolved_schema:
            resolved_schema = _generic_schema(domain_name)
        schema_source = f"Domain: **{domain_name.replace('-', ' ').title()}**"
        st.session_state["active_domain"] = domain_name

# ── Schema editor ─────────────────────────────────────────────────────────

if resolved_schema:
    st.markdown("### Schema Editor")
    if schema_source:
        st.caption(schema_source)
    st.caption("Edit column names, types, and parameters. Add or remove rows.")

    # Convert schema to editor-friendly format
    editor_data = []
    for col in resolved_schema:
        dtype = col.get("data_type", "numeric").lower()
        if dtype in ("numeric", "integer"):
            display_type = "numeric"
        elif dtype == "boolean":
            display_type = "boolean"
        elif dtype == "datetime":
            display_type = "datetime"
        else:
            display_type = "text"

        editor_data.append({
            "column_name": col.get("column_name", "col"),
            "data_type": display_type,
            "mean": col.get("mean", 50.0),
            "std": col.get("std", 20.0),
            "min": col.get("min", 0.0) if display_type == "numeric" else 0.0,
            "max": col.get("max", 100.0) if display_type == "numeric" else 100.0,
        })

    edited = st.data_editor(
        editor_data,
        column_config={
            "column_name": st.column_config.TextColumn("Column Name", width="medium"),
            "data_type": st.column_config.SelectboxColumn(
                "Type", options=["text", "numeric", "boolean", "datetime"], width="small"),
            "mean": st.column_config.NumberColumn("Mean", format="%.2f", width="small"),
            "std": st.column_config.NumberColumn("Std", format="%.2f", width="small"),
            "min": st.column_config.NumberColumn("Min", format="%.2f", width="small"),
            "max": st.column_config.NumberColumn("Max", format="%.2f", width="small"),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

    # ── Generation options ────────────────────────────────────────────────

    st.markdown("### Options")
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.number_input("Number of rows", min_value=10, max_value=100_000,
                                 value=500, step=100)
    with col2:
        inject_imperfections = st.checkbox("Inject realistic imperfections", value=True,
                                           help="Apply domain-specific nulls, outliers, and noise")

    # ── Generate ──────────────────────────────────────────────────────────

    if st.button("⚒️ Generate Dataset", type="primary", use_container_width=True):
        if not edited or len(edited) == 0:
            st.error("At least one column is required.")
            st.stop()

        with st.spinner("Generating dataset..."):
            try:
                df = generate_dataset(
                    kg=kg,
                    domain_name=st.session_state["active_domain"],
                    n_rows=n_rows,
                    custom_schema=edited,
                    inject_imperfections=inject_imperfections,
                )

                st.session_state["last_df"] = df
                st.success(f"Generated {len(df)} rows × {len(df.columns)} columns")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

# ── Preview & Export ──────────────────────────────────────────────────────

if "last_df" in st.session_state:
    df = st.session_state["last_df"]

    st.markdown("### Preview")
    with st.spinner(""):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        null_pct = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
        st.metric("Null %", f"{null_pct}%")

    # Summary stats
    with st.expander("Column Statistics"):
        st.dataframe(df.describe(include="all").round(2), use_container_width=True)

    st.markdown("### Export")
    col1, col2 = st.columns(2)

    with col1:
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            "📥 Download CSV",
            data=csv_buffer,
            file_name=f"datasmith_{st.session_state['active_domain']}_{n_rows}rows.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        json_buffer = io.BytesIO()
        df.to_json(json_buffer, orient="records", date_format="iso")
        json_buffer.seek(0)
        st.download_button(
            "📥 Download JSON",
            data=json_buffer,
            file_name=f"datasmith_{st.session_state['active_domain']}_{n_rows}rows.json",
            mime="application/json",
            use_container_width=True,
        )

    # Regenerate button
    st.divider()
    if st.button("🔄 Regenerate", use_container_width=True):
        del st.session_state["last_df"]
        st.rerun()
