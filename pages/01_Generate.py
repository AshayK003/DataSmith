"""Generate Page — the core of DataSmith.

Domain selection → schema editor → generate → preview → download.
"""

import io
import tempfile
from pathlib import Path

import streamlit as st

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.schema.crawler import SEED_DOMAINS
from datasmith.generation.engine import generate_dataset, export_csv, export_json

st.set_page_config(page_title="Generate — DataSmith", layout="centered")

st.markdown("<h1 style='text-align: center;'>⚒️ Generate Dataset</h1>",
            unsafe_allow_html=True)

# ── Init KG ───────────────────────────────────────────────────────────────

@st.cache_resource
def _get_kg():
    import os
    db_path = os.environ.get("DATASMITH_DB_PATH", "data/datasmith.db")
    db = Database(db_path)
    return KnowledgeGraph(db)

kg = _get_kg()

# ── Domain selector ───────────────────────────────────────────────────────

domains = list(SEED_DOMAINS.keys())
domain_name = st.selectbox("Domain", domains, format_func=lambda d: d.replace("-", " ").title())

if not domain_name:
    st.info("Select a domain to get started.")
    st.stop()

st.caption(SEED_DOMAINS.get(domain_name, ""))

# ── Load schema from KG ───────────────────────────────────────────────────

from datasmith.generation.engine import schema_from_kg, _generic_schema

kg_schema = schema_from_kg(kg, domain_name)
if not kg_schema:
    kg_schema = _generic_schema(domain_name)

# ── Schema editor ─────────────────────────────────────────────────────────

st.markdown("### Schema Editor")
st.caption("Edit column names, types, and parameters. Add or remove rows.")

# Convert schema to editor-friendly format
editor_data = []
for col in kg_schema:
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

# ── Generation options ────────────────────────────────────────────────────

st.markdown("### Options")
col1, col2 = st.columns(2)
with col1:
    n_rows = st.number_input("Number of rows", min_value=10, max_value=100_000,
                             value=500, step=100)
with col2:
    inject_imperfections = st.checkbox("Inject realistic imperfections", value=True,
                                       help="Apply domain-specific nulls, outliers, and noise")

# ── Generate ──────────────────────────────────────────────────────────────

if st.button("⚒️ Generate Dataset", type="primary", use_container_width=True):
    if not edited or len(edited) == 0:
        st.error("At least one column is required.")
        st.stop()

    with st.spinner("Generating dataset..."):
        try:
            df = generate_dataset(
                kg=kg,
                domain_name=domain_name,
                n_rows=n_rows,
                custom_schema=edited,
                inject_imperfections=inject_imperfections,
            )

            st.session_state["last_df"] = df
            st.session_state["last_domain"] = domain_name
            st.success(f"Generated {len(df)} rows × {len(df.columns)} columns")

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

# ── Preview & Export ──────────────────────────────────────────────────────

if "last_df" in st.session_state:
    df = st.session_state["last_df"]
    domain = st.session_state["last_domain"]

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
            file_name=f"datasmith_{domain}_{n_rows}rows.csv",
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
            file_name=f"datasmith_{domain}_{n_rows}rows.json",
            mime="application/json",
            use_container_width=True,
        )

    # Regenerate button
    st.divider()
    if st.button("🔄 Regenerate", use_container_width=True):
        del st.session_state["last_df"]
        del st.session_state["last_domain"]
        st.rerun()
