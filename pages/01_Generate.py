"""Generate Page -- the core of DataSmith.

Natural language -> schema -> editor -> generate -> preview -> download.
"""
import io
import os
import re
import time

import streamlit as st

from datasmith.core.database import Database
from datasmith.schema.knowledge_graph import KnowledgeGraph
from datasmith.llm.client import is_available as llm_available
from datasmith.llm.discovery import discover_schema
from datasmith.schema.crawler import SEED_DOMAINS
from datasmith.generation.engine import generate_dataset, schema_from_kg, get_generic_schema
from datasmith.ui import icons
from datasmith.ui.components import render_header

st.set_page_config(page_title="Generate — DataSmith", layout="centered")

render_header("generate")

st.markdown(f"<h1 style='text-align:center;'>{icons.SPARKLES} Generate Dataset</h1>",
            unsafe_allow_html=True)

# ── Keyboard shortcut ────────────────────────────────────────────────────

st.markdown("""<script>
if (!window._dsKeyHandler) {
    window._dsKeyHandler = function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            var btn = document.querySelector('[data-testid="stButton"][kind="primary"] button');
            if (btn) btn.click();
        }
    };
    document.addEventListener('keydown', window._dsKeyHandler);
}
</script>""", unsafe_allow_html=True)

# ── Session defaults ─────────────────────────────────────────────────────

if "n_rows" not in st.session_state:
    st.session_state["n_rows"] = 500
if "_resolved_schema" not in st.session_state:
    st.session_state["_resolved_schema"] = None
if "_schema_source" not in st.session_state:
    st.session_state["_schema_source"] = None
if "_resolved_schema_ver" not in st.session_state:
    st.session_state["_resolved_schema_ver"] = 0

# ── Init KG ──────────────────────────────────────────────────────────────


@st.cache_resource
def _get_kg():
    db_path = os.environ.get("DATASMITH_DB_PATH", "data/datasmith.db")
    db = Database(db_path)
    return KnowledgeGraph(db)


kg = _get_kg()

if "active_domain" not in st.session_state:
    st.session_state["active_domain"] = "custom"

# ── Schema Discovery ─────────────────────────────────────────────────────

st.markdown("## Describe Your Dataset")
st.caption(
    "Tell DataSmith what kind of data you need in plain English, or pick a domain below."
)

discover_tab, browse_tab = st.tabs(["Describe", "Browse Domains"])

with discover_tab:
    nl_input = st.text_input(
        "Describe your dataset",
        placeholder="e.g., e-commerce transactions with customer demographics and order history",
        help="Plain English description. The AI will suggest a matching schema.",
    )
    use_nl = st.button("Discover Schema", type="primary", use_container_width=True)

with browse_tab:
    domain_name = st.selectbox(
        "Domain",
        list(SEED_DOMAINS.keys()),
        format_func=lambda d: d.replace("-", " ").title(),
        label_visibility="collapsed",
    )
    use_domain = st.button("Load Domain", use_container_width=True)

# ── Resolve schema ───────────────────────────────────────────────────────

if use_nl and nl_input:
    if not llm_available():
        st.warning(
            "No LLM API key configured. "
            "Set `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `LLM_API_KEY` in "
            "Streamlit Cloud secrets for NL discovery. "
            "Using domain browser instead."
        )
    else:
        now = time.time()
        last = st.session_state.get("_last_llm_call", 0.0)
        if now - last < 5.0:
            st.warning("Please wait a few seconds before making another LLM request.")
            st.stop()
        st.session_state["_last_llm_call"] = now
        with st.spinner("Analyzing your description..."):
            resolved = discover_schema(kg, nl_input)
            if resolved:
                st.session_state["_resolved_schema"] = resolved
                st.session_state["_resolved_schema_ver"] += 1
                st.session_state["_schema_source"] = f"Description: _{nl_input}_"
                st.session_state["active_domain"] = nl_input[:40]
            else:
                st.info(
                    "Could not discover a schema from that description. "
                    "Try being more specific or pick a domain below."
                )

elif use_domain:
    with st.spinner("Loading domain..."):
        resolved = schema_from_kg(kg, domain_name)
        if not resolved:
            resolved = get_generic_schema(domain_name)
        st.session_state["_resolved_schema"] = resolved
        st.session_state["_resolved_schema_ver"] += 1
        st.session_state["_schema_source"] = f"Domain: **{domain_name.replace('-', ' ').title()}**"
        st.session_state["active_domain"] = domain_name

# Read persisted schema
resolved_schema = st.session_state["_resolved_schema"]
schema_source = st.session_state["_schema_source"]

# ── Schema editor ────────────────────────────────────────────────────────

if resolved_schema:
    st.markdown("## Schema Editor")
    if schema_source:
        st.caption(schema_source)
        if st.session_state["active_domain"] in SEED_DOMAINS:
            desc = SEED_DOMAINS[st.session_state["active_domain"]]
            domain_obj = kg.get_domain_by_name(st.session_state["active_domain"])
            n_datasets = len(kg.list_datasets(domain_id=domain_obj.id)) if domain_obj else 0
            meta = [f"_{desc}_"]
            if n_datasets:
                meta.append(f"{n_datasets} dataset{'s' if n_datasets != 1 else ''} crawled")
            else:
                meta.append("No datasets crawled yet")
            st.caption(" | ".join(meta))
    st.caption("Edit column names, types, and parameters. Add or remove rows.")

    import pandas as pd
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode

    # ── Build grid data from schema ───────────────────────────────────
    fresh_data = []
    for col in resolved_schema:
        dtype = col.get("data_type", "text").lower()
        display = "numeric" if dtype in ("numeric", "integer") else dtype
        fresh_data.append({
            "column_name": col.get("column_name", "col"),
            "data_type": display,
            "mean": col.get("mean", 50.0) if display == "numeric" else None,
            "std": col.get("std", 20.0) if display == "numeric" else None,
            "min": col.get("min", 0.0) if display == "numeric" else None,
            "max": col.get("max", 100.0) if display == "numeric" else None,
        })

    # ── Schema versioning — reset grid when schema changes ────────────
    CUR_TAG = str(len(resolved_schema)) + "$" + str(hash(tuple(
        c.get("column_name", "") + c.get("data_type", "") for c in resolved_schema
    )))
    if st.session_state.get("_grid_schema_tag") != CUR_TAG:
        st.session_state["_grid_data"] = pd.DataFrame(fresh_data)
        st.session_state["_grid_schema_tag"] = CUR_TAG

    # ── AG Grid ───────────────────────────────────────────────────────
    gb = GridOptionsBuilder.from_dataframe(st.session_state["_grid_data"])
    gb.configure_default_column(
        editable=True, filterable=True, sortable=True, resizable=True
    )
    # Type column → dropdown cell editor
    gb.configure_column(
        "data_type",
        header_name="Type",
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": ["text", "numeric", "boolean", "datetime"]},
        width=120,
    )
    # Numeric params
    for col_name in ["mean", "std", "min", "max"]:
        gb.configure_column(col_name, type=["numericColumn"], precision=2, width=100)

    grid_options = gb.build()

    response = AgGrid(
        st.session_state["_grid_data"],
        gridOptions=grid_options,
        update_on=["cellValueChanged", "selectionChanged"],
        data_return_mode=DataReturnMode.AS_INPUT,
        key="schema_grid",
        height=min(60 * len(fresh_data) + 80, 400),
        allow_unsafe_jscode=False,
    )

    # Capture edits back to session state
    if response.data is not None:
        st.session_state["_grid_data"] = response.data

    # ── Add / Delete rows ─────────────────────────────────────────────
    st.markdown("""<style>
    /* Delete Selected button → red */
    div[data-testid="column"]:nth-child(2) button {
        background-color: #dc2626 !important;
        color: #fff !important;
        border-color: #dc2626 !important;
    }
    div[data-testid="column"]:nth-child(2) button:hover {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
    }
    div[data-testid="column"]:nth-child(2) button:active {
        background-color: #991b1b !important;
    }
    /* Tooltip row below toolbar */
    .row-tooltip { font-size: 0.75rem; color: #888; margin: -8px 0 8px 16px; }
    </style>""", unsafe_allow_html=True)

    tool_c1, tool_c2 = st.columns(2)
    with tool_c1:
        if st.button("+ Add Row", key="add_row_btn", use_container_width=True):
            new_row = pd.DataFrame([{
                "column_name": "new_col",
                "data_type": "text",
                "mean": None, "std": None, "min": None, "max": None,
            }])
            st.session_state["_grid_data"] = pd.concat(
                [st.session_state["_grid_data"], new_row], ignore_index=True
            )
            st.rerun()
    with tool_c2:
        if st.button("− Delete Selected", use_container_width=True):
            # Read selected rows from the persisted grid component value
            raw = st.session_state.get("schema_grid", {})
            if isinstance(raw, dict):
                nodes = raw.get("nodes", [])
                sel_ids = [
                    int(n["id"]) for n in nodes
                    if n.get("isSelected") is True and "id" in n
                ]
                if sel_ids:
                    keep = st.session_state["_grid_data"][
                        ~st.session_state["_grid_data"].index.isin(sel_ids)
                    ]
                    st.session_state["_grid_data"] = keep.reset_index(drop=True)
                    st.rerun()
    st.markdown(
        """<div class="row-tooltip">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#888" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
        Click a row number to select it, then Delete Selected removes it
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Build edited schema from grid data ────────────────────────────
    grid_df = st.session_state["_grid_data"]
    edited = []
    for _, row in grid_df.iterrows():
        raw_name = str(row.get("column_name", "") or "")
        sanitized = re.sub(r'[^\w\- ]', '', raw_name)[:128]
        if not sanitized.strip():
            sanitized = "column"
        entry = {
            "column_name": sanitized,
            "data_type": row.get("data_type", "text"),
        }
        if entry["data_type"] == "numeric":
            entry["mean"] = row.get("mean") if pd.notna(row.get("mean")) else 50.0
            entry["std"] = row.get("std") if pd.notna(row.get("std")) else 20.0
            entry["min"] = row.get("min") if pd.notna(row.get("min")) else 0.0
            entry["max"] = row.get("max") if pd.notna(row.get("max")) else 100.0
        edited.append(entry)

    # ── Generation options ───────────────────────────────────────────────

    st.markdown("## Options")
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.number_input("Number of rows", min_value=10, max_value=100_000,
                                 value=st.session_state["n_rows"], step=100)
        st.session_state["n_rows"] = n_rows
    with col2:
        inject_imperfections = st.checkbox("Inject realistic imperfections", value=True,
                                           help="Apply domain-specific nulls, outliers, and noise")

    # ── Generate ─────────────────────────────────────────────────────────

    if st.button("Generate Dataset", type="primary", use_container_width=True):
        if not edited or len(edited) == 0:
            st.error("At least one column is required.")
            st.stop()

        # Validate numeric ranges
        for col in edited:
            if col.get("data_type") == "numeric":
                if col.get("min", 0) >= col.get("max", 100):
                    st.error(
                        f"Column '{col['column_name']}': min ({col['min']}) must be less than max ({col['max']})."
                    )
                    st.stop()
                if col.get("std", 1) < 0:
                    st.error(f"Column '{col['column_name']}': std cannot be negative.")
                    st.stop()

        with st.status("Generating dataset...", expanded=True) as status:
            try:
                status.update(
                    label=f"Generating {n_rows} rows from {len(edited)} columns...",
                    state="running",
                )
                df = generate_dataset(
                    kg=kg,
                    domain_name=st.session_state["active_domain"],
                    n_rows=n_rows,
                    custom_schema=edited,
                    inject_imperfections=inject_imperfections,
                )

                status.update(label="Done!", state="complete")
                st.session_state["last_df"] = df

            except Exception as e:
                status.update(label="Generation failed", state="error")
                st.error(f"Generation failed: {e}")
                st.stop()

# ── Preview & Export ─────────────────────────────────────────────────────

if "last_df" in st.session_state:
    df = st.session_state["last_df"]

    st.markdown("## Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        null_pct = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
        st.metric("Null %", f"{null_pct}%")

    # Imperfection report -- only when imperfections were injected
    if inject_imperfections:
        with st.expander("Imperfection Report", expanded=False):
            null_cols = df.isnull().sum()
            null_cols = null_cols[null_cols > 0].sort_values(ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Missing Values**")
                if len(null_cols) > 0:
                    for col_name, count in null_cols.head(5).items():
                        pct = count / len(df) * 100
                        st.markdown(f"- {col_name}: **{pct:.1f}%** null")
                else:
                    st.markdown("_No missing values in any column_")

            with c2:
                st.markdown("**Outliers (IQR)**")
                numeric_cols = df.select_dtypes(include="number").columns
                total_out = 0
                for col_name in numeric_cols:
                    q1, q3 = df[col_name].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_out = ((df[col_name] < lo) | (df[col_name] > hi)).sum()
                    if n_out > 0:
                        total_out += n_out
                        st.markdown(f"- {col_name}: **{n_out}** outliers ({n_out/len(df)*100:.1f}%)")
                if total_out == 0:
                    st.markdown("_No outliers detected_")

            st.caption(
                "Imperfections are injected based on the domain profile. "
                "Toggle 'Inject imperfections' above to disable."
            )

    with st.expander("Column Statistics"):
        st.dataframe(df.describe(include="all").round(2), use_container_width=True)

    st.markdown("## Export")

    def _sanitize_csv_formulas(frame: pd.DataFrame) -> pd.DataFrame:
        """Prefix cells starting with =, +, -, @ with ' to prevent formula injection."""
        safe = frame.copy()
        for col in safe.select_dtypes(include=["object", "string"]).columns:
            mask = safe[col].astype(str).str.match(r'^[=+\-@]')
            safe.loc[mask, col] = "'" + safe.loc[mask, col].astype(str)
        return safe

    col1, col2, col3 = st.columns(3)
    with col1:
        csv_buffer = io.BytesIO()
        _sanitize_csv_formulas(df).to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            "Download CSV",
            data=csv_buffer,
            file_name=f"datasmith_{st.session_state['active_domain']}_{st.session_state['n_rows']}rows.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        json_buffer = io.BytesIO()
        df.to_json(json_buffer, orient="records", date_format="iso")
        json_buffer.seek(0)
        st.download_button(
            "Download JSON",
            data=json_buffer,
            file_name=f"datasmith_{st.session_state['active_domain']}_{st.session_state['n_rows']}rows.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        st.download_button(
            "Download Parquet",
            data=parquet_buffer,
            file_name=f"datasmith_{st.session_state['active_domain']}_{st.session_state['n_rows']}rows.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Regenerate", use_container_width=True):
        del st.session_state["last_df"]
        st.rerun()
